from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

from coker.toolkits.spatial import Isometry3, Screw, SE3Adjoint


@dataclass
class Inertia:
    centre_of_mass: Isometry3
    mass: float
    moments: np.ndarray

    def as_matrix(self):
        inertial_component = np.array([
            [self.moments[0], self.moments[1], self.moments[2]],
            [self.moments[1], self.moments[3], self.moments[4]],
            [self.moments[2], self.moments[4], self.moments[5]]]
            ,dtype=float
        )
        mass_component = self.mass * np.eye(3,dtype=float)
        zeros = np.zeros((3, 3), dtype=float)
        return np.block([
            [inertial_component, zeros],
            [zeros, mass_component]
        ])


class JointType:
    @property
    def axes(self) -> List[Screw]:
        return []


class Weld(JointType):
    pass


class Free(JointType):

    @property
    def axes(self) -> List[Screw]:
        return [
            Screw.e_x(),
            Screw.e_y(),
            Screw.e_z(),
            Screw.w_x(),
            Screw.w_y(),
            Screw.w_z(),
        ]


class Revolute(JointType):
    def __init__(self, axis: Screw):
        self.axis = axis

    @property
    def axes(self) -> List[Screw]:
        return [self.axis]


class Planar(JointType):
    def __init__(self, *axes):
        self._axes = axes

    @property
    def axes(self) -> List[Screw]:
        return list(self._axes)


@dataclass
class RigidBody:
    WORLD = -1

    def __init__(self):
        self.joint_bases: List[List[Screw]] = []
        self.inertia: List[Inertia] = []
        self.parents: List[int] = []
        self.transforms: List[Isometry3] = []
        self.end_effectors: List[Tuple[int, Isometry3]] = []

        self._rest_transforms: List[Isometry3] = []

    def add_link(self, parent: int, at: Isometry3, joint, inertia) -> int:
        idx = len(self.parents)

        assert 0 <= parent < idx or parent == self.WORLD

        self.parents.append(parent)
        self.transforms.append(at)
        self.joint_bases.append(joint.axes)
        self.inertia.append(inertia)

        if parent != self.WORLD:
            t = self._rest_transforms[parent] @ at
        else:
            t = at
        self._rest_transforms.append(t)

        return idx

    def add_effector(self, parent: int, at: Isometry3):
        assert RigidBody.WORLD < parent < len(self.parents)
        idx = len(self.end_effectors)
        self.end_effectors.append((parent, at))
        return idx

    def total_joints(self):
        return sum([len(j) for j in self.joint_bases])

    def potential_energy(self, angles, gravity_vector):

        if len(angles.shape) != 1:
            assert angles.shape[0] == self.total_joints()
            q = np.reshape(angles, newshape=(self.total_joints(),))
        else:
            assert angles.shape == (self.total_joints(),)
            q = angles
        joint_transforms = self._get_joint_transforms(q)
        joint_xforms = self._accumulate_joint_xforms(q, joint_transforms)
        origin = np.array([0, 0, 0])
        total_energy = 0
        for i, inertia in enumerate(self.inertia):
            xform = joint_xforms[i] @ self._rest_transforms[i] @ inertia.centre_of_mass
            point = xform @ origin

            total_energy -= inertia.mass * np.dot(gravity_vector, point)

        return total_energy

    def _accumulate_joint_xforms(self, q, joint_transforms):

        accumulated_xform = []
        for i, parent in enumerate(self.parents):
            t = joint_transforms[i]
            while parent != self.WORLD:
                t = joint_transforms[parent] @ t
                parent = self.parents[parent]

            accumulated_xform.append(t)
        return accumulated_xform

    def _get_joint_transforms(self, angles) -> List[Isometry3]:
        joint_idx = 0
        transforms = []
        for link, bases in enumerate(self.joint_bases):
            g_theta = Isometry3.identity()
            for basis in bases:
                xform = basis.exp(angles[joint_idx])

                g_theta = g_theta @ xform
                joint_idx += 1
            transforms.append(g_theta)
        return transforms

    def _get_absolute_joint_xform(self, angles):
        joint_idx = 0
        transforms = []
        for link, bases in enumerate(self.joint_bases):
            g_theta = Isometry3.identity()
            adjoint = SE3Adjoint(self._rest_transforms[link])
            for basis in bases:
                zeta = adjoint.apply(basis)
                g_theta = g_theta @ zeta.exp(angles[joint_idx])
                joint_idx += 1
            transforms.append(g_theta)
        return transforms

    def joint_locations(self, angles, effector) -> List[Isometry3]:
        xforms = self._get_absolute_joint_xform(angles)
        link, _ = self.end_effectors[effector]
        transforms = [
            self._rest_transforms[i] @ xforms[i] for i in self.get_dependent_joints(link)
        ]
        return transforms

    def forward_kinematics(self, angles) -> List[Isometry3]:
        abs_xforms = self._get_absolute_joint_xform(angles)
        transforms = self._accumulate_joint_xforms(angles, abs_xforms)

        out = [
            transforms[parent] @ self._rest_transforms[parent] @ transform
            for parent, transform in self.end_effectors
        ]

        return out

    def get_dependent_joints(self, parent_link: int):
        result = list()
        while parent_link != self.WORLD:
            result.append(parent_link)
            parent_link = self.parents[parent_link]
        return reversed(result)

    def spatial_manipulator_jacobian(self, angles):
        return np.concatenate(
            [self.spatial_single_manipulator_jacobian(angles, i) for i, _ in enumerate(self.end_effectors)]
        )

    def spatial_single_manipulator_jacobian(self, angles, end_effector):
        abs_xforms = self._get_absolute_joint_xform(angles)
        xforms = self._accumulate_joint_xforms(angles, abs_xforms)

        effector_parent, _ = self.end_effectors[end_effector]
        dependent_joints = self.get_dependent_joints(effector_parent)

        columns = []

        for joint_idx, bases in enumerate(self.joint_bases):
            if joint_idx in dependent_joints:
                parent = self.parents[joint_idx]
                for zeta in bases:
                    zeta_prime = SE3Adjoint(self._rest_transforms[joint_idx]).apply(zeta)
                    if parent != self.WORLD:
                        zeta_prime = SE3Adjoint(xforms[parent]).apply(zeta_prime)
                    columns.append(
                        np.reshape(zeta_prime.to_array(), newshape=(6, 1))
                    )
            else:
                zeta_prime = np.zeros((6, len(bases)))
                columns.append(zeta_prime)

        return np.concatenate(columns, axis=1)

    def body_manipulator_jacobian(self, angles):
        g_list = self.forward_kinematics(angles)
        ad_g_inv = [
            SE3Adjoint(g).inverse().as_matrix()
            for i, g in enumerate(g_list)
        ]
        columns = [
            ad_g @ self.spatial_single_manipulator_jacobian(angles, i)
            for i, ad_g in enumerate(ad_g_inv)
        ]
        return np.concatenate(columns, axis=1)
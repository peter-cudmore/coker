from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

from coker.toolkits.spatial import (
    Isometry3,
    Screw,
    SE3Adjoint,
    SE3CoAdjoint,
    hat,
    se3Adjoint,
)


@dataclass
class Inertia:
    centre_of_mass: Isometry3
    mass: float
    moments: np.ndarray

    @staticmethod
    def zero() -> "Inertia":
        return Inertia(Isometry3.identity(), 0, np.zeros((6,)))

    def as_matrix(self):
        inertial_component = np.array(
            [
                [self.moments[0], self.moments[1], self.moments[2]],
                [self.moments[1], self.moments[3], self.moments[4]],
                [self.moments[2], self.moments[4], self.moments[5]],
            ],
        )
        mass_component = self.mass * np.eye(
            3,
        )
        zeros = np.zeros(
            (3, 3),
        )
        return np.block([[inertial_component, zeros], [zeros, mass_component]])

    def as_matrix_origin(self):
        p_hat = hat(self.centre_of_mass.translation)
        assert self.centre_of_mass.rotation.angle == 0
        m = self.as_matrix()
        m[3:6, 0:3] = self.mass * p_hat
        m[0:3, 3:6] = -self.mass * p_hat
        return m


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
        """
        - g_i(q) is the position of the ith link
        - lam(i) is the parent of the ith link

        transforms:
            transform X_{lam(i), i} is the location of the origin of the ith
            link in lam(i) coordinates, so that at rest
            g_i(0) = prod_{j=0, i}(X_{lam(j), j} = X_{-1, 0} @ X_{0, 1} @ X_{lam(1), 1}) @ ...

        rest transforms:
            the rest transform is g_i(0) s.t. g_i(q) = exp(zeta q) g_i(0)


        """
        self.joints: List[JointType] = []

        self.inertia: List[Inertia] = []
        self.parents: List[int] = []
        self.transforms: List[Isometry3] = []
        self.end_effectors: List[Tuple[int, Isometry3]] = []
        self._rest_transforms: List[Isometry3] = []

    @property
    def joint_bases(self) -> List[List[Screw]]:
        return [j.axes for j in self.joints]

    def joint_global_basis(self, joint_index):
        bases = self.joint_bases[joint_index]

        adjoint = SE3Adjoint(self._rest_transforms[joint_index])
        return [adjoint.apply(b) for b in bases]

    def add_link(
        self, parent: int, at: Isometry3, joint: JointType, inertia: Inertia
    ) -> int:
        idx = len(self.parents)

        assert 0 <= parent < idx or parent == self.WORLD

        self.parents.append(parent)
        self.transforms.append(at)
        self.joints.append(joint)
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
            q = np.reshape(angles, shape=(self.total_joints(),))
        else:
            assert angles.shape == (self.total_joints(),)
            q = angles
        joint_transforms = self._get_absolute_joint_xform(q)
        joint_xforms = self._accumulate_joint_xforms(joint_transforms)
        origin = np.array([0, 0, 0])
        total_energy = 0
        for i, inertia in enumerate(self.inertia):
            xform = (
                joint_xforms[i]
                @ self._rest_transforms[i]
                @ inertia.centre_of_mass
            )
            point = xform @ origin

            total_energy -= inertia.mass * np.dot(gravity_vector, point)

        return total_energy

    def _accumulate_joint_xforms(self, joint_transforms):

        accumulated_xform = []
        for i, parent in enumerate(self.parents):

            if parent != self.WORLD:
                t = accumulated_xform[parent] @ joint_transforms[i]
            else:
                t = joint_transforms[i]

            accumulated_xform.append(t)

        return accumulated_xform

    def _get_joint_local_bases(self):
        result = []
        for bases in self.joint_bases:
            for basis in bases:
                result.append(basis)
        return result

    def _get_joint_global_bases(self):
        result = []
        for rest_transform, bases in zip(
            self._rest_transforms, self.joint_bases
        ):
            for basis in bases:
                result.append(SE3Adjoint(rest_transform).apply(basis))
        return result

    def _get_joint_dependency_map(self):
        result = {}
        joint_idx = 0
        for i, bases in enumerate(self.joint_bases):
            result[i] = []
            for _ in bases:
                result[i].append(joint_idx)
                joint_idx += 1

        return result

    def get_dependant_joints(self, effector_index):
        joints = []

        parent, _ = self.end_effectors[effector_index]
        joint_map = self._get_joint_dependency_map()
        while parent != self.WORLD:
            joints += joint_map[parent]
            parent = self.parents[parent]
        return sorted(joints)

    def _get_joint_transforms(self, angles) -> List[Isometry3]:
        """Computes `g_si = Prod_j(exp(zeta_i q_i))`
        For each of the j angles.
        Zeta is the joint basis with respect to the joint origin.

        """

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
        r"""Express each g_j = Prod_i exp(theta_i zeta_i)
        in coordinates w.r.t the world

        The position of each joint $i$ is given by
        \prod_{j <= i} g_{j-1; j} exp(zeta_jq_j)

        where g_{i-1; j} is the transfrom of the joint origin to the parent frame
        zeta_j is the joint basis in the joint frame
        q_j is the joint angle

        want to move this to \Prod [exp(z_jq_j] g_i(0)

        # g_{j-1; j} = g_{j-1; 0} g_{0; j} = g_{j-1}^{-1} g_j
        so
            prod = g_0 exp(zeta_0) g^{-1}_0 * g_1 exp(zeta_1) g_1^{-1} ...
            i.e. z_j = Adj_{g_j}[zeta_j]

        """
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

    def joint_transforms(self, angles, effector=None) -> List[Isometry3]:

        joint_index = 0
        transforms = []
        for parent, xform, bases in zip(
            self.parents, self.transforms, self.joint_bases
        ):
            g = transforms[parent] @ xform if parent != self.WORLD else xform
            for basis in bases:
                g = g @ basis.exp(angles[joint_index])
                joint_index += 1

            transforms.append(g)

        if effector is None:
            return transforms

        out = []
        parent, _ = self.end_effectors[effector]

        while parent != self.WORLD:
            out.append(transforms[parent])
            parent = self.parents[parent]
        return list(reversed(out))

    def forward_kinematics(self, angles) -> List[Isometry3]:
        xforms = self.joint_transforms(angles)
        return [xforms[parent] @ xform for parent, xform in self.end_effectors]

    def get_dependent_links(self, parent_link: int):
        result = list()
        while parent_link != self.WORLD:
            result.append(parent_link)
            parent_link = self.parents[parent_link]
        return reversed(result)

    def spatial_manipulator_jacobian(self, angles):
        return [
            self.spatial_single_manipulator_jacobian(angles, i)
            for i, _ in enumerate(self.end_effectors)
        ]

    def spatial_single_manipulator_jacobian(self, angles, end_effector):
        abs_xforms = self._get_absolute_joint_xform(angles)
        xforms = self._accumulate_joint_xforms(abs_xforms)

        effector_parent, _ = self.end_effectors[end_effector]
        dependent_link = set(self.get_dependent_links(effector_parent))

        columns = []

        for link_idx, bases in enumerate(self.joint_bases):
            if link_idx in dependent_link:
                parent = self.parents[link_idx]
                for zeta in bases:
                    zeta_prime = SE3Adjoint(
                        self._rest_transforms[link_idx]
                    ).apply(zeta)
                    if parent != self.WORLD:
                        zeta_prime = SE3Adjoint(xforms[parent]).apply(
                            zeta_prime
                        )
                    columns.append(
                        np.reshape(zeta_prime.to_array(), shape=(6, 1))
                    )
            else:
                zeta_prime = np.zeros((6, len(bases)))
                columns.append(zeta_prime)

        return np.concatenate(columns, axis=1)

    def body_manipulator_jacobian(self, angles):
        g_list = self.forward_kinematics(angles)
        ad_g_inv = [SE3Adjoint(g).inverse().as_matrix() for g in g_list]
        columns = [
            ad_g @ self.spatial_single_manipulator_jacobian(angles, i)
            for i, ad_g in enumerate(ad_g_inv)
        ]
        return columns

    def inverse_dynamics(
        self,
        angles,
        angle_rates,
        angle_accel,
        gravity: np.ndarray,
        end_effector_forces: Dict[int, Screw] = None,
    ) -> np.ndarray:
        """Compute torques for forces"""
        transforms = []
        velocities = []
        accelerations = []
        forces = []
        inverse_bases = []
        a_g = np.concatenate([np.zeros((3,)), gravity])

        x_forms = self._get_joint_transforms(angles)

        def spatial_cross(a, b):
            y_1 = np.cross(a[0:3], b[0:3])
            y_2 = np.cross(a[3:6], b[0:3]) + np.cross(a[0:3], b[3:6])
            return np.concatenate([y_1, y_2])

        def spatial_cross_star(a, b):
            y_1 = np.cross(a[0:3], b[0:3]) + np.cross(a[3:6], b[3:6])
            y_2 = np.cross(a[0:3], b[3:6])
            return np.concatenate([y_1, y_2])

        joint_idx = 0
        for link, bases in enumerate(self.joint_bases):
            parent = self.parents[link]
            s = []

            # g_i is the transform of the parent frame into the child frame
            # so that for a linear chain g_{01} @ g_{12} @ ... @ g_n
            # is the same as g_{0n}
            g_i = self.transforms[link] @ x_forms[link]

            vJ = np.zeros((6,))
            aJ = np.zeros((6,))
            for basis in bases:
                s_i = basis.to_array()
                s.append(s_i.reshape((6, 1)))
                vJ += s_i * angle_rates[joint_idx]
                aJ += s_i * angle_accel[joint_idx]
                joint_idx += 1

            adj = SE3Adjoint(g_i.inverse()).as_matrix()
            if parent != self.WORLD:
                v_i = adj @ velocities[parent] + vJ
                s_dot = spatial_cross(v_i, vJ)
                a_i = adj @ accelerations[parent] + aJ + s_dot
            else:
                v_i = vJ
                a_i = -adj @ a_g + aJ

            # Ineratia at center of mass,
            # need to transfrom to origin
            com_adj = SE3Adjoint(self.inertia[link].centre_of_mass.inverse())

            iota = (
                com_adj.as_matrix().transpose()
                @ self.inertia[link].as_matrix()
                @ com_adj.as_matrix()
            )
            f_i = iota @ a_i + spatial_cross_star(v_i, (iota @ v_i))
            transforms.append(g_i)
            velocities.append(v_i)
            accelerations.append(a_i)
            forces.append(f_i)
            if s:
                s_matrix = np.concatenate(s, axis=1)
                inverse_bases.append(s_matrix)
            else:
                inverse_bases.append(None)
        torques = []

        if end_effector_forces is not None:
            fk = self.forward_kinematics(angles)
            if isinstance(end_effector_forces, dict):
                iterator = end_effector_forces.items()
            else:
                iterator = enumerate(end_effector_forces)
            for idx, force in iterator:
                parent, _ = self.end_effectors[idx]
                forces[parent] -= SE3CoAdjoint(fk[idx]).as_matrix() @ force

        for i, S_i in reversed(list(enumerate(inverse_bases))):
            parent = self.parents[i]
            f_i = forces[i]
            if parent != self.WORLD:
                t = SE3CoAdjoint(transforms[i].inverse())
                parent_force = t.as_matrix() @ f_i
                forces[parent] = forces[parent] + parent_force

            if S_i is None:
                continue
            torque = S_i.T @ f_i
            torques = [torque] + torques
        result = np.concatenate(torques)
        return result

    def _get_link_com_jacobians(self, angles):
        joint_transforms = self._get_absolute_joint_xform(
            angles
        )  # exp(zeta_hat)
        joint_map = self._get_joint_dependency_map()  # link: [joint idx]
        joint_bases = self._get_joint_global_bases()  # zeta_j
        total_angles = self.total_joints()
        jacobians = []
        for i, inertia in enumerate(self.inertia):
            g = inertia.centre_of_mass @ self._rest_transforms[i]  # g_sli
            jacobian = [np.zeros((6, 1))] * total_angles
            next_idx = i

            while next_idx != self.WORLD:
                g = joint_transforms[next_idx] @ g
                for joint_idx in reversed(joint_map[next_idx]):
                    zeta_j = joint_bases[joint_idx]
                    adj_inv = SE3Adjoint(g).inverse()
                    zeta_j_dagger = adj_inv.apply(zeta_j)
                    jacobian[joint_idx] = np.reshape(
                        zeta_j_dagger.to_array(), shape=(6, 1)
                    )

                next_idx = self.parents[next_idx]

            jac = np.concatenate(jacobian, axis=1)
            jacobians.append(jac)

        return jacobians

    def mass_matrix(self, angles):
        jacobians = self._get_link_com_jacobians(angles)
        n = self.total_joints()
        mass_matrix = np.zeros((n, n))
        for j, inertia in zip(jacobians, self.inertia):

            mass_matrix += j.T @ inertia.as_matrix() @ j

        return mass_matrix

    def add_body(self, body: "RigidBody", at: Isometry3, parent: int):
        links = []

        iterator = zip(
            body.parents, body.joints, body.transforms, body.inertia
        )
        for i, (p_body, joint, xform, inertia) in enumerate(iterator):
            idx = len(self.parents)

            self.parents.append(
                parent if p_body == self.WORLD else links[p_body]
            )
            self.inertia.append(inertia)

            p_actual = self.parents[idx]
            if p_body == body.WORLD:
                body_xform = at @ xform

            else:
                body_xform = xform

            self.joints.append(joint)

            rest_xform = (
                self._rest_transforms[p_actual] @ body_xform
                if p_actual != self.WORLD
                else body_xform
            )
            self.transforms.append(body_xform)
            self._rest_transforms.append(rest_xform)

            links.append(idx)

        effectors = [
            self.add_effector(links[p_body], xform)
            for p_body, xform in body.end_effectors
        ]

        return links, effectors

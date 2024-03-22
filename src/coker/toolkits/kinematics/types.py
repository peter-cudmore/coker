import dataclasses
from dataclasses import dataclass
from typing import Tuple, Optional, List, Type
from dataclass_wizard import JSONWizard, LoadMixin
from itertools import accumulate

from coker.toolkits.spatial import Isometry3, Rotation3

import numpy as np
Vec3 = Tuple[float, float, float]
Vec6 = Tuple[float, float, float, float, float, float]


class JointType:
    def basis(self) -> List[Vec6]:
        raise NotImplementedError

    def labels(self) -> List[str]:
        raise NotImplementedError

    def dimension(self):
        return len(self.basis())


class ScrewJoint(JointType):
    def __init__(self, axis: Vec6):
        self.axis = axis

    def labels(self):
        return [r'angle']

    def basis(self):
        return [self.axis]


class Hinge(JointType):
    def __init__(self, axis: Vec3):
        self.axis = axis

    def labels(self):
        return [r"angle"]

    def basis(self):
        return [(self.axis[0], self.axis[1], self.axis[2], 0, 0, 0)]


class Free(JointType):
    def labels(self):
        return ["wx", "wy", "wz" "x", "y", "z"]

    def basis(self) -> List[Vec6]:
        return [tuple(1.0 if i == j else 0.0 for j in range(6)) for i in range(6)]


def joint_from_basis(j: Optional[Vec6]):
    if j is None:
        return Free()
    else:
        return ScrewJoint(j)


@dataclass
class SO3(JSONWizard):
    translation: Vec3
    rotation: Vec3

    def clone(self):
        return dataclasses.replace(self)

    def to_isometry(self) -> Isometry3:
        return Isometry3(
            rotation=Rotation3.from_vector(self.rotation),
            translation=np.array(self.translation)
        )


@dataclass
class KinematicParameters(JSONWizard):
    centre_of_mass: SO3
    mass: float
    inertia: Vec6

    def clone(self):
        return dataclasses.replace(self)


@dataclass
class KinematicTree(JSONWizard, LoadMixin):
    parents: List[Optional[int]]
    link_names: List[str]
    joint_basis: List[Optional[Vec6]]
    kinematics: List[KinematicParameters]
    transforms: List[SO3]

    end_effector_names: List[str]
    end_effector_parents: List[int]
    end_effector_transform: List[SO3]

    @property
    def joint_labels(self):
        return [
            f"{self.link_names[i]}/{label}"
            for i, joint in enumerate(self.joints)
            for label in joint.labels()
        ]

    @property
    def joints(self):
        return [joint_from_basis(j) for j in self.joint_basis]

    @property
    def degrees_of_freedom(self) -> int:
        return sum(
            self.joint_size(j) for j in range(len(self.joint_basis))
        )

    def clone(self) -> 'KinematicTree':
        return dataclasses.replace(self)

    def joint_size(self, joint: int):
        if self.joint_basis[joint] is None:
            return 6
        else:
            return 1

    def joint_sizes_and_offsets(self) -> Tuple[List[int], List[int]]:
        sizes = [
            self.joint_size(j) for j in range(len(self.joint_basis))
        ]
        offsets = [0, *list(accumulate(sizes[:-1]))]
        return sizes, offsets

#    def add_body(self, name: str,
#                 parent_location: SO3,
#                 kinematics: KinematicParameters,
#                 joint_type:
#                 ):


AnchorSpec = Tuple[int, int, SO3]
BodyAnchor = Optional[AnchorSpec]


@dataclass
class CompositeBodyModel(JSONWizard):
    name: str
    body_names: List[str]

    body_components: List[int]

    body_anchors: List[BodyAnchor]

    components: List[KinematicTree]

    def end_effectors(self) -> List[str]:
        results = []
        for name, i in zip(self.body_names, self.body_components):
            results += [
                f"{name}/{effector}"
                for effector in self.components[i].end_effector_names
            ]

        return results

import numpy as np
from typing import Callable, Tuple, List, Dict, Optional
from coker.toolkits.kinematics.types import KinematicTree, CompositeBodyModel, JointType

from coker.toolkits.spatial import Screw, Isometry3, SE3Adjoint, se3Adjoint, SE3CoAdjoint, UnitQuaternion
from coker.toolkits.kinematics.types import KinematicParameters


def forward_kinematics(tree: KinematicTree,
                       args: Optional[List] = None,
                       ) -> Tuple[List[Callable], List[Callable]]:

    iterator = zip(
        tree.link_names,
        tree.joints,
        tree.parents,
            tree.transforms
    )
    transforms = []
    origin = np.array([0, 0, 0], dtype=np.float32)
    bases = []
    index = 0
    for name, joint, parent, transform in iterator:
        iso = transform.to_isometry()

        if parent is not None:
            t = transforms[parent] @ iso
        else:
            t = iso

        transforms.append(t)
        point = t @ origin
        ad_p = SE3Adjoint(Isometry3(translation=point))
        this_basis = []
        for basis in joint.basis():
            v = ad_p.apply(Screw.from_tuple(*basis))
            this_basis.append(
                (index, v)
            )
            index += 1

    def f_joints(q):
        joints = []

        g = []
        for base_list in bases:
            g_i = Isometry3.identity()
            for idx, v in base_list:
                g_i @= v.exp(q[idx])
            g.append(g_i)

        for parent, xform in zip(tree.parents, transforms):

            next_joint = parent

            g_angles = np.eye(4)

            while next_joint is not None:

                g_angles = g[next_joint] @ g_angles
                next_joint = tree.parents[next_joint]

            t = g_angles @ xform
            joints.append(t)
        return joints

    return f_joints, None
    end_effectors = []
    for (parent, xform) in zip(tree.end_effector_parents,
                               tree.end_effector_transform):
        iso = Isometry3(translation=xform.translation, rotation=xform.rotation)
        g_angles = sp.Matrix.eye(4)
        next_joint = parent

        while next_joint is not None:
            g_angles = g[next_joint] @ g_angles
            next_joint = tree.parents[next_joint]

        t = g_angles @ (transforms[parent] @ iso).to_homogenous()

        if lambdify:
            f = sp.lambdify(angles, t, cse=True)
            end_effectors.append(f)
        else:
            end_effectors.append(t)
    # should be a 4 x 4 x 6 symbolic
    return joints, end_effectors


def flatten_model(model: CompositeBodyModel) -> KinematicTree:
    work_set = set(range(len(model.body_names)))

    link_mapping: Dict[Tuple[int, int], int] = {}

    tree = None

    while work_set:
        completed_node = None
        for body_idx in work_set:
            if not model.body_anchors[body_idx]:
                assert tree is None, "Error, Already found a root"
                # got the root element
                # add the corresponding component 1-for-1
                # directly to the tree
                # then add the link mapping
                c_idx = model.body_components[body_idx]
                tree = model.components[c_idx].clone()
                tree.link_names = [
                    f"{model.body_names[body_idx]}/{name}"
                    for name in tree.link_names
                ]
                link_mapping.update({
                    (0, i): i for i in range(len(tree.link_names))
                })
                completed_node = body_idx
                break

            parent_body, parent_link, xform = model.body_anchors[body_idx]

            try:
                new_parent = link_mapping[(parent_body, parent_link)]
            except KeyError:
                # haven't hit this one yet
                continue

            component = model.components[model.body_components[body_idx]]

            for old_link, old_parent in enumerate(component.parents):
                new_link = len(tree.parents)
                if old_parent is not None:
                    tree.parents.append(link_mapping[(body_idx, old_parent)])
                else:
                    tree.parents.append(new_parent)

                link_mapping[(body_idx, old_link)] = new_link

            tree.joint_basis += [j for j in component.joint_basis]
            tree.kinematics += [k for k in component.kinematics]
            tree.transforms += [xform, *component.transforms[1:]]
            tree.link_names += [
                f"{model.body_names[body_idx]}/{name}"
                for name in component.link_names
            ]

            tree.end_effector_names += [
                f"{model.body_names[body_idx]}/{name}"
                for name in component.end_effector_names
            ]
            tree.end_effector_transform += [
                t for t in component.end_effector_transform
            ]
            tree.end_effector_parents += [
                link_mapping[(body_idx, p)] for p in component.end_effector_parents
            ]
            completed_node = body_idx
            break

        assert completed_node is not None
        work_set.remove(completed_node)

    return tree


def generalised_inertial_matrix(params: KinematicParameters):
    from robot_toolkit.symbolic import hat

    r, p = Isometry3(
        translation=params.centre_of_mass.translation,
        rotation=params.centre_of_mass.rotation
    ).as_rotation_vector()
    m = params.mass
    p_hat = hat(p)
    ixx, ixy, ixz, iyy, iyz, izz = params.inertia
    inertial_matrix = sp.Matrix(
        [[ixx, ixy, izz],
         [ixy, iyy, iyz],
         [ixz, iyz, izz]]
    )
    return sp.Matrix(sp.BlockMatrix(
      [[r @ inertial_matrix @ r.T + m * p_hat.T @ p_hat, m * p_hat],
       [m * p_hat.T, m * sp.eye(3)]]
    ))


# tree -> model
#
#  for each joint, need to:
#       - assign index or slice from the arguments
#         (offset from first joint)
#       - be able to generate joint jacobian
#       - be able to generate joint transform
#       - be able to know if it is driven (and if so by which signal)
#         or if it's free

def inverse_dynamics(
        tree: KinematicTree,
        external_forces: List
):

    n = len(tree.joint_basis)
    v = [PluckerVector.zero()] * n
    a = [PluckerVector.zero()] * n
    f = [PluckerVector.zero()] * n
    S = [PluckerVector.zero()] * n
    Xup = [Isometry3.identity()] * n

    sizes, offsets = tree.joint_sizes_and_offsets()

    q = sp.symbols("," .join(f"q_{i}" for i in range(n)))
    dq = sp.symbols("," .join(f"dq_{i}" for i in range(n)))
    ddq = sp.symbols("," .join(f"ddq_{i}" for i in range(n)))

    dq_array = sp.Matrix(dq)
    ddq_array = sp.Matrix(ddq)

    for i, (joint_basis, parent, xform) in enumerate(zip(tree.joint_basis,
                                          tree.parents,
                                          tree.transforms)):

        joint_slice = slice(offsets[i], offsets[i] + sizes[i])

        XJ, S_i = get_transform_and_jacobian_by_joint_type(joint_basis, *q[joint_slice])
        S[i] = S_i
        dq_i = sp.Matrix(dq_array[joint_slice])
        ddq_i = sp.Matrix(ddq_array[joint_slice])
        vJ = PluckerVector.from_vec6(S[i] @ dq_i)   # vec 6
        aJ = PluckerVector.from_vec6(S[i] @ ddq_i)

        Xup[i] = XJ @ Isometry3(translation=xform.translation, rotation=xform.rotation)

        if parent is not None:
            Ad_t = SE3Adjoint(Xup[i].inv())
            v[i] = Ad_t @ v[parent]
            a[i] = Ad_t @ a[parent]
        else:
            a[i] = PluckerVector(translation=[0, 0, -9.8], rotation=[0, 0, 0])

        v[i] = v[i] + vJ
        ad_V = se3Adjoint(vJ)
        a[i] = a[i] + aJ + ad_V.apply(vJ)
        intertial_matrix = generalised_inertial_matrix(tree.kinematics[i])
        f[i] = intertial_matrix @ a[i] + ad_V.transpose() @ (intertial_matrix @ v[i])

    torques = [PluckerVector.zero()] * n
    for i in reversed(range(n)):
        tau = f[i].to_vec6().dot(S[i])
        torques[i] = tau
        parent = tree.parents[i]
        if parent is not None:
            f[parent] = f[parent] + SE3CoAdjoint(Xup[i]).apply(f[i])

    return torques


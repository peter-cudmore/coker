import operator
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

from coker.toolkits.spatial import (
    Isometry3,
    Screw,
    SE3Adjoint,
    SE3CoAdjoint,
    hat,
)


@dataclass
class Inertia:
    """Mass properties expressed in a link's centre-of-mass frame.

    Args:
        centre_of_mass: Pose mapping centre-of-mass coordinates into the link
            frame. Translation uses metres.
        mass: Mass in kilograms.
        moments: Array of shape ``(6,)`` containing
            ``[Ixx, Ixy, Ixz, Iyy, Iyz, Izz]`` in kilogram metres squared.

    Examples:
        >>> from coker.toolkits.kinematics import Inertia
        >>> Inertia.zero().as_matrix().shape
        (6, 6)
    """

    centre_of_mass: Isometry3
    mass: float
    moments: np.ndarray

    @staticmethod
    def zero() -> "Inertia":
        """Return zero mass and moments at the link origin.

        Returns:
            Inertia: Massless inertia with an identity centre-of-mass pose.
        """
        return Inertia(Isometry3.identity(), 0, np.zeros((6,)))

    def as_matrix(self):
        """Return the centre-of-mass spatial inertia.

        Returns:
            numpy.ndarray: Shape ``(6, 6)``, ordered angular then linear.
                The diagonal blocks are the moment tensor and ``mass * I``.
        """
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
        """Return the origin-coupled inertia for an unrotated mass frame.

        This inserts the translation cross-coupling blocks into
        :meth:`as_matrix`; it does not add a parallel-axis term to its
        rotational block. Use :meth:`RigidBody.inverse_dynamics` for the full
        centre-of-mass pose transformation used by that calculation.

        Returns:
            numpy.ndarray: Shape ``(6, 6)``, angular then linear ordering.

        Raises:
            ValueError: The centre-of-mass frame has nonzero rotation.
        """
        if self.centre_of_mass.rotation.angle != 0:
            raise ValueError("The centre-of-mass frame must have zero rotation")
        p_hat = hat(self.centre_of_mass.translation)
        m = self.as_matrix()
        m[3:6, 0:3] = self.mass * p_hat
        m[0:3, 3:6] = -self.mass * p_hat
        return m


class JointType:
    """Joint interface exposing ordered local motion screws.

    The base implementation has no coordinates, like :class:`Weld`.
    Subclasses provide :attr:`axes` in the child link's rest frame.

    Examples:
        >>> from coker.toolkits.kinematics import JointType
        >>> JointType().axes
        []
    """

    @property
    def axes(self) -> List[Screw]:
        """List[Screw]: Local screws in joint-coordinate order."""
        return []


class Weld(JointType):
    """Fixed joint with no coordinates.

    Examples:
        >>> from coker.toolkits.kinematics import Weld
        >>> Weld().axes
        []
    """

    pass


class Free(JointType):
    """Six-coordinate joint with ordered translation and rotation screws.

    Coordinates are ``[x, y, z, rx, ry, rz]``: translations in metres followed
    by rotations in radians. The pose is the product of these six screw
    exponentials in that order, not an exponential of their sum.

    Examples:
        >>> from coker.toolkits.kinematics import Free
        >>> len(Free().axes)
        6
    """

    @property
    def axes(self) -> List[Screw]:
        """List[Screw]: Translation x/y/z, then rotation x/y/z."""
        return [
            Screw.e_x(),
            Screw.e_y(),
            Screw.e_z(),
            Screw.w_x(),
            Screw.w_y(),
            Screw.w_z(),
        ]


class Revolute(JointType):
    """Single-coordinate joint driven by a local screw.

    Despite the name, a pure translation screw also defines a sliding joint;
    there is no exported ``Prismatic`` class.

    Examples:
        >>> from coker.toolkits.kinematics import Revolute
        >>> from coker.toolkits.spatial import Screw
        >>> len(Revolute(Screw.w_z()).axes)
        1
    """

    def __init__(self, axis: Screw):
        """Create a joint with one local axis.

        Args:
            axis: Motion screw in the child link's rest frame. A unit
                rotational screw uses radians; a unit translation screw uses
                metres. Other screw scalings scale the coordinate accordingly.
        """
        self.axis = axis

    @property
    def axes(self) -> List[Screw]:
        """List[Screw]: The single local motion screw."""
        return [self.axis]


class Planar(JointType):
    """Joint with a caller-specified ordered sequence of local screws.

    The name does not enforce planar geometry or a fixed number of axes.
    Choose coplanar translation/rotation screws when a planar joint is needed.

    Examples:
        >>> from coker.toolkits.kinematics import Planar
        >>> from coker.toolkits.spatial import Screw
        >>> len(Planar(Screw.e_x(), Screw.e_y(), Screw.w_z()).axes)
        3
    """

    def __init__(self, *axes):
        """Store axes in their coordinate and exponential-composition order.

        Args:
            *axes: Local motion screws, each contributing one coordinate.
        """
        self._axes = axes

    @property
    def axes(self) -> List[Screw]:
        """List[Screw]: Local screws in the supplied order."""
        return list(self._axes)


@dataclass
class RigidBody:
    """Articulated link tree for numerical and traceable rigid-body operations.

    Add parents before children using :meth:`add_link`, then attach end
    effectors with :meth:`add_effector`. Link indices count links, whereas
    coordinates count their motion screws; welded links have no coordinates.
    Coordinate vectors concatenate each link's axes in insertion order.

    Poses map child coordinates into parent coordinates. Thus ``A @ B``
    applies ``B`` first, and a link pose is its parent's world pose followed
    by its rest pose and its ordered local screw exponentials. Effector
    poses map effector coordinates into world coordinates.

    Use metres, kilograms and seconds consistently, radians for unit rotation
    screws, and metres for unit translation screws. Spatial vectors and
    Jacobian rows are ordered angular then linear, unlike the coordinate
    order of :class:`Free`. See :meth:`to_function` for basic numerical and
    traced usage.

    Attributes:
        joints: Joint declarations in link order.
        inertia: Inertias in link order.
        parents: Parent link indices.
        transforms: Child-to-parent rest poses.
        end_effectors: ``(parent_link, local_pose)`` pairs in insertion order.
    """

    #: Sentinel ``-1`` for a root link's parent.
    WORLD = -1

    def __init__(self):
        """Create an empty link tree."""
        self.joints: List[JointType] = []

        self.inertia: List[Inertia] = []
        self.parents: List[int] = []
        self.transforms: List[Isometry3] = []
        self.end_effectors: List[Tuple[int, Isometry3]] = []
        self._rest_transforms: List[Isometry3] = []

    @property
    def joint_bases(self) -> List[List[Screw]]:
        """List[List[Screw]]: Local axes grouped in link insertion order."""
        return [j.axes for j in self.joints]

    def joint_global_basis(self, joint_index):
        """Return a link's rest-frame screws expressed in world coordinates.

        Args:
            joint_index: Link index, not flattened coordinate index.

        Returns:
            List[Screw]: Rest-world screws in this link's axis order.
        """
        bases = self.joint_bases[joint_index]

        adjoint = SE3Adjoint(self._rest_transforms[joint_index])
        return [adjoint.apply(b) for b in bases]

    def add_link(
        self, parent: int, at: Isometry3, joint: JointType, inertia: Inertia
    ) -> int:
        """Append a link and return its index.

        Args:
            parent: Existing link index, or :attr:`WORLD` for a root.
            at: Child-to-parent pose at zero joint coordinates.
            joint: Local motion-axis declaration.
            inertia: Link mass properties.

        Returns:
            int: New link index.

        Raises:
            TypeError: The parent is not an integer.
            ValueError: The parent is neither WORLD nor an existing link.
        """
        parent = operator.index(parent)
        idx = len(self.parents)

        if not (0 <= parent < idx or parent == self.WORLD):
            raise ValueError("The parent must be WORLD or an existing link index")

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
        """Attach an end-effector frame to an existing link.

        Args:
            parent: Existing link index; WORLD is not an effector parent.
            at: Effector-to-link pose.

        Returns:
            int: End-effector index in insertion order.

        Raises:
            TypeError: The parent is not an integer.
            ValueError: The parent does not identify an existing link.
        """
        parent = operator.index(parent)
        if not 0 <= parent < len(self.parents):
            raise ValueError("The parent must be an existing link index")
        idx = len(self.end_effectors)
        self.end_effectors.append((parent, at))
        return idx

    def total_joints(self):
        """Return the total number of scalar motion coordinates.

        Returns:
            int: Sum of axis counts, not the number of links.
        """
        return sum([len(j) for j in self.joint_bases])

    def potential_energy(self, angles, gravity_vector):
        """Compute gravitational potential energy relative to the world origin.

        Args:
            angles: Complete coordinate array of shape ``(n,)`` or ``(n, 1)``.
            gravity_vector: World-frame acceleration array of shape ``(3,)``,
                in metres per second squared.

        Returns:
            float: ``-sum(mass * gravity dot world_centre_of_mass)`` in joules.

        Raises:
            ValueError: The coordinate array has an incompatible shape.
        """

        coordinates = self.total_joints()
        if len(angles.shape) != 1:
            if angles.shape[:1] != (coordinates,):
                raise ValueError(f"Expected {coordinates} joint coordinates")
            q = np.reshape(angles, shape=(coordinates,))
        else:
            if angles.shape != (coordinates,):
                raise ValueError(f"Expected {coordinates} joint coordinates")
            q = angles
        joint_transforms = self._get_absolute_joint_xform(q)
        joint_xforms = self._accumulate_joint_xforms(joint_transforms)
        origin = np.array([0, 0, 0])
        total_energy = 0
        for i, inertia in enumerate(self.inertia):
            xform = joint_xforms[i] @ self._rest_transforms[i] @ inertia.centre_of_mass
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
        for rest_transform, bases in zip(self._rest_transforms, self.joint_bases):
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
        """Return flattened coordinates influencing an end effector.

        Args:
            effector_index: End-effector index in insertion order.

        Returns:
            List[int]: Coordinate indices sorted in model coordinate order.
        """
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

        where g_{i-1; j} is the transform of the joint origin to the
        parent frame
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
        """Return link-to-world poses at the supplied coordinates.

        Args:
            angles: Complete coordinate vector of shape ``(n,)``.
            effector: Optional effector index restricting output to its
                ancestral links, from root to parent.

        Returns:
            List[Isometry3]: One pose per link, or per ancestral link when
                ``effector`` is specified. No effector offset is included.
        """

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
        """Return effector-to-world poses in effector insertion order.

        Args:
            angles: Complete coordinate vector of shape ``(n,)``.

        Returns:
            List[Isometry3]: One pose per end effector.
        """
        xforms = self.joint_transforms(angles)
        return [xforms[parent] @ xform for parent, xform in self.end_effectors]

    def get_dependent_links(self, parent_link: int):
        """Iterate a link's ancestry in root-to-link order.

        Args:
            parent_link: Existing link index, or WORLD for an empty ancestry.

        Returns:
            iterator: Link indices including ``parent_link`` unless it is WORLD.
        """
        result = list()
        while parent_link != self.WORLD:
            result.append(parent_link)
            parent_link = self.parents[parent_link]
        return reversed(result)

    def spatial_manipulator_jacobian(self, angles):
        """Return all world-frame spatial manipulator Jacobians.

        Args:
            angles: Complete coordinate vector of shape ``(n,)``.

        Returns:
            List[numpy.ndarray]: One ``(6, n)`` matrix per effector; see
                :meth:`spatial_single_manipulator_jacobian` for conventions.
        """
        return [
            self.spatial_single_manipulator_jacobian(angles, i)
            for i, _ in enumerate(self.end_effectors)
        ]

    def spatial_single_manipulator_jacobian(self, angles, end_effector):
        """Return one world-frame spatial twist Jacobian.

        Args:
            angles: Complete coordinate vector of shape ``(n,)``.
            end_effector: Effector index.

        Returns:
            numpy.ndarray: Shape ``(6, n)`` with angular rows then linear
                spatial-twist rows. Columns follow model coordinate order;
                unrelated branches have zero columns. For effector position
                ``p``, the Cartesian position Jacobian is
                ``J[3:, :] - hat(p) @ J[:3, :]``, not simply ``J[3:, :]``.

        Notes:
            Columns within a multi-axis link use the link's rest basis
            transported by its ancestors, without successive within-link
            adjoint updates. Do not assume a general multi-axis product
            derivative from this method.
        """
        abs_xforms = self._get_absolute_joint_xform(angles)
        xforms = self._accumulate_joint_xforms(abs_xforms)

        effector_parent, _ = self.end_effectors[end_effector]
        dependent_link = set(self.get_dependent_links(effector_parent))

        columns = []

        for link_idx, bases in enumerate(self.joint_bases):
            if link_idx in dependent_link:
                parent = self.parents[link_idx]
                for zeta in bases:
                    zeta_prime = SE3Adjoint(self._rest_transforms[link_idx]).apply(zeta)
                    if parent != self.WORLD:
                        zeta_prime = SE3Adjoint(xforms[parent]).apply(zeta_prime)
                    columns.append(np.reshape(zeta_prime.to_array(), shape=(6, 1)))
            else:
                zeta_prime = np.zeros((6, len(bases)))
                columns.append(zeta_prime)

        return np.concatenate(columns, axis=1)

    def body_manipulator_jacobian(self, angles):
        """Express spatial manipulator Jacobians in each effector frame.

        Args:
            angles: Complete coordinate vector of shape ``(n,)``.

        Returns:
            List[numpy.ndarray]: ``(6, n)`` matrices in effector order, each
                ``Adjoint(g.inverse()) @ J_spatial``. Rows are angular then
                linear; the multi-axis limits of the spatial method apply.
        """
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
        """Compute generalised efforts with recursive inverse dynamics.

        Args:
            angles: Coordinate vector of shape ``(n,)``.
            angle_rates: Coordinate rates of shape ``(n,)``.
            angle_accel: Coordinate accelerations of shape ``(n,)``.
            gravity: World-frame acceleration array of shape ``(3,)``.
            end_effector_forces: Optional mapping from effector indices to
                world-frame wrench arrays of shape ``(6,)``, ordered moment
                then force. A sequence in effector order is also accepted.
                Each wrench is transformed by the coadjoint of that
                effector's world pose and subtracted from its parent's load.

        Returns:
            numpy.ndarray: Shape ``(n,)`` in coordinate order. Unit rotation
                axes produce newton metres; unit translation axes produce
                newtons. Includes inertia, velocity and gravity effects.

        Notes:
            Joint axes are treated as fixed local motion subspaces by this
            recursion. General multi-axis product joints need not have the
            same velocity-dependent terms as a chain of single-axis links.
        """
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
        joint_transforms = self._get_absolute_joint_xform(angles)  # exp(zeta_hat)
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
        """Compute the coordinate-space inertia from centre-of-mass Jacobians.

        Args:
            angles: Complete coordinate vector of shape ``(n,)``.

        Returns:
            numpy.ndarray: Shape ``(n, n)`` in coordinate order, summing
                ``J_com.T @ inertia.as_matrix() @ J_com`` over links.
        """
        jacobians = self._get_link_com_jacobians(angles)
        n = self.total_joints()
        mass_matrix = np.zeros((n, n))
        for j, inertia in zip(jacobians, self.inertia):

            mass_matrix += j.T @ inertia.as_matrix() @ j

        return mass_matrix

    def add_body(self, body: "RigidBody", at: Isometry3, parent: int):
        """Append another body's links and effectors under a parent.

        Joint and inertia objects are reused, not deep-copied.

        Args:
            body: Source body; do not attach a body to itself.
            at: Attachment pose from the source world frame into ``parent``.
                Each source root's rest pose becomes ``at @ root_pose``.
            parent: Existing destination link index, or WORLD.

        Returns:
            tuple: ``(links, effectors)`` lists mapping source insertion
                indices to destination indices.
        """
        links = []

        iterator = zip(body.parents, body.joints, body.transforms, body.inertia)
        for i, (p_body, joint, xform, inertia) in enumerate(iterator):
            idx = len(self.parents)

            self.parents.append(parent if p_body == self.WORLD else links[p_body])
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

    def to_function(self):
        """Return this body's end-effector kinematics as a traceable callable.

        Returns:
            callable: Accepts one complete coordinate vector of shape
                ``(n,)`` and returns ``(positions, jacobians)``. For ``m``
                effectors these arrays have shapes ``(3*m,)`` and ``(3*m, n)``.
                Positions and Cartesian Jacobian row blocks follow effector
                insertion order in world coordinates.

        At least one effector and a nonempty Jacobian column construction are
        required. This uses the numerical forward-kinematics and spatial
        Jacobian methods, including their multi-axis limitations; it does not
        introduce another kinematic model. Pass it directly to
        :func:`coker.function` for a backend-specific :class:`coker.Function`.
        See the kinematics API reference for a numerical-to-traceable example.
        """

        def kinematics(angles):
            transforms = self.forward_kinematics(angles)
            positions = np.concatenate(
                [transform.translation for transform in transforms]
            )
            jacobians = np.concatenate(
                [
                    spatial[3:, :] - hat(transform.translation) @ spatial[:3, :]
                    for transform, spatial in zip(
                        transforms, self.spatial_manipulator_jacobian(angles)
                    )
                ],
                axis=0,
            )
            return positions, jacobians

        return kinematics

from matplotlib import pyplot as plt
import numpy as np
from typing import List
from coker.toolkits.kinematics.rigid_body import RigidBody
from coker.toolkits.spatial import Isometry3

from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.animation import FuncAnimation

e_x = np.array([1, 0, 0])
e_y = np.array([0, 1, 0])
e_z = np.array([0, 0, 1])


class JointWidget:
    def __init__(self, transform: Isometry3, scale):
        self.x_axis = Line3D([], [], [], color="blue")
        self.y_axis = Line3D([], [], [], color="green")
        self.z_axis = Line3D([], [], [], color="red")
        self.point = Line3D(
            [], [], [], marker="o", markersize=1, markerfacecolor="black"
        )
        self.scale = scale
        self.set_transform(transform)

    def set_transform(self, transform: Isometry3):
        point = transform.apply(
            np.zeros(
                3,
            )
        )

        x, y, z = point.tolist()
        self.point.set_data_3d([x], [y], [z])

        dx = transform.apply(e_x) - point
        dy = transform.apply(e_y) - point
        dz = transform.apply(e_z) - point
        x_x, x_y, x_z = (point + self.scale * dx).tolist()
        y_x, y_y, y_z = (point + self.scale * dy).tolist()
        z_x, z_y, z_z = (point + self.scale * dz).tolist()

        self.x_axis.set_data_3d([x, x_x], [y, x_y], [z, x_z])
        self.y_axis.set_data_3d([x, y_x], [y, y_y], [z, y_z])
        self.z_axis.set_data_3d([x, z_x], [y, z_y], [z, z_z])

        return self.point, self.x_axis, self.y_axis, self.z_axis

    def get_artists(self):
        return self.point, self.x_axis, self.y_axis, self.z_axis


class CenterOfMassWidget:
    def __init__(self, parent: int, transform: Isometry3):
        self.parent = parent
        self.transform = transform
        self.artist = Line3D([], [], [], linestyle="", marker="o")

    def set_transform(self, joint_transforms: List[Isometry3]):
        transform = joint_transforms[self.parent] @ self.transform

        x, y, z = transform.apply(np.zeros(3)).tolist()
        self.artist.set_data_3d([x], [y], [z])

    def get_artists(self):
        return self.artist


class LimbWidget:
    def __init__(self, path: List[int], effector_xform: Isometry3):
        self.path = path
        self.line = Line3D([], [], [], color="black")
        self.xform = effector_xform
        self.bounds = []

    def get_artists(self):
        return self.line

    def set_transform(self, transforms: List[Isometry3]):
        xforms = [transforms[i] for i in self.path]
        xforms.append(xforms[-1] @ self.xform)

        x, y, z = zip(*(t.apply(np.zeros((3,))).tolist() for t in xforms))

        self.line.set_data_3d(x, y, z)

        return self.line


class KinematicsVisualiser:
    def __init__(self, model: RigidBody, scale=0.25):
        self.model = model
        self.figure = plt.figure()
        ax = self.figure.add_subplot(projection="3d", aspect="equal")

        rest_config = np.zeros(model.total_joints())
        self.masses = [
            CenterOfMassWidget(i, inertia.centre_of_mass)
            for i, inertia in enumerate(model.inertia)
        ]

        self.joints = [
            JointWidget(t, scale) for t in model.joint_transforms(rest_config)
        ]

        self.end_effector_coords = [
            JointWidget(t, scale)
            for t in model.forward_kinematics(rest_config)
        ]

        for mass in self.masses:
            ax.add_artist(mass.get_artists())

        for joints in self.joints:
            [ax.add_artist(a) for a in joints.get_artists()]

        self.end_effectors = [
            LimbWidget([*model.get_dependent_links(parent)], xform)
            for i, (parent, xform) in enumerate(model.end_effectors)
        ]

        for end_effectors in self.end_effector_coords:
            [ax.add_artist(a) for a in end_effectors.get_artists()]

        locations = model.joint_transforms(rest_config)
        for end_effectors in self.end_effectors:
            end_effectors.set_transform(locations)
            ax.add_artist(end_effectors.get_artists())

        self.ax = ax
        self.update_values(rest_config)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    def update_values(self, angles):
        joint_locations = self.model.joint_transforms(angles)
        fk = self.model.forward_kinematics(angles)

        for joint, transform in zip(self.joints, joint_locations):
            joint.set_transform(transform)

        for effector, transform in zip(self.end_effector_coords, fk):
            effector.set_transform(transform)

        for end_effector in self.end_effectors:
            end_effector.set_transform(joint_locations)

        for mass in self.masses:
            mass.set_transform(joint_locations)

    def get_artists(self):
        return (
            [j.get_artists() for j in self.joints]
            + [e.get_artists() for e in self.end_effectors]
            + [e.get_artists() for e in self.end_effector_coords]
            + [m.get_artists() for m in self.masses]
        )

    def set_view(self, x_lim=None, y_lim=None, z_lim=None):
        if x_lim is not None:
            self.ax.set_xlim(*x_lim)
        if y_lim is not None:
            self.ax.set_ylim(*y_lim)
        if z_lim is not None:
            self.ax.set_zlim(*z_lim)
        self.ax.set_aspect("equal")

    def animate_sweep(self, angles_over_time, loop=False, interval=100):
        def update(frame):
            angles = angles_over_time[frame]
            self.update_values(angles)
            return self.get_artists()

        ani = FuncAnimation(
            fig=self.figure,
            func=update,
            frames=len(angles_over_time),
            interval=interval,
            repeat=loop,
        )
        plt.show()

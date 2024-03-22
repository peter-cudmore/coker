import numpy as np
import sys

import matplotlib.pyplot as plt
import pathlib

project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root.absolute() / 'src'))
from coker.toolkits.kinematics import CompositeBodyModel, flatten_model, forward_kinematics, KinematicTree

leg_json = project_root / 'assets' / 'leg.json'
hexapod_json = project_root / "assets" / "hexapod.json"


import matplotlib
matplotlib.use('qtagg')

# plt.ion()


def main():
    # flat_model = KinematicTree.from_json(leg_json.read_text())
    flat_model = flatten_model(CompositeBodyModel.from_json(hexapod_json.read_text()))

    args = np.zeros(shape=(flat_model.degrees_of_freedom,))

    render(flat_model, args)


def render(flat_model, args, only_effector=None):

    joints, effectors = forward_kinematics(flat_model, args)

    joint_locations = []

    origin = np.array([0, 0, 0, 1], dtype=float)

    for i, (joint, parent) in enumerate(zip(joints, flat_model.parents)):

        point = joint(*args) @ origin
        joint_locations.append(point)

    seen_edges = set()

    effector_lines = []

    for i, (effector, parent) in enumerate(zip(effectors, flat_model.end_effector_parents)):

        start_point = effector(*args) @ origin
        line = [start_point[0:3], joint_locations[parent][0:3]]

        last_node = parent
        next_node = flat_model.parents[parent]

        while next_node is not None and (next_node, last_node) not in seen_edges:
            line.append(joint_locations[next_node][0:3])
            seen_edges.add((next_node, last_node))
            last_node = next_node
            next_node = flat_model.parents[next_node]

        effector_lines.append(list(zip(*[item.tolist() for item in line])))

    ax = plt.figure().add_subplot(projection='3d')

    for x, y, z in effector_lines:
        ax.scatter(x, y, z, 'o')
        ax.plot(x, y, z)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect('equal')
    plt.show()


# def run_simulation(model, controller):


if __name__ == '__main__':
    main()

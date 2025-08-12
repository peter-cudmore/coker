import numpy as np
import xml.etree.ElementTree as ET

from coker.toolkits.kinematics import RigidBody, Inertia
from coker.toolkits.spatial import Rotation3


def to_string(item) -> str:
    if isinstance(item, np.ndarray):
        return " ".join([f"{x}" for x in item.tolist()])
    if isinstance(item, Rotation3):
        return " ".join([f"{x}" for x in item.as_quaternion().to_elements()])

    if isinstance(item, list):
        return " ".join([f"{i}" for i in item])

    raise NotImplementedError


def build_inertial_tags(inertial: Inertia):
    pos = to_string(inertial.centre_of_mass.translation)
    fullinertia = to_string(
        [
            inertial.moments[0],
            inertial.moments[3],
            inertial.moments[5],
            inertial.moments[1],
            inertial.moments[2],
            inertial.moments[4],
        ]
    )

    return dict(pos=pos, fullinertia=fullinertia, mass=f"{inertial.mass}")


def to_mujoco_xml(body: RigidBody, name: str) -> ET.ElementTree:
    root = ET.Element("mujoco")
    root.attrib["model"] = name
    world = ET.SubElement(root, "worldbody")
    floor = ET.SubElement(world, "geom")
    floor.attrib.update(
        dict(name="floor", size="0 0 0.05", type="plane", condim="3")
    )

    bodies = []
    joints = []
    for i, (parent_idx, xform, joint, inertia) in enumerate(
        zip(body.parents, body.transforms, body.joints, body.inertia)
    ):

        if parent_idx == body.WORLD:
            this_body = ET.SubElement(world, "body")
        else:
            this_body = ET.SubElement(bodies[parent_idx], "body")

        pos = " ".join([f"{x}" for x in xform.translation.tolist()])

        this_body.attrib.update(dict(name=f"body_{i}", pos=pos))

        inertia_el = ET.SubElement(this_body, "inertial")
        inertia_el.attrib.update(build_inertial_tags(inertia))

        bodies.append(this_body)

        if xform.rotation is not None and xform.rotation != Rotation3.zero():
            orientation = " ".join(
                [f"{q}" for q in xform.rotation.as_quaternion().to_elements()]
            )
            this_body.attrib.update(dict(quat=orientation))

        if len(joint.axes) == 6:
            ET.SubElement(this_body, "freejoint")
            continue

        for axis in joint.axes:
            j = len(joints)
            joint_el = ET.SubElement(this_body, "joint")
            joints.append(joint_el)
            if (axis.translation == np.array([0, 0, 0])).all():
                rot = " ".join([f"{x}" for x in axis.rotation])
                joint_el.attrib.update(dict(name=f"q_{j}", axis=rot))
            else:
                raise NotImplementedError

        parent_bdy = bodies[parent_idx]
        geom = ET.SubElement(parent_bdy, "geom")

        geom.attrib.update(
            dict(
                type="capsule",
                size="0.01",
                fromto="0 0 0 " + to_string(xform.apply(np.zeros((3,)))),
            )
        )

    for parent_idx, xform in body.end_effectors:
        parent = bodies[parent_idx]
        geom = ET.SubElement(parent, "geom")

        geom.attrib.update(
            dict(
                type="capsule",
                size="0.01",
                fromto="0 0 0 " + to_string(xform.translation),
            )
        )
    return ET.ElementTree(root)

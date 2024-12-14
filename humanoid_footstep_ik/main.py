from typing import Literal
from numpy.typing import NDArray
from pathlib import Path
import numpy as np

from pydrake.geometry import StartMeshcat
from pydrake.geometry.all import Box, MeshcatVisualizer
from pydrake.multibody.inverse_kinematics import InverseKinematics

from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, CoulombFriction
from pydrake.multibody.tree import JointIndex
from pydrake.solvers import Solve
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
import pickle

from dataclasses import dataclass


@dataclass
class FootstepTrajectory:
    com_z: float
    foot_z: float
    foot_half_width: float
    foot_half_length: float
    stones: list[tuple[float, float]]  # [(x, y)_lower, (x, y)_upper]
    com_xy_position: NDArray[np.float64]
    cop_xy_position: NDArray[np.float64]
    left_foot_xy_position: NDArray[np.float64]
    right_foot_xy_position: NDArray[np.float64]
    com_xy_velocity: NDArray[np.float64]
    com_xy_acceleration: NDArray[np.float64]
    contact_modes: list[Literal["Ld_Rd", "Ld_Ru", "Lu_Rd"]]

    @classmethod
    def load(cls, filepath: Path) -> "FootstepTrajectory":
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        return cls(
            com_z=data["com_z"],
            foot_z=data["foot_z"],
            foot_half_width=data["foot_half_width"],
            foot_half_length=data["foot_half_length"],
            stones=data["stones"],
            com_xy_position=data["com_xy_position"],
            cop_xy_position=data["cop_xy_position"],
            left_foot_xy_position=data["left_foot_xy_position"],
            right_foot_xy_position=data["right_foot_xy_position"],
            com_xy_velocity=data["com_xy_velocity"],
            com_xy_acceleration=data["com_xy_acceleration"],
            contact_modes=data["contact_modes"],
        )


if __name__ == "__main__":

    datapath = Path("data/example_data.pkl")
    traj = FootstepTrajectory.load(datapath)

    breakpoint()

    meshcat = StartMeshcat()

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

    parser = Parser(plant)
    # atlas_model_file = "package://drake_models/atlas/atlas_convex_hull.urdf"
    atlas_model_file = "package://drake_models/atlas/atlas_minimal_contact.urdf"
    atlas_model_instance = parser.AddModelsFromUrl(atlas_model_file)[0]

    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    diagram.set_name("plant and scene_graph")

    BOX_HEIGHT = 0.5
    z_pos = -BOX_HEIGHT / 2
    boxes = [
        Box(1.0, 2.0, BOX_HEIGHT),
        Box(1.0, 2.0, BOX_HEIGHT),
    ]
    box_pos = [
        np.array([-0.7, 0.0]),
        np.array([0.7, 0.0]),
    ]

    friction = CoulombFriction(static_friction=0.9, dynamic_friction=0.5)

    for idx, (box, pos) in enumerate(zip(boxes, box_pos)):
        transform = RigidTransform(np.array([pos[0], pos[1], z_pos]))  # type: ignore
        plant.RegisterCollisionGeometry(
            plant.world_body(),
            transform,
            box,
            f"box_{idx}",
            friction,
        )
        plant.RegisterVisualGeometry(
            plant.world_body(),
            transform,
            box,
            f"box_{idx}_visual",
            [0.5, 0.5, 0.8, 1.0],  # RGBA color (light blue in this case)
        )

    plant.Finalize()

    print_details = False
    if print_details:
        # Query the number of positions and velocities
        num_positions = plant.num_positions()
        num_velocities = plant.num_velocities()
        total_states = num_positions + num_velocities

        print(f"Number of positions: {num_positions}")
        print(f"Number of velocities: {num_velocities}")
        print(f"Total states: {total_states}")

        # Get position and velocity names
        position_names = []
        velocity_names = []

        for joint_index in range(plant.num_joints()):
            joint = plant.get_joint(JointIndex(joint_index))
            for position_index in range(joint.num_positions()):
                pos_name = joint.name()
                if joint.num_positions() > 1:
                    pos_name += f"_{position_index}"
                position_names.append(pos_name)
            for velocity_index in range(joint.num_velocities()):
                vel_name = joint.name()
                if joint.num_velocities() > 1:
                    vel_name += f"_{velocity_index}"
                velocity_names.append(vel_name)

        # Print position and velocity names
        print("Positions:", position_names)
        print("Velocities:", velocity_names)

        # Print all frame names
        print("Frames in the robot:")
        for frame in plant.GetFrameIndices(atlas_model_instance):
            frame_name = plant.get_frame(frame).name()
            print(frame_name)

        # Iterate over all joints
        for joint_index in range(plant.num_joints()):
            joint = plant.get_joint(JointIndex(joint_index))
            print(
                f"Joint '{joint.name()}' corresponds to generalized coordinates: "
                f"{joint.position_start()}, {joint.position_start() + joint.num_positions() - 1}"
            )
            print(
                f"  Associated frames: {joint.frame_on_parent().name()} and {joint.frame_on_child().name()}"
            )

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    default_positions = plant.GetPositions(plant_context)

    # Inverse kinematics
    base_frame = plant.GetFrameByName("pelvis", atlas_model_instance)
    left_foot_frame = plant.GetFrameByName("l_foot", atlas_model_instance)
    right_foot_frame = plant.GetFrameByName("r_foot", atlas_model_instance)

    ik = InverseKinematics(plant)
    # CoM
    ik.AddPositionConstraint(
        frameB=base_frame,  # End-effector frame
        p_BQ=np.array(
            [0.0, 0.0, 0.0]
        ),  # Point Q in frame B (end-effector origin) # type: ignore
        frameA=plant.world_frame(),  # World frame
        p_AQ_lower=[0, 0, 0.7],  # type: ignore
        p_AQ_upper=[0, 0, 1.0],  # type: ignore
    )
    ik.AddOrientationConstraint(
        frameAbar=base_frame,
        R_AbarA=RotationMatrix(),
        frameBbar=plant.world_frame(),
        R_BbarB=RotationMatrix(),
        theta_bound=0.01,
    )

    # Left foot
    ik.AddPositionConstraint(
        frameB=left_foot_frame,
        p_BQ=np.array(
            [0.0, 0.0, 0.0]
        ),  # Point Q in frame B (end-effector origin) # type: ignore
        frameA=plant.world_frame(),  # World frame
        p_AQ_lower=[0.1, 0.2, 0.0],  # type: ignore
        p_AQ_upper=[0.1, 0.2, 0.2],  # type: ignore
    )
    ik.AddOrientationConstraint(
        frameAbar=left_foot_frame,
        R_AbarA=RotationMatrix(),
        frameBbar=plant.world_frame(),
        R_BbarB=RotationMatrix(),
        theta_bound=0.01,
    )

    # Right foot
    ik.AddPositionConstraint(
        frameB=right_foot_frame,
        p_BQ=np.array(
            [0.0, 0.0, 0.0]
        ),  # Point Q in frame B (end-effector origin) # type: ignore
        frameA=plant.world_frame(),  # World frame
        p_AQ_lower=[0.2, -0.2, 0.0],  # type: ignore
        p_AQ_upper=[0.2, -0.2, 0.2],  # type: ignore
    )
    ik.AddOrientationConstraint(
        frameAbar=right_foot_frame,
        R_AbarA=RotationMatrix(),
        frameBbar=plant.world_frame(),
        R_BbarB=RotationMatrix(),
        theta_bound=0.01,
    )

    result = Solve(ik.prog())
    if result.is_success():
        solution = result.GetSolution(ik.q())
    else:
        raise RuntimeError("Couldn't find IK solution")

    plant.SetPositions(plant_context, solution)

    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(0.05)

    while True:
        ...

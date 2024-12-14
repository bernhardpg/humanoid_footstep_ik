import logging
from typing import Literal
from numpy.typing import NDArray
from pathlib import Path
import numpy as np

from pydrake.geometry import StartMeshcat
from pydrake.geometry.all import MeshcatVisualizer
from pydrake.geometry.all import Box as DrakeBox
from pydrake.multibody.inverse_kinematics import InverseKinematics

from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    CoulombFriction,
    MultibodyPlant,
)
from pydrake.multibody.tree import JointIndex, ModelInstanceIndex
from pydrake.solvers import Solve
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
import pickle

from dataclasses import dataclass, field


@dataclass
class VisualizationParams:
    stone_height: float


@dataclass
class Stone:
    name: str
    width: float
    depth: float
    com: NDArray[np.float64]
    color: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.5, 0.5, 0.8, 1.0])  # light purple
    )

    @classmethod
    def from_bbox(
        cls,
        name: str,
        bbox: tuple[tuple[float, float], tuple[float, float]],
    ) -> "Stone":
        """
        Creates a Stone instance from a bounding box.

        Args:
            bbox (tuple[tuple[float, float], tuple[float, float]]):
                A bounding box defined as [(x_lower, y_lower), (x_upper, y_upper)].

        Returns:
            Stone: An instance of the Stone class.
        """
        # Extract lower and upper bounds
        (x_lower, y_lower), (x_upper, y_upper) = bbox

        # Compute width and depth
        width = x_upper - x_lower
        depth = y_upper - y_lower

        # Compute center of mass (CoM)
        com = np.array(
            [(x_lower + x_upper) / 2, (y_lower + y_upper) / 2],
            dtype=np.float64,
        )

        return cls(name=name, width=width, depth=depth, com=com)

    def add_to_plant(self, plant: MultibodyPlant, stone_height: float) -> None:
        friction = CoulombFriction(static_friction=0.9, dynamic_friction=0.5)

        z_pos = -stone_height / 2  # z=0 is top of stone

        transform = RigidTransform(np.array([self.com[0], self.com[1], z_pos]))  # type: ignore
        plant.RegisterCollisionGeometry(
            plant.world_body(),
            transform,
            self.get_drake_box(stone_height),
            self.name,
            friction,
        )
        plant.RegisterVisualGeometry(
            plant.world_body(),
            transform,
            self.get_drake_box(stone_height),
            self.name,
            self.color,  # type: ignore
        )

    def get_drake_box(self, stone_height: float) -> DrakeBox:
        return DrakeBox(self.width, self.depth, stone_height)


@dataclass
class FootstepTrajectory:
    com_z: float
    foot_z: float
    foot_half_width: float
    foot_half_length: float
    stones: list[Stone]
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
            stones=[
                Stone.from_bbox(f"stone_{idx}", bbox)
                for idx, bbox in enumerate(data["stones"])
            ],
            com_xy_position=data["com_xy_position"],
            cop_xy_position=data["cop_xy_position"],
            left_foot_xy_position=data["left_foot_xy_position"],
            right_foot_xy_position=data["right_foot_xy_position"],
            com_xy_velocity=data["com_xy_velocity"],
            com_xy_acceleration=data["com_xy_acceleration"],
            contact_modes=data["contact_modes"],
        )


def print_atlas_model_details(
    plant: MultibodyPlant, atlas_model_instance: ModelInstanceIndex
) -> None:
    # Query the number of positions and velocities
    num_positions = plant.num_positions()
    num_velocities = plant.num_velocities()
    total_states = num_positions + num_velocities

    print(f"Number of positions: {num_positions}")
    print(f"Number of velocities: {num_velocities}")
    print(f"Total states: {total_states}")
    print()

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
    print()

    # Print all frame names
    print("Frames in the robot:")
    for frame in plant.GetFrameIndices(atlas_model_instance):
        frame_name = plant.get_frame(frame).name()
        print(frame_name)
    print()

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


def solve_ik(
    plant: MultibodyPlant, atlas_model_instance: ModelInstanceIndex
) -> NDArray[np.float64]:
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

    return solution  # type: ignore


def visualize_trajectory(
    traj: FootstepTrajectory, viz_params: VisualizationParams, debug: bool = False
) -> None:
    meshcat = StartMeshcat()

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

    parser = Parser(plant)
    atlas_model_file = "package://drake_models/atlas/atlas_convex_hull.urdf"
    # atlas_model_file = "package://drake_models/atlas/atlas_minimal_contact.urdf"
    atlas_model_instance = parser.AddModelsFromUrl(atlas_model_file)[0]

    if debug:
        print_atlas_model_details(plant, atlas_model_instance)

    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    diagram.set_name("plant and scene_graph")

    for stone in traj.stones:
        stone.add_to_plant(plant, viz_params.stone_height)

    plant.Finalize()

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    default_positions = plant.GetPositions(plant_context)

    solution = solve_ik(plant, atlas_model_instance)
    plant.SetPositions(plant_context, solution)

    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(0.05)

    while True:
        ...


if __name__ == "__main__":
    # Silence Drake messages
    # logging.getLogger("drake").setLevel(logging.WARNING)

    datapath = Path("data/example_data.pkl")
    traj = FootstepTrajectory.load(datapath)

    viz_params = VisualizationParams(stone_height=0.5)

    visualize_trajectory(traj, viz_params)

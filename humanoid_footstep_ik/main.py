import logging
from typing import Literal
from numpy._core.numerictypes import float64
from numpy.typing import NDArray
from pathlib import Path
import numpy as np

from pydrake.geometry import Rgba, Role, SceneGraph, StartMeshcat
from pydrake.geometry.all import MeshcatVisualizer
from pydrake.geometry.all import Box as DrakeBox
from pydrake.multibody.inverse_kinematics import InverseKinematics

from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    CoulombFriction,
    MultibodyPlant,
)
from pydrake.multibody.tree import BodyIndex, JointIndex, ModelInstanceIndex
from pydrake.solvers import Solve
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import Context, DiagramBuilder
import pickle

from dataclasses import dataclass, field


@dataclass
class VisualizationParams:
    stone_height: float
    robot_z_rot: float


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

    def get_unique_foot_positions(
        self, foot: Literal["right", "left"]
    ) -> NDArray[np.float64]:
        if foot == "right":
            positions = self.right_foot_xy_position
        else:
            positions = self.left_foot_xy_position

        return np.unique(positions, axis=0)


def solve_ik(
    plant: MultibodyPlant,
    atlas_model_instance: ModelInstanceIndex,
    robot_z_rotation: float,
    com: NDArray[np.float64],
    l_foot: NDArray[np.float64],
    r_foot: NDArray[np.float64],
    q0: NDArray[np.float64],
) -> NDArray[np.float64]:
    base_frame = plant.GetFrameByName("pelvis", atlas_model_instance)
    left_foot_frame = plant.GetFrameByName("l_foot", atlas_model_instance)
    right_foot_frame = plant.GetFrameByName("r_foot", atlas_model_instance)

    robot_rotation = RollPitchYaw(
        np.array([0.0, 0.0, robot_z_rotation])  # type: ignore
    ).ToRotationMatrix()

    ik = InverseKinematics(plant)

    # Stay close to nominal pose
    prog = ik.get_mutable_prog()
    q = ik.q()
    prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)  # type: ignore
    prog.SetInitialGuess(q, q0)  # type: ignore

    # CoM
    ik.AddPositionConstraint(
        frameB=base_frame,  # End-effector frame
        p_BQ=np.array(
            [0.0, 0.0, 0.0]
        ),  # Point Q in frame B (end-effector origin) # type: ignore
        frameA=plant.world_frame(),  # World frame
        p_AQ_lower=com,  # type: ignore
        p_AQ_upper=com,  # type: ignore
    )
    ik.AddOrientationConstraint(
        frameAbar=base_frame,
        R_AbarA=robot_rotation,
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
        p_AQ_lower=l_foot,  # type: ignore
        p_AQ_upper=l_foot,  # type: ignore
    )
    ik.AddOrientationConstraint(
        frameAbar=left_foot_frame,
        R_AbarA=robot_rotation,
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
        p_AQ_lower=r_foot,  # type: ignore
        p_AQ_upper=r_foot,  # type: ignore
    )
    ik.AddOrientationConstraint(
        frameAbar=right_foot_frame,
        R_AbarA=robot_rotation,
        frameBbar=plant.world_frame(),
        R_BbarB=RotationMatrix(),
        theta_bound=0.01,
    )

    result = Solve(ik.prog())
    if result.is_success():
        solution = result.GetSolution(ik.q())
        positions_for_model = plant.GetPositionsFromArray(
            atlas_model_instance, solution
        )
    else:
        raise RuntimeError("Couldn't find IK solution")

    return positions_for_model  # type: ignore


class VisualizationFoot:
    def __init__(
        self, right_or_left: Literal["left", "right"], plant: MultibodyPlant, name: str
    ) -> None:
        self.plant = plant
        self.right_or_left = right_or_left
        self.height = FOOT_HEIGHT

        foot_file = Path(f"assets/atlas/{right_or_left}_foot.urdf")
        assert foot_file.exists()
        self.model_instance = Parser(plant).AddModels(str(foot_file))[0]

        # NOTE: We must rename the feet so that they have unique names
        plant.RenameModelInstance(self.model_instance, name)

        if self.right_or_left == "left":
            self.foot_body = plant.GetBodyByName("l_foot", self.model_instance)
        else:
            self.foot_body = plant.GetBodyByName("r_foot", self.model_instance)

    def set_pose(
        self,
        plant_context: Context,
        pos_xy: NDArray[np.float64],
        pos_z: float,
        rot_z: float,
    ) -> None:
        pos = np.concatenate([pos_xy, [pos_z + self.height / 2]])
        pose = RigidTransform(RollPitchYaw(0, 0, rot_z), pos)  # type: ignore
        self.plant.SetFreeBodyPose(plant_context, self.foot_body, pose)


class VisualizationAtlas:
    def __init__(self, plant: MultibodyPlant, name: str, default_z_rot: float) -> None:
        self.plant = plant
        self.foot_height = FOOT_HEIGHT

        # self.atlas_model_file = "package://drake_models/atlas/atlas_convex_hull.urdf"
        self.atlas_model_file = (
            "package://drake_models/atlas/atlas_minimal_contact.urdf"
        )
        self.model_instance = Parser(plant).AddModelsFromUrl(self.atlas_model_file)[0]
        # NOTE: We must rename the atlases so that they have unique names
        self.name = name
        plant.RenameModelInstance(self.model_instance, self.name)
        self.num_positions = 37

        self.right_shoulder_x_idx = 19
        self.right_shoulder_z_idx = 18

        self.left_shoulder_x_idx = 11
        self.left_shoulder_z_idx = 10

        self.robot_z_rot = default_z_rot

        # [com_quat, com_pos, ...]
        self.q0 = np.zeros((self.num_positions,))
        default_rot = (
            RollPitchYaw(np.array([0.0, 0.0, default_z_rot]))  # type: ignore
            .ToQuaternion()
            .wxyz()
        )
        NUM_QUATERNIONS = 4
        self.q0[:NUM_QUATERNIONS] = default_rot

        self.q0[self.left_shoulder_x_idx] = -1.4
        self.q0[self.left_shoulder_z_idx] = -0.3

        self.q0[self.right_shoulder_x_idx] = 1.4
        self.q0[self.right_shoulder_z_idx] = 0.3

    def make_isolated_plant(self) -> tuple[MultibodyPlant, ModelInstanceIndex]:
        builder = DiagramBuilder()
        isolated_plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        isolated_model_instance = Parser(isolated_plant).AddModelsFromUrl(
            self.atlas_model_file
        )[0]
        isolated_plant.RenameModelInstance(isolated_model_instance, self.name)
        isolated_plant.Finalize()
        return isolated_plant, isolated_model_instance

    def set_com_and_feet_pos(
        self,
        plant_context: Context,
        com_xy: NDArray[np.float64],
        com_z: float,
        l_foot_xy: NDArray[np.float64],
        l_foot_z: float,
        r_foot_xy: NDArray[np.float64],
        r_foot_z: float,
    ) -> None:
        # Set Atlas positions
        com = np.concatenate([com_xy, [com_z]])
        l_foot = np.concatenate([l_foot_xy, [l_foot_z + self.foot_height / 2]])
        r_foot = np.concatenate([r_foot_xy, [r_foot_z + self.foot_height / 2]])

        # Make a plant with only this atlas for the IK
        isolated_plant, isolated_model_instance = self.make_isolated_plant()
        solution = solve_ik(
            isolated_plant,
            isolated_model_instance,
            self.robot_z_rot,
            com,
            l_foot,
            r_foot,
            self.q0,
        )
        # Set the positions of Atlas in the original plant

        self.plant.SetPositions(
            plant_context,
            self.model_instance,
            solution,  # type: ignore
        )

    def print_details(self) -> None:
        """
        Prints a lot of details about the model. Useful for debugging.
        """
        plant, model_instance = self.make_isolated_plant()

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
        for frame in plant.GetFrameIndices(model_instance):
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
        print()

        for body_index in plant.GetBodyIndices(plant.GetModelInstanceByName(self.name)):
            body = plant.get_body(body_index)
            print(f"Body name: {body.name()}")

    # def make_transparent(self, scene_graph: SceneGraph, alpha: float = 0.5) -> None:
    #    """
    #    Makes the Atlas model transparent by adjusting the alpha value of its illustration geometries.
    #
    #    Args:
    #        scene_graph (SceneGraph): The SceneGraph associated with the plant.
    #        alpha (float): Transparency value (0.0 = fully transparent, 1.0 = fully opaque).
    #    """
    #    if not (0.0 <= alpha <= 1.0):
    #        raise ValueError("Alpha must be between 0.0 (transparent) and 1.0 (opaque).")
    #
    #    # Get the source ID for the plant
    #    source_id = self.plant.get_source_id()
    #
    #    # Define the transparent color
    #    transparent_color = Rgba(1.0, 1.0, 1.0, alpha)  # White with transparency
    #
    #    # Access SceneGraph's model inspector
    #    inspector = scene_graph.model_inspector()
    #
    #    # Iterate over all bodies in the Atlas model instance
    #    for body_index in self.plant.GetBodyIndices(self.model_instance):
    #        body = self.plant.get_body(body_index)
    #
    #        # Retrieve geometry IDs associated with the body's visual geometries
    #        geometry_ids = inspector.GetGeometries(body.index(), Role.kIllustration)
    #
    #        for geometry_id in geometry_ids:
    #            # Update the geometry's illustration properties with transparency
    #            illustration_properties = inspector.GetIllustrationProperties(geometry_id)
    #            if illustration_properties is not None:
    #                illustration_properties.UpdateProperty("phong", "diffuse", transparent_color)


def visualize_trajectory(
    traj: FootstepTrajectory, viz_params: VisualizationParams, debug: bool = False
) -> None:
    meshcat = StartMeshcat()

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

    left_foot_positions = traj.get_unique_foot_positions("left")
    right_foot_positions = traj.get_unique_foot_positions("right")

    left_feet = [
        VisualizationFoot("left", plant, name=f"left_{idx}")
        for idx in range(len(left_foot_positions))
    ]
    right_feet = [
        VisualizationFoot("right", plant, name=f"right_{idx}")
        for idx in range(len(right_foot_positions))
    ]

    # TODO: Right now we get into trouble if the atlases collide
    # indices_to_visualize = [0, 10]
    indices_to_visualize = [0, 10]
    atlases = [
        VisualizationAtlas(
            plant, name=f"atlas_at_{idx}", default_z_rot=viz_params.robot_z_rot
        )
        for idx in indices_to_visualize
    ]

    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    diagram.set_name("plant and scene_graph")

    for stone in traj.stones:
        stone.add_to_plant(plant, viz_params.stone_height)

    plant.Finalize()

    if debug:
        atlases[0].print_details()

    turn_off_collision_checking = True
    if turn_off_collision_checking:
        source_id = plant.get_source_id()
        for geometry_id in scene_graph.model_inspector().GetAllGeometryIds():
            scene_graph.RemoveRole(source_id, geometry_id, Role.kProximity)

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # Set the positions of the free-floating feet
    for pos, foot in zip(left_foot_positions, left_feet):
        foot.set_pose(
            plant_context,
            pos_xy=pos,
            pos_z=traj.foot_z,
            rot_z=viz_params.robot_z_rot,
        )

    # Set the positions of the free-floating feet
    for pos, foot in zip(right_foot_positions, right_feet):
        foot.set_pose(
            plant_context,
            pos_xy=pos,
            pos_z=traj.foot_z,
            rot_z=viz_params.robot_z_rot,
        )

    # Set the positions of the atlases
    for i, atlas in zip(indices_to_visualize, atlases):
        atlas.set_com_and_feet_pos(
            plant_context,
            traj.com_xy_position[i],
            traj.com_z,
            traj.left_foot_xy_position[i],
            traj.foot_z,
            traj.right_foot_xy_position[i],
            traj.foot_z,
        )

    # default_positions = plant.GetPositions(plant_context)

    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(0.0)

    while True:
        ...


if __name__ == "__main__":
    # Silence Drake messages
    # logging.getLogger("drake").setLevel(logging.WARNING)

    FOOT_HEIGHT = 0.15

    datapath = Path("data/example_data.pkl")
    traj = FootstepTrajectory.load(datapath)
    # TODO: We have the wrong robot height
    traj.com_z = 0.8

    viz_params = VisualizationParams(stone_height=0.5, robot_z_rot=-np.pi / 2)

    visualize_trajectory(traj, viz_params, debug=False)

from pathlib import Path

# Import some basic libraries and functions for this tutorial.
import numpy as np
import os

from pydrake.common import temp_directory
from pydrake.geometry import SceneGraphConfig, StartMeshcat
from pydrake.geometry.all import MeshcatVisualizer
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import JointIndex
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import AddDefaultVisualization, ModelVisualizer

if __name__ == "__main__":

    meshcat = StartMeshcat()

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

    parser = Parser(plant)
    atlas_model_file = "package://drake_models/atlas/atlas_convex_hull.urdf"
    parser.AddModelsFromUrl(atlas_model_file)

    plant.Finalize()

    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    diagram.set_name("plant and scene_graph")

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

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    default_positions = plant.GetPositions(plant_context)

    new_positions = default_positions.copy()
    new_positions[6] = 1.0  # set the z-position a bit higher
    plant.SetPositions(plant_context, new_positions)

    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(0.05)

    while True:
        ...

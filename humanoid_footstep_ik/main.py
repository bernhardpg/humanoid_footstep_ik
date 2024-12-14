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

    context = diagram.CreateDefaultContext()

    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(0.0)

    while True:
        ...

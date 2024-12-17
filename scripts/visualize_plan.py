import numpy as np
from pathlib import Path

from humanoid_footstep_ik.visualizer import (
    FootstepTrajectory,
    VisualizationParams,
    visualize_trajectory,
)


if __name__ == "__main__":
    # Silence Drake messages
    # logging.getLogger("drake").setLevel(logging.WARNING)

    datapath = Path("data/example_data.pkl")
    traj = FootstepTrajectory.load(datapath)
    # TODO: We have the wrong robot height
    traj.com_z = 0.8

    viz_params = VisualizationParams(
        stone_height=0.5, robot_z_rot=-np.pi / 2, num_atlas_frames=3
    )

    visualize_trajectory(traj, viz_params, debug=False)

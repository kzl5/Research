import genesis as gs
import numpy as np
import math
gs.init(backend=gs.cuda)

scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -3.5, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        res           = (960, 640),
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    show_viewer=True,
)

ur5e = scene.add_entity(
    gs.morphs.MJCF(
        file  = 'xml/universal_robots_ur5e/ur5e.xml',
    ),
)

cam = scene.add_camera(
    res    = (1280, 960),
    pos    = (3.5, 0.0, 2.5),
    lookat = (0, 0, 0.5),
    fov    = 30,
    GUI    = False
)

plane = scene.add_entity(gs.morphs.Plane())

scene.build()

jnt_names = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow',
    'wrist_1',
    'wrist_2',
    'wrist_3',
]
dofs_idx = [ur5e.get_joint(name).dof_idx_local for name in jnt_names]

ur5e.set_dofs_kp(
    kp = np.array([4500, 4500, 3500, 3500, 2000, 2000]),
    dofs_idx_local = dofs_idx,
)

end_effector = ur5e.get_link('wrist_1_link')

# Define multiple target positions (Cartesian coordinates)
target_positions = [
    [0.5, 0, 0.5],
    [0.3, 0.2, 0.6],
    [0.4, -0.2, 0.7],
    [0.5, 0, 0.5],  # return to start
]


cam.start_recording()

# Execute paths to each target
for target_pos in target_positions:
    qpos = ur5e.inverse_kinematics(
        link = end_effector,
        pos = target_pos,

    )
    
    path = ur5e.plan_path(
        qpos_goal     = qpos,
        num_waypoints = 200, # 2s duration
        quat = [0, 1, 0, 0]
    )
    
    # execute the planned path
    for waypoint in path:
        ur5e.control_dofs_position(waypoint)
        scene.step()
        cam.render()
    
    # allow robot to settle at this waypoint
    for i in range(100):
        scene.step()
        cam.render()

cam.stop_recording(save_to_filename='IKtest.mp4', fps=60)


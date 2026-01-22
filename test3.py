import numpy as np
import genesis as gs

gs.init(backend=gs.cpu)

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
    show_viewer = True,
)

# plane = scene.add_entity(
#     gs.morphs.Plane(),
# )
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

scene.build()

cam.start_recording()

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
    kp = np.array([4500, 4500, 4500, 4500, 4500, 4500]),
    dofs_idx_local = dofs_idx,
)
i = 0
for i in range(300):
    ur5e.set_dofs_position(np.array([i*(np.pi/180), 0, 0, 0, 0, 0]), dofs_idx)
    scene.step()

cam.stop_recording(save_to_filename='video.mp4', fps=60)
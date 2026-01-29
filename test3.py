import numpy as np
import genesis as gs
import math

gs.init(backend=gs.cuda)

scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -3.5, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        res           = (960, 640),
        max_FPS       = 5,
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

jnt_names = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow',
    'wrist_1',
    'wrist_2',
    'wrist_3',
]
dofs_idx = [ur5e.get_joint(name).dof_idx_local for name in jnt_names]

# ur5e.set_dofs_kp(
#     kp = np.array([4500, 4500, 4500, 4500, 4500, 4500]),
#     dofs_idx_local = dofs_idx,
# )

ur5e.set_dofs_force_range(
    lower = np.array([-150, -150, -150, -150, -150, -150]),
    upper = np.array([ 150,  150,  150,  150,  150,  150]),
    dofs_idx_local = dofs_idx,
)

ur5e.set_dofs_kp(
    kp = np.array([300, 300, 300, 300, 300, 300]),
    dofs_idx_local = dofs_idx,
)

cam.start_recording()

########################################
for i in range(150):
    j = math.sin(math.radians(i*math.pi))
    if j == 1:
        ur5e.set_dofs_position(np.array([0,0,0,0,0,0]), dofs_idx)
    elif j == 0:
        ur5e.set_dofs_position(np.array([math.radians(90),math.radians(90),0,0,0,0]), dofs_idx)
    elif j == -1:
        ur5e.set_dofs_position(np.array([0,0,0,0,0,0]), dofs_idx)
    scene.step()
    cam.render()

cam.stop_recording(save_to_filename='video.mp4', fps=60)

# TO DO:
# 1: ADD FORCE CONTROL GAINS TO THIS SCRIPT
# 2: MODIFY FOR LOOP TO GENERATE ALL POSSIBLE POSITIONS OF THE ROBOT FROM ZERO POSITION
# 3: RUN VISUALIZATION FOR FORCE DATA PER JOINT
# 4: WRITE UP FOR BAI OBTAIN FEEDBACK ASAP
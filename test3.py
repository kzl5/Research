import numpy as np
import genesis as gs
import math
import matplotlib.pyplot as plt

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

controlforcearr = np.array([])
dofforcearr = np.array([])

for i in range(360):
    ur5e.set_dofs_position(np.array([math.radians(i), math.radians(i), math.radians(i), 0, 0, 0]), dofs_idx)

    print('control force: ', ur5e.get_dofs_control_force(dofs_idx))
    print('dof force: ', ur5e.get_dofs_force(dofs_idx))

    controlforcearr = np.append(controlforcearr, ur5e.get_dofs_control_force(dofs_idx))
    dofforcearr = np.append(dofforcearr, ur5e.get_dofs_force(dofs_idx))

    scene.step()

cam.stop_recording(save_to_filename='video.mp4', fps=60)

# %% 

timearr = np.array(range(0, 360))
plt.figure()
plt.plot(controlforcearr, label='Control Force')



# %%

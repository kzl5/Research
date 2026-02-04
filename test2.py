import numpy as np
import genesis as gs
import math

########################## init ##########################
gs.init(backend=gs.cuda)

########################## create a scene ##########################
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



########################## entities ##########################

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

########################## build ##########################
scene.build()

# start camera recording. Once this is started, all the rgb images rendered will be recorded internally


jnt_names = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow',
    'wrist_1',
    'wrist_2',
    'wrist_3',
]
dofs_idx = [ur5e.get_joint(name).dof_idx_local for name in jnt_names]

############ Optional: set control gains ############
# set positional gains
ur5e.set_dofs_kp(
    kp             = np.array([4500, 4500, 3500, 3500, 2000, 2000]),
    dofs_idx_local = dofs_idx,
)
# set velocity gains
ur5e.set_dofs_kv(
    kv             = np.array([450, 450, 350, 350, 200, 200]),
    dofs_idx_local = dofs_idx,
)
# set force range for safety
ur5e.set_dofs_force_range(
    lower          = np.array([-87, -87, -87, -87, -12, -12]),
    upper          = np.array([ 87,  87,  87,  87,  12,  12]),
    dofs_idx_local = dofs_idx,
)

cam.start_recording()

# Hard reset
for i in range(150):
    if i < 50:
        ur5e.set_dofs_position(np.array([0, 0, 0, 0, 0, 0]), dofs_idx)
    elif i < 100:
        ur5e.set_dofs_position(np.array([math.radians(180), math.radians(180), 0, 0, 0, 0]), dofs_idx)
    else:
        ur5e.set_dofs_position(np.array([0, 0, 0, 0, 0, 0]), dofs_idx)
    scene.step()
    cam.render()

controlforcearr = []
internalforcearr = []

# PD control
for i in range(1500):
    if i == 0:
        ur5e.control_dofs_position(
            np.array([math.radians(0), math.radians(0), math.radians(0), 0, 0, 0]),
            dofs_idx,
        )
    elif i == 250:
        ur5e.control_dofs_position(
            np.array([math.radians(-180), math.radians(-45), math.radians(-45), 0, 0, 0]),
            dofs_idx,
        )
    elif i == 500:
        ur5e.control_dofs_position(
            np.array([math.radians(-90), math.radians(-45), math.radians(-45), 0, 0, 0]),
            dofs_idx,
        )
    elif i == 750:
        ur5e.control_dofs_position(
            np.array([0, 0, 0, 0, 0, 0]),
            dofs_idx,
        )
    elif i == 1000:
        ur5e.control_dofs_position(
            np.array([math.radians(-90), math.radians(-45), math.radians(-45), 0, 0, 0]), 
            dofs_idx)
    elif i == 1250:
        ur5e.control_dofs_force(
            np.array([math.radians(90), math.radians(45), math.radians(45), 0, 0, 0]),
            dofs_idx,
        )
    # This is the control force computed based on the given control command
    # If using force control, it's the same as the given control command
    ctrl_raw = ur5e.get_dofs_control_force(dofs_idx)
    try:
        # handle PyTorch tensors (possibly on CUDA)
        import torch
        if isinstance(ctrl_raw, torch.Tensor):
            ctrl = ctrl_raw.detach().cpu().numpy()
        else:
            ctrl = np.asarray(ctrl_raw)
    except Exception:
        # non-torch objects -> try numpy conversion
        ctrl = np.asarray(ctrl_raw)
    print('control force:', ctrl)
    controlforcearr.append(ctrl)


    # This is the actual force experienced by the dof
    int_raw = ur5e.get_dofs_force(dofs_idx)
    try:
        import torch
        if isinstance(int_raw, torch.Tensor):
            intr = int_raw.detach().cpu().numpy()
        else:
            intr = np.asarray(int_raw)
    except Exception:
        intr = np.asarray(int_raw)
    print('internal force:', intr)
    internalforcearr.append(intr)

    scene.step()
    cam.render()

    # stop recording and save video. If `filename` is not specified, a name will be auto-generated using the caller file name.

# %%

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Convert to arrays
arr_ctrl = np.array(controlforcearr)
arr_int = np.array(internalforcearr)

# Color map and joint labels
colors = cm.get_cmap('tab10')
joint_labels = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']

# Control forces (one line per joint)
plt.figure(figsize=(10, 5))
if arr_ctrl.ndim == 2:
    T, J = arr_ctrl.shape
    for j in range(J):
        plt.plot(range(T), arr_ctrl[:, j], color=colors(j % 10), label=joint_labels[j])
else:
    plt.plot(range(len(arr_ctrl)), arr_ctrl, color='blue', label='Control')
plt.xlabel('Time Step')
plt.ylabel('Control Torque')
plt.title('Control Torque vs. Time')
plt.grid(True)
plt.legend()
plt.savefig('control_torque.png')
print('Saved control plot to control_torque.png')
# Internal forces (one line per joint)
plt.figure(figsize=(10, 5))
if arr_int.ndim == 2:
    T, J = arr_int.shape
    for j in range(J):
        plt.plot(range(T), arr_int[:, j], color=colors(j % 10), label=joint_labels[j])
else:
    plt.plot(range(len(arr_int)), arr_int, color='orange', label='Internal')
plt.xlabel('Time Step')
plt.ylabel('Internal Torque')
plt.title('Internal Torque vs. Time')
plt.grid(True)
plt.legend()
plt.savefig('internal_torque.png')
print('Saved internal plot to internal_torque.png')

# Show and block so windows remain until you close them
try:
    plt.show(block=True)
except TypeError:
    plt.show()

try:
    input('Press Enter to close plots and finish...')
except Exception:
    pass

cam.stop_recording(save_to_filename='video.mp4', fps=60)
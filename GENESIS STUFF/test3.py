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


ur5e = scene.add_entity(
    gs.morphs.MJCF(
        file  = 'xml/universal_robots_ur5e/ur5e.xml',
    ),
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
    kp = np.array([4500, 4500, 4500, 4500, 4500, 4500]),
    dofs_idx_local = dofs_idx,
)
i = 0
j = 0
k = 0

controlforcearr = []
internalforcearr = []
########################################
while i * j * k < 360 * 180 * 160:
    ur5e.set_dofs_position(np.array([math.radians(0), math.radians(0), math.radians(0), 0, 0, 0]), dofs_idx)
    for i in range(360):
        for j in range(180):
            for k in range(160):
                ur5e.set_dofs_position(np.array([math.radians(i), math.radians(j), math.radians(k), 0, 0, 0]), dofs_idx)

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

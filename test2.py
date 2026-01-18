import numpy as np
import genesis as gs

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
plane = scene.add_entity(
    gs.morphs.Plane(),
)
ur5e = scene.add_entity(
    gs.morphs.MJCF(
        file  = 'xml/universal_robots_ur5e/ur5e.xml',
    ),
)
########################## build ##########################
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
# Hard reset
for i in range(150):
    if i < 50:
        ur5e.set_dofs_position(np.array([1, 1, 0, 0, 0, 0]), dofs_idx)
    elif i < 100:
        ur5e.set_dofs_position(np.array([-1, 0.8, 1, -2, 1, 0.5]), dofs_idx)
    else:
        ur5e.set_dofs_position(np.array([0, 0, 0, 0, 0, 0]), dofs_idx)
    scene.step()

# PD control
for i in range(1250):
    if i == 0:
        ur5e.control_dofs_position(
            np.array([1, 1, 0, 0, 0, 0]),
            dofs_idx,
        )
    elif i == 250:
        ur5e.control_dofs_position(
            np.array([-1, 0.8, 1, -2, 1, 0.5]),
            dofs_idx,
        )
    elif i == 500:
        ur5e.control_dofs_position(
            np.array([0, 0, 0, 0, 0, 0]),
            dofs_idx,
        )
    elif i == 750:
        # control first dof with velocity, and the rest with position
        ur5e.control_dofs_position(
            np.array([0, 0, 0, 0, 0, 0])[1:],
            dofs_idx[1:],
        )
        ur5e.control_dofs_velocity(
            np.array([1.0, 0, 0, 0, 0, 0])[:1],
            dofs_idx[:1],
        )
    elif i == 1000:
        ur5e.control_dofs_force(
            np.array([0, 0, 0, 0, 0, 0]),
            dofs_idx,
        )
    # This is the control force computed based on the given control command
    # If using force control, it's the same as the given control command
    print('control force:', ur5e.get_dofs_control_force(dofs_idx))

    # This is the actual force experienced by the dof
    print('internal force:', ur5e.get_dofs_force(dofs_idx))

    scene.step()
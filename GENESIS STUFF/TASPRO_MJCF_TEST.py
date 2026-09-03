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

TASPRO = scene.add_entity(
    gs.morphs.MJCF(
        file  = 'xml/TEST/test2.xml',
    ),
)

plane = scene.add_entity(gs.morphs.Plane())

scene.build()

while True:
    scene.step()

#Add custom test obj/stl file to UR5e MJCF model and XML file
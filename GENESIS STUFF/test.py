import genesis as gs
import numpy as np
########################################init
print(gs.__path__) # check if this is as expected
gs.init(backend=gs.cuda)

#######################################scene create
scene = gs.Scene(
    show_viewer=True, 
    sim_options=gs.options.SimOptions(
        dt = 0.01,
        gravity = (0, 0, -9.81)
        )
    )

####plane
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(
        file='xml/TASPRO/TASPRO.xml',
        pos = (0,0,0),
        euler = (0,0,0),
        scale = 1.0
        )
)

scene.build()

while True:
    scene.step()
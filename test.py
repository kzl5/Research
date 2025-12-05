import genesis as gs
print(gs.__path__) # check if this is as expected
gs.init(backend=gs.cpu)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/universal_robots_ur5e/ur5e.xml'),
)

scene.build()

while True:
    scene.step()
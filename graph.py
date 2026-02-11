import numpy as np
import matplotlib as plt
from test3 import controlforcearr, dofforcearr

timearr = np.arange(len(controlforcearr))

plt.plot(timearr, controlforcearr, label='control force')
plt.xlabel('Time step')
plt.ylabel('Force (N)')
plt.title('Control Force vs DOF Force over Time')
plt.legend('Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6')
plt.show()

plt.plot(timearr, dofforcearr, label='dof force')
plt.xlabel('Time step')
plt.ylabel('Force (N)')
plt.title('DOF Force vs Time')
plt.legend('Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6')
plt.show()
import h5py
import numpy as np
import matplotlib.pyplot as plt
import csv

file_name = "beam_shooting.h5"
file = h5py.File(file_name)

time = np.array(file["time"]).reshape(-1, 1)
n_pts = file['number_of_iterations'].len()
n_nodes = len([i for i in list(file.keys()) if 'node' in i])

xyz = np.zeros([3, n_nodes, n_pts])
v_xyz = np.zeros([3, n_nodes, n_pts])

for i in range(n_nodes):
    node_name = "node_" + str(i)
    motion = file[node_name + "/MOTION"]
    velocity = file[node_name + "/VELOCITY"]
    xyz[:, i, :] = np.array(motion[:3, :])
    v_xyz[:, i, :] = np.array(velocity[:3, :])

# write solution at time=end to file to compare with x0 for shooting
# nb: solution we want is z coordinate only
x_end = np.concatenate((xyz[2, :, -1], v_xyz[2, :, -1]))
np.savetxt('x_end.dat', x_end)
print('Post-Processing Finished')

# csv and plot
# with open('results.csv', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)
#     node_to_write = 16
#     for i in range(len(time)):
#         writer.writerow([time[i,0], xyz[2, node_to_write, i]])
#
# plt.plot(time, xyz[2, node_to_write, :])
# plt.xlabel('time (s)')
# plt.ylabel('Z (m)')
# plt.title('Beam Displacement: Node ' + str(node_to_write))
# plt.grid(True)
# plt.tight_layout()
# plt.show()

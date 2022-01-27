import h5py
import numpy as np


def postprocess(file_name, node_xyz):

    file = h5py.File(file_name + '.h5')

    time = np.array(file["time"]).reshape(-1, 1)
    n_pts = file['number_of_iterations'].len()
    n_nodes = len([i for i in list(file.keys()) if 'node' in i])

    # Initialise arrays
    xyz = np.zeros([3, n_nodes, n_pts])
    v_xyz = np.zeros([3, n_nodes, n_pts])
    theta_xyz = np.zeros([3, n_nodes, n_pts])
    v_theta_xyz = np.zeros([3, n_nodes, n_pts])

    for i in range(n_nodes):
        node_name = "node_" + str(i)
        motion = file[node_name + "/MOTION"]
        velocity = file[node_name + "/VELOCITY"]

        # Subtract nodal values from Motion to get absolute displacements
        xyz[:, i, :] = np.array(motion[:3, :]) - node_xyz[i, :].repeat(n_pts).reshape(3, n_pts)
        theta_xyz[:, i, :] = np.array(motion[4:, :])
        v_xyz[:, i, :] = np.array(velocity[:3, :])
        v_theta_xyz[:, i, :] = np.array(velocity[3:, :])

        # Ensure clamped nodes are exactly 0
        if i == 0 or i == 1:
            xyz[:, i, :] = abs(xyz[:, i, :]) * 0
            theta_xyz[:, i, :] = abs(theta_xyz[:, i, :]) * 0
            v_xyz[:, i, :] = abs(v_xyz[:, i, :]) * 0
            v_theta_xyz[:, i, :] = abs(v_theta_xyz[:, i, :]) * 0

    # write solution at time=T (ie end) to file to compare with x0 for shooting
    xz_T = np.stack((xyz[0, :, -1], xyz[2, :, -1], theta_xyz[1, :, -1]), axis=1).flatten()
    v_T = np.stack((v_xyz[0, :, -1], v_xyz[2, :, -1], v_theta_xyz[1, :, -1]), axis=1).flatten()
    xT = np.concatenate((xz_T, v_T), axis=0)

    np.savetxt('xT.dat', xT, fmt='%.18f')
    print('Post-Processing Finished')


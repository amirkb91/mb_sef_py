import h5py
import numpy as np


def postprocess(file_name, node_xyz):

    file = h5py.File(file_name + '.h5')

    time = np.array(file["time"]).reshape(-1, 1)
    n_pts = file['number_of_iterations'].len()
    # nodes_list = [i for i in list(file.keys()) if 'node' in i]
    n_nodes = len([i for i in list(file.keys()) if 'node' in i])

    # Initialise arrays
    xyz = np.zeros([3, n_nodes, n_pts])
    v_xyz = np.zeros([3, n_nodes, n_pts])
    theta_xyz = np.zeros([3, n_nodes, n_pts])
    v_theta_xyz = np.zeros([3, n_nodes, n_pts])

    for i in range(n_nodes):
        node_name = "node_" + str(i+2)
        motion = file[node_name + "/MOTION"]
        velocity = file[node_name + "/VELOCITY"]

        # Store displacements
        # Subtract nodal values from Motion to get absolute displacements
        xyz[:, i, :] = np.array(motion[:3, :]) - node_xyz[i+2, :].repeat(n_pts).reshape(3, n_pts)
        # Find rotational displacements from quaternion (only theta_y so quaternion axis is [0,1,0]
        e0 = np.array(motion[3, :])
        ey = np.array(motion[5, :])
        theta_xyz[1, i, :] = np.arcsin(ey) * 2

        # Store velocities
        v_xyz[:, i, :] = np.array(velocity[:3, :])
        v_theta_xyz[:, i, :] = np.array(velocity[3:, :])

    # write solution at time=T (ie end) to file to compare with x0 for shooting
    xz_T = np.stack((xyz[0, :, -1], xyz[2, :, -1], theta_xyz[1, :, -1]), axis=1).flatten()
    v_T = np.stack((v_xyz[0, :, -1], v_xyz[2, :, -1], v_theta_xyz[1, :, -1]), axis=1).flatten()
    xT = np.concatenate((xz_T, v_T), axis=0)

    xT_name = 'IO/xT_' + file_name.split('IO/lg_')[1] + '.dat'
    np.savetxt(xT_name, xT, fmt='%.18f')
    print('Post-Processing Finished')


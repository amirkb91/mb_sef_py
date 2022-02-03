import numpy as np
import os

from mb_sef_py.core import Model, NodalFrame
from mb_sef_py.elements import BeamProperties_EIGJ, discretize_beam, ClampedFrameProperties, ExternalForceProperties
from mb_sef_py.math import Frame, UnitQuaternion
from mb_sef_py.solvers import TimeIntegrationParameters, GeneralizedAlpha
from mb_sef_py.utils import Logger, SensorNode, LogNodalFields
import post_processing

os.chdir(os.path.dirname(__file__))

# Beam Properties
E, G, rho = 2.1e11, 8.e10, 7850
b = .01
A = b**2
I = b**4 / 12
J = 2*I
beam_props = BeamProperties_EIGJ(EA=E*A, GA_1=G*A, GA_2=G*A,
                                 GJ=G*J, EI_1=E*I, EI_2=E*I,
                                 m=rho*A, m_11=rho*J, m_22=rho*I, m_33=rho*I)

# Beam Model
model = Model()
p0 = np.array([0., 0., 0.])
p1 = np.array([1., 0., 0.])
f_0 = Frame(x=p0)
node_0 = model.add_node(NodalFrame, f_0)
f_1 = Frame(x=p1)
node_1 = model.add_node(NodalFrame, f_1)

# Clamped-Clamped Boundary Condition
cl_props = ClampedFrameProperties()
model.add_element(cl_props, node_0)
model.add_element(cl_props, node_1)

# Mesh
number_of_element = 20
discretize_beam(model, node_0, node_1, number_of_element, beam_props)

# List of node coordinates
n_free_nodes = number_of_element + 1 - 2  # two clamped nodes
node_xyz = np.zeros([number_of_element + 1, 3])
for i in range(number_of_element+1):
    node_xyz[i, :] = np.copy(model.get_node(0, i).frame_0.x)

# Apply Initial Conditions
# Read x0 and T0 from MATLAB file
# T0 is stored in first line, followed by x0 (position and velocity)
f = open('x0T.dat', 'r')
mat_in = f.readlines()
T0 = float(mat_in[0])
x0 = np.asarray(mat_in[1:]).astype(float)
f.close()

for i in range(2, number_of_element+1):
    # Initial Displacements
    # Copy current frame
    node_x = np.copy(model.get_node(0, i).frame_0.x)
    # Update coordinates by adding displacements from file
    node_x[0] = node_x[0] + x0[3*(i-2)]
    node_x[2] = node_x[2] + x0[3*(i-2)+1]
    # Calculate quaternion for theta_y rotational DoF
    e0 = np.cos(x0[3*(i-2)+2]/2)  # cos(alpha/2)
    e = np.sin(x0[3*(i-2)+2]/2) * np.array([0., 1., 0.])  # sin(alpha/2) * rotation axis(Y)
    q = UnitQuaternion(e0=e0, e=e)
    # Create and set frame with IC applied
    frame_0 = Frame(x=node_x, q=q)
    model.get_node(0, i).set_frame_0(frame_0)

    # Initial Velocities
    j = 3 * (i - 2 + n_free_nodes)
    v_vec_0 = np.array([x0[j], 0., x0[j+1], 0., x0[j+2], 0.])
    model.get_node(0, i).set_initial_velocity(v_vec_0)

# Log
logfile = 'beam_shooting'
logger = Logger(logfile, periodicity=1)
log_nodes = list(range(2, number_of_element+1))
for i in log_nodes:
    log_node = model.get_node(0, i)
    logger.add_sensor(SensorNode(log_node, LogNodalFields.MOTION))
    logger.add_sensor(SensorNode(log_node, LogNodalFields.VELOCITY))

# Time integration
time_integration_parameters = TimeIntegrationParameters()
time_integration_parameters.rho = 1.0  # rho = 1 ensures no damping
time_integration_parameters.T = T0
time_integration_parameters.h = 2.e-4
time_integration_parameters.tol_res_forces = 1.e-6
integrator = GeneralizedAlpha(model, time_integration_parameters, logger)
integrator.solve()

# Post Processing
post_processing.postprocess(logfile, node_xyz)

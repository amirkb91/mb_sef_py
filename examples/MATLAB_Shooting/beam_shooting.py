import numpy as np

from mb_sef_py.core import Model, NodalFrame
from mb_sef_py.elements import BeamProperties_EIGJ, discretize_beam, ClampedFrameProperties, ExternalForceProperties
from mb_sef_py.math import Frame, UnitQuaternion
from mb_sef_py.solvers import TimeIntegrationParameters, GeneralizedAlpha
from mb_sef_py.utils import Logger, SensorNode, LogNodalFields

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

s_0 = np.array([1., 0., 0.])
n_0 = np.array([0., 1., 0.])
b_0 = np.array([0., 0., 1.])
q_0 = UnitQuaternion()
q_0.set_from_triad(s_0, n_0, b_0)
q_1 = UnitQuaternion()
q_1.set_from_triad(s_0, n_0, b_0)
f_0    = Frame(x=p0, q=q_0)
node_0 = model.add_node(NodalFrame, f_0)
f_1    = Frame(x=p1, q=q_1)
node_1 = model.add_node(NodalFrame, f_1)

cl_props = ClampedFrameProperties()
model.add_element(cl_props, node_0)
model.add_element(cl_props, node_1)

number_of_element = 30
discretize_beam(model, node_0, node_1, number_of_element, beam_props)

# Initial Conditions
# Read x0 and T0 from MATLAB file
# T0 is stored in first line, followed by x0 (position and velocity)
f = open('x0t0.dat', 'r')
mat_in = f.readlines()
T0 = float(mat_in[0])
x0 = list(np.float_(mat_in[1:]))
f.close()

for i in range(number_of_element + 1):
    # Initial Displacements
    x_vec_0 = np.copy(model.get_node(0, i).frame_0.x)
    x_vec_0[2] = x0[i]  # update z coordinate to existing solution
    frame_0 = Frame(x=x_vec_0)  # create frame with new vec_0
    model.get_node(0, i).set_frame_0(frame_0)

    # Initial Velocity
    v_vec_0 = np.array([0., 0., x0[i + number_of_element + 1], 0., 0., 0.])
    model.get_node(0, i).set_initial_velocity(v_vec_0)

# Log
logger = Logger('beam_shooting', periodicity=1)
log_nodes = list(range(0, number_of_element+1))
for i in range(len(log_nodes)):
    log_node = model.get_node(0, log_nodes[i])
    logger.add_sensor(SensorNode(log_node, LogNodalFields.MOTION))
    logger.add_sensor(SensorNode(log_node, LogNodalFields.VELOCITY))

# Time integration
time_integration_parameters = TimeIntegrationParameters()
time_integration_parameters.rho = 1.0  # rho = 1 ensures no damping
time_integration_parameters.T = T0
time_integration_parameters.h = 2.e-4
time_integration_parameters.tol_res_forces = 1.e-5
integrator = GeneralizedAlpha(model, time_integration_parameters, logger)
integrator.solve()

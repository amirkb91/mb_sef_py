import numpy as np

from mb_sef_py.core import Model, NodalFrame
from mb_sef_py.elements import BeamProperties_EIGJ, discretize_beam, ClampedFrameProperties, ExternalForceProperties
from mb_sef_py.math import Frame, UnitQuaternion
from mb_sef_py.solvers import TimeIntegrationParameters, GeneralizedAlpha
from mb_sef_py.utils import Logger, SensorNode, LogNodalFields


# External Loading
def loading(t):
    force = 0
    freq  = 100
    loads = np.zeros((6, ))
    loads[2] = force * np.sin(freq * t)
    return loads


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

# The following s_0 n_0 b_0 gives a UnitQuaternion with e0=1 and e=[0,0,0]
# which is the default values from the class constructor
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
# model.add_element(cl_props, node_1)

number_of_element = 30
discretize_beam(model, node_0, node_1, number_of_element, beam_props)

# Initial velocity
# node_1.set_initial_velocity(np.array([0., 0., .5, 0., 0., 0.]))

# Initial deformation
# tip node only
frame_0x = np.copy(node_1.frame_0.x)
frame_0x[2] = 1e-5
Frame_0 = Frame(x=frame_0x)
node_1.set_frame_0(Frame_0)

# --- Read Initial deformation from File
# x0_sol = []
# with open('x0.dat') as x0_file:
#     for line in x0_file:
#         x0_sol.append(float(line))
#
# for i in range(number_of_element + 1):
#     vec_0 = np.copy(model.get_node(0, i).frame_0.x)
#     vec_0[2] = x0_sol[i]  # update z coordinate to existing solution
#     Frame_0 = Frame(x=vec_0)  # create frame with new vec_0
#     model.get_node(0, i).set_frame_0(Frame_0)

# External Force
ef_node = 6
ef_node = model.get_node(0, ef_node)
ef_props = ExternalForceProperties()
ef_props.time_dependent_force = loading
model.add_element(ef_props, ef_node)

# Log
logger = Logger('beam_cclamped', periodicity=1)
log_nodes = [1, 16]
for i in range(len(log_nodes)):
    log_node = model.get_node(0, log_nodes[i])
    logger.add_sensor(SensorNode(log_node, LogNodalFields.MOTION))

# Time integration
time_integration_parameters = TimeIntegrationParameters()
time_integration_parameters.rho = .95
time_integration_parameters.T = .01
time_integration_parameters.h = 1.e-3
time_integration_parameters.tol_res_forces = 1.e-5
integrator = GeneralizedAlpha(model, time_integration_parameters, logger)
integrator.solve()

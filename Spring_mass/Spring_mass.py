import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.integrate import solve_ivp

from sympy.physics.mechanics import LagrangesMethod, Lagrangian
from sympy.physics.mechanics import ReferenceFrame, Particle, Point
from sympy.physics.mechanics import dynamicsymbols
from sympy import symbols, lambdify

# Define generalised coordinates, q(t) and \dot q(t) as dynamic symbols
q = dynamicsymbols('q')
qd = dynamicsymbols('q', 1)

# Define symbols for constants
m, k, b, g = symbols('m k b g')

# Setup reference frame, 
N = ReferenceFrame('N')

# Create a point and set its velocity in the frame
P = Point('P')
P.set_vel(N, -qd * N.y)

# Create a particle with point P attached to it, with mass m.
Pa = Particle('Pa', P, m)

# Define the Potential Energy of the particle 
Pa.potential_energy = -m * g * q + k * q**2 / 2

# Formulate the Lagrangian
L = Lagrangian(N, Pa)

# Non-conservative force list: Damping
fl = [(P, b * qd * N.y)]

# Generate the Euler-Lagrange equations of motion (E-L EoM) such that E-L EoM = 0
LM = LagrangesMethod(L, [q], forcelist = fl, frame = N)
LM.form_lagranges_equations()

# Solve for states and lambdify 
rhs = LM.rhs()
rhs_func = lambdify([q,qd,m,k,b, g], rhs)

# Define ODE
def ODE(t, y):
 q_val, qd_val = y
 return rhs_func(q_val, qd_val, m, k, b, g).flatten()

# Set Constants & Initial conditions 
m = 1       # Mass [kg]
k = 20      # Spring constant [N/m]
b = 0.55    # Damping coefficient [(N s)/m]
g = -9.8    # Gravitational acceleration [m/(s^2)]

q0 = 0      # Generalised initial position q(0) = 0 [m]
qd0 = 0     # Generalised initial velocity qd(0) = 0 [m/s]

t_f = 10    # Animation length in seconds
fps = 30    # Set the frames per second for animation

# Solve the ODE
sol = solve_ivp(ODE, [0, t_f], (q0, qd0), t_eval=np.linspace(0, t_f, t_f*fps+1))

# Extract the solutions: y = q(t), yd = qd(t) 
y, yd = sol.y
t = sol.t

# --- Animations --- 

# Setup FFMpeg writer
ffmpeg_writer = animation.FFMpegWriter(fps = fps)

# Animate Spring Mass System

fig, ax = plt.subplots()
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(min(y)-0.125, max(y)+0.125)
ax.set_xticks([],[])
ax.set_title('Spring-Mass System')

spring, = ax.plot([0, 0], [0, y[0]], 'b', lw=2)
mass, = ax.plot(0,y[0], 'ro')
elapsed_time = ax.text(0.05, 0.95, '', ha = 'left', va = 'top', transform=ax.transAxes)

def update(frame):
 spring.set_ydata( [0, y[frame]])
 mass.set_ydata( [y[frame]])
 elapsed_time.set_text(f'Elapsed Time: {t[frame]:6.2f}')
 return spring, mass, elapsed_time

spring_mass_anim = animation.FuncAnimation(fig, update, frames = len(t), interval = 30, blit = True)
spring_mass_anim.save('Output/Spring_mass.gif', writer=ffmpeg_writer)

# Animate Phase Diagram

fig, ax = plt.subplots()
ax.set_title('Spring-Mass System: Phase Space Diagram')
ax.set_xlabel(r'$ q \left( t \right) \ \left[ m \right]$', weight = 'bold')
ax.set_ylabel(r'$ \dot q \left( t \right) \ \left[ m \ s^{-1} \right]$', weight = 'bold')
ax.set_xlim(min(y)-0.1,max(y)+0.1)
ax.set_ylim(min(yd)-0.1,max(yd)+0.1)
ax.grid()

phase_curve, = ax.plot(y[0], yd[0], color = 'blue')
phase_dot, = ax.plot(y[0], yd[0], 'ro')

def update_phase_curve(frame):
 phase_curve.set_data( y[:frame+1], yd[:frame+1] )
 phase_dot.set_data( [y[frame]], [yd[frame]])
 return phase_curve, phase_dot

phase_curve_anim = animation.FuncAnimation(fig, update_phase_curve, frames = len(t), interval = 30, blit = True)
phase_curve_anim.save('Output/Phase_curve.gif', writer=ffmpeg_writer)

# Animate q(t) and \dot q(t) vs time

fig, ax = plt.subplots()
ax.set_title(f'Spring-Mass System: $\\frac{{mg}}{{k}} = {m*g/k:.2f}$')
ax.set_xlabel(r'$t \ \left[s \right]$', weight = 'bold')
ax.set_ylabel(r'$q\left( t \right) \ \left[m \right], \dot q \left( t \right) \ \left[m \ s^{-1} \right] $', weight = 'bold')
ax.set_xlim(0, t_f)
ax.set_ylim(min(yd)-0.1,max(yd)+0.1)
ax.grid()

q_t, = ax.plot(t[0], y[0], color = 'blue', label = r'$q \left( t \right)$')
qd_t, = ax.plot(t[0], yd[0], 'r', label = r'$\dot q \left( t \right)$')
ax.legend(prop = {'weight':'bold', 'size': 12})

def update_q(frame):
 q_t.set_data( t[:frame+1], y[:frame+1] )
 qd_t.set_data( t[:frame+1], yd[:frame+1])
 return q_t, qd_t

time_domain_anim = animation.FuncAnimation(fig, update_q, frames = len(t), interval = 30, blit = True)
time_domain_anim.save('Output/Time_domain.gif', writer = ffmpeg_writer)
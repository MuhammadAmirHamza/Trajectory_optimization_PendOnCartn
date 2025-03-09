#%% diff equation solver
from scipy.integrate import odeint
import numpy as np

# odeint is used to solve first order equation
# consider : theta''(t) + b*theta'(t) + c*sin(theta(t)) = 0
# omega = theta'(t)
# then equations are theta'(t) = omega ; omega'(t) = -b * omega'(t) - c*sin(theta(t))

# zero input response
def pend(y, t, b, c):
    theta, omega = y
    dydt = [omega, -b * omega - c * np.sin(theta)]
    return dydt

# time 
t = np.linspace(0, 5, 500)
y0 = [np.pi, 0.1]

b = 0.25
c = 5.0

sol = odeint(pend, y0, t, args=(b, c))

import matplotlib.pyplot as plt
plt.plot(t, sol[:, 0], 'b', label='theta(t)')
plt.plot(t, sol[:, 1], 'g', label='omega(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

plt.figure()
plt.plot(sol[:, 0], sol[:, 1])
plt.show()

#%% simulating he whole pendulum system

def pend(state, t, u):
    # Parameters
    M = 1.0  # Mass of the cart (kg)
    m = 0.2  # Mass of the pendulum (kg)
    L = 0.5  # Length of the pendulum (m)
    g = 9.81  # Gravity (m/s^2)

    x, theta, x_dot, theta_dot = state

    denom = M + m*np.sin(theta)**2
    state_dot = [x_dot, 
                 theta_dot,
                 (u - m * L * theta_dot ** 2 * np.sin(theta) + m * g * np.cos(theta) * np.sin(theta))/denom,
                 (-u * np.cos(theta) + m*L*theta_dot**2 * np.cos(theta) * np.sin(theta) - (M + m) * g * np.sin(theta))/(L*denom),
                ] 

    return state_dot

state0 = [0, 0.1, 0.1, 0]
t = np.linspace(0, 10, 1001)
u = np.zeros(t.shape)
u[200: 300] = 1

solution = np.zeros(len(state0)).reshape(1, -1)
solution = state0

for i in range(1, len(t)):
    tspan = t[i-1:i + 1]
    # controller can be implemented here
    sol = odeint(pend, state0, tspan, args = (u[i-1],))
    solution = np.vstack([solution, sol[-1]])
    state0 = sol[-1]


# Plot results
plt.figure()
plt.plot(t, solution[:, 0], label='Cart Position (x)')
plt.plot(t, solution[:, 1], label='Pendulum Angle (θ)')
plt.xlabel('Time (s)')
plt.ylabel('State')
plt.legend()
plt.title('Pendulum on a Cart Dynamics')
plt.grid()
plt.show()


plt.figure()
plt.plot(t, u)
plt.plot(t, solution[:, 2], label='Cart vel (x_dot)')
plt.plot(t, solution[:, 3], label='Pendulum vel (θ_dot)')
plt.xlabel('Time (s)')
plt.ylabel('State')
plt.legend()
plt.title('Pendulum on a Cart Dynamics')
plt.grid()
plt.show()

#%% solving the pendulum system using gekko
from gekko import GEKKO
# problem  : 5dy/dt = -y(t) + u(t) ; y(0) = 1 ; u = 2 * u(t-10)

m = GEKKO(remote = False)
fs = 250
m.time = np.linspace(0, 5, 5*fs + 1)

state0 = [0, np.pi, 0, 0]
x = m.Var(value = state0[0])
theta = m.Var(value = state0[1])
x_dot = m.Var(value = state0[2])
theta_dot = m.Var(value = state0[3])

input = np.zeros(m.time.shape)
input[2*fs: 3*fs] = 1
F = m.MV(value = input)
F.STATUS = 0

# sys parameters
M = 1.0  # Mass of the cart (kg)
mp = 0.2  # Mass of the pendulum (kg)
L = 0.5  # Length of the pendulum (m)
g = 9.81  # Gravity (m/s^2)

# Equations of motion
denom = M + mp * m.sin(theta)**2  # Denominator
m.Equation(x.dt() == x_dot)  # dx/dt = x_dot
m.Equation(theta.dt() == theta_dot)  # dtheta/dt = theta_dot
m.Equation(x_dot.dt() == (F - mp * L * theta_dot**2 * m.sin(theta) + mp * g * m.cos(theta) * m.sin(theta)) / denom)  # d(x_dot)/dt
m.Equation(theta_dot.dt() == (-F * m.cos(theta) + mp * L * theta_dot**2 * m.cos(theta) * m.sin(theta) - (M + mp) * g * m.sin(theta)) / (L * denom))  # d(theta_dot)/dt

m.options.IMODE = 4
m.options.SOLVER = 1
m.solve()

# Plot results
plt.figure(figsize=(10, 6))

# Plot cart position (x)
plt.subplot(2, 1, 1)
plt.plot(m.time, x.value, label='Cart Position (x)', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid()

# Plot pendulum angle (theta)
plt.subplot(2, 1, 2)
plt.plot(m.time, theta.value, label='Pendulum Angle (θ)', color='r')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
#%%
from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

# Model
m = GEKKO(remote=False)  # Use remote=False for faster debugging

# Time discretization (fine grid)
m.time = np.linspace(0, 10, 5001)

# Parameters
g = 9.81  # Gravity
L = 1.0   # Pendulum length
mc = 2.0  # Cart mass
mp = 0.5  # Pendulum mass

# Variables (with good initial guesses)
theta = m.Var(value=np.radians(80))  # Initial angle (80 degrees) - IN RADIANS!
theta_dot = m.Var(value=0)
x = m.Var(value=0)
x_dot = m.Var(value=0)
u = m.MV(value=0) # No force for this case.
u.STATUS = 0

# Equations of Motion (Lagrangian formulation - example)
#  (Double-check these equations - this is just an example!)
m.Equation(x_dot.dt() == (mp*L*theta_dot**2*m.sin(theta) - mp*g*m.sin(theta)*m.cos(theta) + u) / (mc + mp*(1-m.cos(theta)**2)))
m.Equation(theta_dot.dt() == (-mp*L*theta_dot**2*m.sin(theta)*m.cos(theta)  + (mc+mp)*g*m.sin(theta) - u*m.cos(theta))/(L*(mc + mp*(1-m.cos(theta)**2))))
m.Equation(x.dt() == x_dot)
m.Equation(theta.dt() == theta_dot)

# Solver Settings
m.options.IMODE = 4  # Dynamic simulation
m.options.SOLVER = 1  # APOPT (try 3 for IPOPT if needed)
# m.options.OTOL = 1e-8  # Tighter tolerance (optional - start without this)
# m.options.RTOL = 1e-8

# Solve
m.solve(disp=True)

# Plot results
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(m.time, np.degrees(theta.value), 'b-', label='Theta (degrees)') # Convert back to degrees for plotting
plt.ylabel('Theta (degrees)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(m.time, x.value, 'r-', label='Cart Position (x)')
plt.xlabel('Time (s)')
plt.ylabel('x')
plt.legend()
plt.grid(True)
plt.show()

#%%
"""
Implementation of different dynamical models. The return type is a casadi system object.
Implemented models:
1. Pendulum on a cart
"""
from casadi import *
import numpy as np
#%%
class PendulumOnCart():
    """
    Simulates and returns the dynamics of pendulum on a cart
    """
    def __init__(self, m_c = 1, m_p = 0.1, l = 0.5, g = 9.8):
        """Initialization

        Args:
            m_c (int, optional): mass of cart. Defaults to 1.
            m_p (float, optional): mass of pendulum. Defaults to 0.1.
            l (float, optional): length of pendulum. Defaults to 0.5.
            g (float, optional): gravity. Defaults to 9.8.
        """
        self.m_c = m_c
        self.m_p = m_p
        self.l = l
        self.g = g

    def state_names(self):
        print("lin_pos, ang_pos, lin_vel, ang_vel")

    def model(self):
        """
        Returns the symbolic model of the plant to be used with casadi
        """ 
        # states
        x = MX.sym('x')
        theta = MX.sym('theta')
        v = MX.sym('v')
        omega = MX.sym('omega')
        u = MX.sym('u')

        # system parameter
        m_c = self.m_c
        m_p = self.m_p 
        l = self.l 
        g = self.g 

        # dynamics 
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        denominator = m_c + m_p * sin_theta**2

        # Accelerations
        x_ddot = (m_p * l * sin_theta * omega**2 + u - m_p * g * cos_theta * sin_theta) / denominator
        theta_ddot = ((m_c + m_p) * g * sin_theta - cos_theta * (m_p * l * sin_theta * omega**2 + u)) / (l * denominator)

        # differential equations
        x_dot = v
        theta_dot = omega
        v_dot = x_ddot
        omega_dot = theta_ddot

        self.ode = vertcat(x_dot, theta_dot, v_dot, omega_dot)
        self.x = vertcat(x, theta, v, omega)

        self.sys = {'ode' : self.ode, 'x' : self.x, 'p' : u}
        return self.sys
    
    def integrate(self, t0 = 0, tf = 1, x_0 = [0, 0, 0, 0], u = 1,  steps = None):
        
        self.model()
        if steps == None:
            opts = {'t0' : t0, 'tf' : tf}
            intg = integrator('full_horizon', 'cvodes', self.sys, opts)
            final_state = intg(p = u, x0 = x_0)
            return final_state
        else:
            # if simulation is in steps
            grid = np.linspace(t0, tf, steps + 1)
            opts = {'t0': t0, 'tf' : tf, 'grid' : grid}
            intg = integrator('gird_integrator', 'cvodes', self.sys, opts)
            states = intg(p = u, x0 = x_0) 
            return states

if __name__ == "__main__":
    obj = PendulumOnCart()
    print(obj.integrate(tf = 10, steps = 10))
    states = obj.integrate(tf = 10, steps = 100)
    # ang_pos = states[1, :]
    # ang_vel = states[3, :]
    # import matplotlib.pyplot as plt
    # plt.plot(ang_pos, ang_vel)
    # plt.show()
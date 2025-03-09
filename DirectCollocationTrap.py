#%%
import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from Models import PendulumOnCart
import joblib
import os

# class DirectCollocation:
class DirectCollocation:
    """Implements direct collocaiton using Trapezoidal technique"""
    def __init__(self, sys = PendulumOnCart().model(),
                 x0 = [0, pi, 0, 0], xf = [0,0,0,0],
                 Q = np.eye(4) * 4,
                 R = 2,
                 Qf = np.eye(4) * 20,
                 t0 = 0,
                 tf = 5, 
                 N = 250,
                 ):
        self.sys = sys
        self.x0 = DM(x0)
        self.xf = DM(xf)
        self.t0 = t0
        self.tf = tf
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.N = N
        self.dt = (tf - t0) / self.N
    
    def optimize_trapezoidal(self):
        """
        Trapezoidal method
        """
        x = self.sys['x']
        u = self.sys['p']
        sys_dynamics = Function('system_dynamics', [self.sys['x'], self.sys['p']],\
                                [self.sys['ode']])
        cost = Function('system_cost', [self.sys['x'], self.sys['p']],
                        [x.T @ self.Q @ x + u.T @ self.R @ u])
        
        # optimization parameter
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # decision variables
        n = self.sys['x'].size1()
        X = MX.sym('x', n, self.N )
        U = MX.sym('u', 1, self.N )
        w = vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))

        lbw += [-inf] * n * self.N # states constraints
        ubw += [inf] * n * self.N # states constraints
        lbw += [-15] * self.N # control constraints
        ubw += [15] * self.N # control constraints

        # boundar constraints
        g += [X[:, 0] - self.x0]
        lbg += [*np.zeros(n)]
        ubg += [*np.zeros(n)]
        g += [X[:, self.N -1] - self.xf]
        lbg += [*np.zeros(n)]
        ubg += [*np.zeros(n)]

        # trapezoidal constraint
        for i in range(self.N - 1):
            uk  = U[:, i]
            uk1 = U[:, i + 1]
            xk  = X[:, i]
            xk1 = X[:, i + 1]

            fk = sys_dynamics(xk, uk)
            fk1 = sys_dynamics(xk1, uk1)

            g += [xk1 - xk - self.dt/2 * (fk + fk1)]
            lbg += [*np.zeros(n)]
            ubg += [*np.zeros(n)]

            J += cost(xk, uk)
        # terminal cost
        J += xk1.T @ self.Qf @ xk1

        # initial guess (equilibrium point)
        for i in range(n):
            w0 += [float(self.x0[i])] * self.N
        w0 += [0] * self.N
        w0 = DM(w0).reshape((-1, 1))
        
        # nlp problem
        nlp = {'f' : J, 'x' : w, 'g' : vertcat(*g)}

        # solver
        solver = nlpsol('nlp_solver', 'ipopt', nlp)
        solution = solver(x0 = DM(w0), lbx = DM(lbw), ubx = DM(ubw), lbg = DM(lbg), ubg = DM(ubg))
        self.opt_states_trap = solution['x'].full().reshape(n + 1, self.N)[:n, :]
        self.opt_control_trap = solution['x'].full().reshape(n + 1, self.N)[-1, :]
        return self.opt_states_trap, self.opt_control_trap 
    
    
    def simulation(self, u_opt = None, initial_cond = [0, pi - 1, 0, 0]):

        u_opt = np.concatenate((u_opt, np.zeros(1)))
        # preturbed initial state
        pos = np.zeros(self.N)
        vel = np.zeros(self.N)
        theta = np.zeros(self.N)
        theta_dot = np.zeros(self.N)
        state = DM(initial_cond)
        lin_pts = 25
        # discretize the system
        sys_dics = integrator('disc_sys', 'cvodes', self.sys, 0, self.dt/lin_pts)
        for i in range(self.N ):
            uk = u_opt[i]
            uk1 = u_opt[i+1]
            for t in range(lin_pts):
                ut = uk + (uk1 - uk) * t/lin_pts
                state = sys_dics(x0 = state, p = ut)['xf'] 
            pos[i] = float(state[0])
            theta[i] = float(state[1])
            vel[i] = float(state[2])
            theta_dot[i] = float(state[3])

        # for original initial state
        org_pos = np.zeros(self.N)
        org_vel = np.zeros(self.N)
        org_theta = np.zeros(self.N)
        org_theta_dot = np.zeros(self.N)
            
        state = self.x0
        # discretize the system
        for i in range(self.N ):
            uk = u_opt[i]
            uk1 = u_opt[i+1]
            for t in range(lin_pts):
                ut = uk + (uk1 - uk) * t/lin_pts
                state = sys_dics(x0 = state, p = ut)['xf']  
            org_pos[i] = float(state[0])
            org_theta[i] = float(state[1])
            org_vel[i] = float(state[2])
            org_theta_dot[i] = float(state[3])
        
        plt.plot(org_theta, org_theta_dot, label = 'ang. trajectory')
        # Add arrows at specific points
        for i in range(0, len(org_theta), 20):  # Add an arrow every 20 points
            dx = org_theta[i+1] - org_theta[i]
            dy = org_theta_dot[i+1] - org_theta_dot[i]
            plt.arrow(org_theta[i], org_theta_dot[i], dx, dy, 
                    shape='full', lw=0, length_includes_head=True, 
                    head_width=0.5, head_length=0.5, color='red')
        plt.title('Direct Collocation (Trapzoidal)')
        plt.legend()
        plt.xlabel('angular pos.')
        plt.ylabel('angular vel.')
        plt.grid()
        plt.show()

        plt.figure()
        plt.plot(org_theta, org_theta_dot, label = "original initial state")
        plt.plot(theta, theta_dot, label = 'preturbed initial state')
        plt.xlabel('angular pos.')
        plt.ylabel('angular vel.')
        plt.grid()
        plt.title('Direct Collocation (Trapzoidal)')
        plt.legend()
        plt.show()

        plt.figure(1)
        t = np.linspace(self.t0, self.tf, self.N)
        plt.plot(t, u_opt[:-1], label = 'control input')
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.title('Direct Collocation (Trapzoidal)')
        plt.grid()
        plt.legend()
        plt.show()

        # plt.figure()
        # plt.plot(pos)
        # plt.plot(vel)
        # plt.show()

if __name__ == "__main__":
    sys = PendulumOnCart().model()
    shoot = DirectCollocation(N = 250)
    opt_states, opt_control = shoot.optimize_trapezoidal()
shoot.simulation(u_opt = opt_control, initial_cond = [0, pi - 0.2, 0, 0])
    #%%
# plt.plot(opt_states[0, :], label = 'lin_pos')
# plt.plot(opt_states[1, :], label = 'ang_pos')
# plt.plot(opt_states[2, :], label = 'lin_vel')
# plt.plot(opt_states[3, :], label = 'ang_vel')
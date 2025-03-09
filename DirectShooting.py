"""
This script will be implementing the direct optimization methods.
Direct optimization solves the NLP problem directly, without driving optimality conditions
from Pontryagin's Principle.
Open loop trajectory optimization
1. Direct shooting method
2. Direct collocial method
3. Direct multiple shooting method
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from Models import PendulumOnCart
import joblib
import os

#%% Direct Shooting method
class DirectShooting:
    """
    Implements direct shooting method
    """
    def __init__(self, sys = PendulumOnCart().model(),
                 x_0 = [0, pi, 0, 0], x_f = [0, 0, 0, 0],
                 t0 = 0, tf = 1, N = 10,
                 Q = np.eye(4) * 4, R = 2, Qf = np.eye(4) * 20):
        """_summary_

        Args:
            sys                  : system dynamics
            x_0                  : initial state of the system
            x_f                  : final state of the system
            t0                   : initial time
            tf                   : final time
            N                    : num steps
            Q (_type_, optional) : running state cost. Defaults to np.eye(4).
            R (int, optional)    : running control cost. Defaults to 1.
            Qf (_type_, optional): final state cost. Defaults to np.eye(4).
        """
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.sys = sys
        self.x0 = DM(x_0)
        self.xf = DM(x_f)
        self.t0 = t0
        self.tf = tf
        self.N = N
        self.dt = (tf - t0)/N
        # integrating the cost element in sys
        # self.sys['quad'] = sys['x'].T @ Q @ sys['x'] + sys['p'].T @ R @ sys['p']
        self.u_opt = []

    def optimize_numerical_integration(self):
        """
        Direct shooting method
        """
        # discretize the system
        sys_dics = integrator('disc_sys', 'cvodes', self.sys, 0, self.dt)
        # parameters of the optimization
        w = []
        lbw = []
        ubw = []
        w0 = []
        g = []
        lbg = []
        ubg = []
        J = 0

        U = MX.sym('u', 1, self.N)
        w = U.reshape((-1, 1))
        lbw = [-15] * self.N
        ubw = [15] * self.N

        w0 = DM(np.zeros((self.N, 1)))
        x = self.x0        
        for i in range(self.N):
            uk = U[:, i]

            # running cost            
            J += x.T @ self.Q @ x + uk.T @ self.R @ uk
            # system simulation            
            x = sys_dics(x0 = x, p = uk)['xf']

        # terminal cost
        J += x.T @ self.Qf @ x

        # terminal constraints
        g += [x - self.xf]
        lbg += [*np.zeros((1, 4))]         
        ubg += [*np.zeros((1, 4))]

        # nlp problem
        nlp = {'f' : J, 'x' : w, 'g': vertcat(*g)}

        # solver
        solver = nlpsol('nlp_solver', 'ipopt', nlp)

        # solve the problem
        solution = solver(x0 = w0, lbx = DM(lbw), ubx = DM(ubw),\
                            lbg = DM(lbg), ubg = DM(ubg))
        self.u_opt = solution['x'].full().reshape(1, self.N)

        return self.u_opt

    def simulation(self, initial_cond = [0, pi - 1, 0, 0]):

        # preturbed initial state
        pos = np.zeros(self.N)
        vel = np.zeros(self.N)
        theta = np.zeros(self.N)
        theta_dot = np.zeros(self.N)
        state = DM(initial_cond)
        # discretize the systemoptimize_numerical_integration
        sys_dics = integrator('disc_sys', 'cvodes', self.sys, 0, self.dt)
        for i in range(self.N):
            state = sys_dics(x0 = state, p = self.u_opt[0,i]) 
            state = state['xf']
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
        for i in range(self.N):
            state = sys_dics(x0 = state, p = self.u_opt[0, i]) 
            state = state['xf']
            org_pos[i] = float(state[0])
            org_theta[i] = float(state[1])
            org_vel[i] = float(state[2])
            org_theta_dot[i] = float(state[3])
        
        plt.plot(org_theta, org_theta_dot, label = 'ang. trajectory')
        for i in range(0, len(org_theta), 20):  # Add an arrow every 20 points
            dx = org_theta[i+1] - org_theta[i]
            dy = org_theta_dot[i+1] - org_theta_dot[i]
            plt.arrow(org_theta[i], org_theta_dot[i], dx, dy, 
                    shape='full', lw=0, length_includes_head=True, 
                    head_width=0.3, head_length=0.3, color='red')
        plt.title('Direct shooting')
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
        plt.title('Direct shooting')
        plt.legend()
        plt.show()

        plt.figure(1)
        t = np.linspace(self.t0, self.tf, self.N)
        plt.plot(self.u_opt[0, :], label = 'control input')
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.title('Direct shooting')
        plt.grid()
        plt.legend()
        plt.show()
        # plt.figure()
        # plt.plot(pos)
        # plt.plot(vel)
        # plt.show()


    def save_data(self):
        data = {
            "t0" : self.t0,
            "tf" : self.tf,
            "x0" : self.x0,
            "xf" : self.xf,
            "sys" : "pendulum",
            "N" : self.N,
            "Q" : self.Q,
            "R" : self.R,
            "Qf" : self.Qf,
            "u_opt" : self.u_opt
        }
  
        # dynamic naming
        i = 1
        while os.path.exists(f"DirectShootingResults/data_{i}.pkl"):
            i += 1
        joblib.dump(data, f'DirectShootingResults/data_{i}.pkl')
#%%
if __name__ == "__main__":

    obj = DirectShooting(tf = 5, N = 250)
    # opt_u = obj.optimize_numerical_integration()
    # obj.save_data()
    # obj.optimize_euler()
    #%%
    data = joblib.load('DirectShootingResults/data_13.pkl')
    obj.u_opt = data['u_opt']
    obj.simulation(initial_cond=[0, pi - 0.2, 0, 0])

# %%

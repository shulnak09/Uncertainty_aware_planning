import numpy as np
import matplotlib.pyplot as plt 
import math
from typing import Tuple

from cost import *
from dynamics import *

'''
MPPI is sampling-based planner. It does not require gradient descent 
or IPOPT solver. It populates the space with trajectory samples and then
choses the best candidate based on cost
'''

class MPPI:
    def __init__(
            self,
            dt: float = 0.05,
            robot_radius:float = 0.3,
            max_steer:float = 0.5236,
            max_accel:float = 2.0,
            horizon:int = 30,
            num_samples:int = 200,
            param_exploration:float = 0.0,
            param_lambda:float = 50.0,
            param_alpha:float = 1.0,
            sigma:np.ndarray = np.array([0.3, 0],[0, 0.1]),
            Q:np.ndarray = np.array(np.diag([50.0, 50.0, 1.0, 20.0])),
            R:np.ndarray = np.array(np.diag([50.0, 50.0, 1.0, 20.0])),
            collision_thres:float = 0.3

    ):
        """
        dt: time horizon for each step
        robot_radius: radius of the robot
        max_steer: maximum allowable steer for robot (in radians)
        max_accel: maximum allowable acceleration for robot (in m/s^2)
        horizon: number of steps for each trajectory
        num_samples: number of sampled trajectories 
        param_exploration:  exporation parameter for sampling
        param_lamda: param inverse temperature
        param_alpha: 
        sigma: added Gaussian Noise to the control
        Q: weighting matrix for stage cost
        R: weighting matrix for terminal cost
        collision_thres: safe distance from obstacles to avoid collision
        """

        self.dt = dt
        self.T = horizon
        self.K = num_samples
        self.param_exploration = param_exploration
        self.param_lambda = param_lambda
        self.param_alpha = param_alpha
        self.param_gamma = self.param_lambda(1.0 -  self.param_alpha)
        self.sigma = sigma
        self.Q = Q
        self.R = R
        self.collision_thres = collision_thres

        # state params:
        self.x_dim = 4
        self.u_dim = 2

        # Vehicle params:
        self.robot_radius = robot_radius
        self.max_steer = max_steer
        self.max_accel = max_accel

        # MPPI Variables:
        self.u_prev = np.zeros((self.T, self.u_dim))

    
    def  compute_control(self, observed_x):
        
        u = self.u_prev

        # set intial x value from observation
        x0 = observed_x

        # prepare stage cost for K samples
        S = np.zeros((self.K))

        # Sample Noise from Gaussian Distribution and add to control:
        noise = self.compute_noise(self.sigma, self.K, self.T, self.u_dim)

        # Prepare buffer for all control inputs K x nu
        v = np.zeros((self.K, self.T, self.u_dim)) # v_t represents control input sequence with noise

        for k in range(self.K):

            x = x0

            for t in range(1, self.T+1):

                if k < (1.0 -   self.param_alpha) * self.K:
                    v[k, t-1] = u[t-1] + noise[k, t-1] # Sampling for exploration
                
                else:
                    v[k, t-1] = noise[k, t-1] # Exploitation

            
                # Update state:
                x = dynamics(x, self.clamp_control(v[k, t-1]), self.dt)

                # Compute cost associated with each trajectory:
                S[k] += stage_cost(x) + self.param_gamma * u[t-1].T @ np.linalg.inv(self.sigma) @ v[k, t-1]

            S[k] += terminal_cost(x)


        # Compute weights:
        w = self.compute_weights(S)
        
        # Compute the control sequence
        w_epsilon = np.zeros((self.T, self.u_dim))
        for t in range(0, self.T):
            for k in range(self.K):
                w_epsilon[t] += w[k] * noise[k, t]  






    
    
    def compute_noise(self, sigma, samples, horizon, dim_u):
        'Compute the Noise associated with control command'
        mu = np.zeros((dim_u))
        noise = np.random.multivariate_normal(mu, sigma, (samples, horizon))
        return noise
    
    def clamp_control(self, v) -> float:
        'clamp the control inputs to max possible values'
        v[0] = np.clip(v[0], -self.max_steer, self.max_steer)
        v[1] = np.clip(v[1], -self.max_accel, self.max_accel)
        return v
   
    def is_collided(x_t, obs_t) -> bool:
       
        ' check if collision happened between the robot and obstacle '
        ' Both robot and obstacle are circular so collision happens if  |x_t - obs_t| < d_thres '
        x, y, yaw, v = x_t
        x_obs, y_obs = obs_t
        d_thres =  0.2

        collided =  True if np.linalg.norm(x_t[:2] - obs_t, ord=2) < d_thres else False    
        return collided


    def compute_weights(self, S: np.ndarray) -> np.ndarray:

        # Find the minimum cost trajectory
        rho = S.min()

        # Compute eta
        for k in range(self.K):
            eta += np.exp((-1.0/self.param_lambda) *(S[k] - rho))

        # Compute weights:
        w = np.zeros((self.K))
        for k in range(self.K):
            w[k] = (1.0/eta) * np.exp( (-1.0/self.param_lambda) *(S[k] -rho))

        return w
    
    def Savitzky_Golay_filter(self):
        pass
         


        









        
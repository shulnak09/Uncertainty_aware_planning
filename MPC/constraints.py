'''
This file contains different types of constraint tightening:
1. Hard Constraints
2. Safety filter like control Barrier Functions
3. Chance Constraints
'''
import casadi as ca
import numpy as np
from utils import *


class robot_constraint:
    def __init__(self, X, P, n_states, N, N_obs, n_obs,  step_horizon, obs_horizon, rob_diam, obs_diam, prediction_model, pred_trajs, mpc_iter, traj_ids):
        

        self.X = X
        self.P = P
        self.n_states = n_states
        self.N = N
        self.N_obs = N_obs
        self.n_obs = n_obs
        self.step_horizon  = step_horizon
        self.obs_horizon = obs_horizon
        self.rob_diam = rob_diam
        self.obs_diam = obs_diam
        self.prediction_model = prediction_model
        self.pred_trajs = pred_trajs
        self.mpc_iter = mpc_iter
        self.traj_ids = traj_ids


    def hard_constraint(self, d_safe = 0.2, scale = 1e-3):
        g = []
        print(self.mpc_iter)
        for n in range(self.n_obs):
            for k in range(self.N_obs+1):

                
                if self.prediction_model == 'deterministic' or self.prediction_model == 'reactive':
                    factor = 0
                elif self.prediction_model == 'stochastic':
                    factor = 1

                lk = self.compute_lk(k, n) 
                lam_1 = self.P[2 * self.n_states + 4 * n * (self.N_obs + 1) + 4 * k + 2]
                

                x_diff = (self.X[0,k]-self.P[2*self.n_states+4*n*(self.N_obs+1)+4*k])
                y_diff = (self.X[1,k]-self.P[2*self.n_states+4*n*(self.N_obs+1)+4*k+1])
                distance = ca.sqrt((ca.vertcat(x_diff, y_diff).T  @ ca.vertcat(x_diff, y_diff))) 
                # distance_scaled = sigmoid_scaling(distance, scale = 1.0)


                if self.mpc_iter < self.pred_trajs[self.traj_ids[n]]['input'].shape[0]:
                    g = ca.vertcat(g,
                                - distance + \
                                (self.rob_diam/2 + self.obs_diam/2 + factor * lam_1 + d_safe))
                    
                else:
                    g = ca.vertcat(g,
                                - distance)
                
        return g

        
    def Control_Barrier(self, gamma_1=0.5, gamma_2 = 0.1, d_safe = 0.1, scale = 1e3,  order = 'second'):        

        g = []
        for n in range(self.n_obs):
            for k in range(self.N_obs+1):

                if self.prediction_model == 'deterministic' or self.prediction_model == 'reactive':
                    factor = 0
                elif self.prediction_model == 'stochastic':
                    factor = 1

                l_k = self.compute_lk(k, n) 
                lam_1 = self.P[2 * self.n_states + 4 * n * (self.N_obs + 1) + 4 * k + 2]        
                h =  ((self.X[0,k]-self.P[2*self.n_states+4*n*(self.N_obs+1)+4*k]) **2 + (self.X[1,k]-self.P[2*self.n_states+4*n*(self.N_obs+1)+4*k+1]) **2) 
                - (self.rob_diam/2 + self.obs_diam/2 + factor * lam_1 + d_safe)**2  # h(x) : Control Barrier  Function 

                if order == 'first' and k <self.N: 

                    l_k1 = self.compute_lk(k+1, n) 
                    lam_k1 = self.P[2 * self.n_states + 4 * n * (self.N_obs + 1) + 4 * k + 2]    
                    h_1 =  ((self.X[0,k+1]-self.P[2*self.n_states+4*n*(self.N_obs+1)+4*k])**2 + (self.X[1,k+1]-self.P[2*self.n_states+4*n*(self.N_obs+1)+4*k+1])**2) 
                    - (self.rob_diam/2 + self.obs_diam/2 + factor*lam_k1 + d_safe)**2
                    
                    g = ca.vertcat(g,
                                -h_1 + (1- gamma_1)*h)
                    
                
                # if order == 'second' and k < self.N-1: # second order CBF
                #     l_k1 = self.compute_lk(k+1, n) 
                #     lam_k1 = self.P[2 * self.n_states + 4 * n * (self.N_obs + 1) + 4 * k + 2]  
                #     h_1 =  ((self.X[0,k+1]-self.P[2*self.n_states+4*n*(self.N_obs+1)+4*k])**2 + (self.X[1,k+1]-self.P[2*self.n_states+4*n*(self.N_obs+1)+4*k+1])**2) 
                #     - (self.rob_diam/2 + self.obs_diam/2 + factor*lam_k1 + d_safe)**2
                    
                #     l_k2 = self.compute_lk(k+2, n) 
                #     lam_k2 = self.P[2 * self.n_states + 4 * n * (self.N_obs + 1) + 4 * k + 2]  
                #     h_2 = ((self.X[0,k+2]-self.P[2*self.n_states+4*n*(self.N_obs+1)+4*k])**2 + (self.X[1,k+2]-self.P[2*self.n_states+4*n*(self.N_obs+1)+4*k+1])**2) 
                #     - (self.rob_diam/2 + self.obs_diam/2 + factor*lam_k2 + d_safe)**2


                # g = ca.vertcat(g,
                #                 -(h_1 - h + gamma_1*h),
                #                 -((h_2 - 2*h_1 + h) + (gamma_1 + gamma_2) * (h_1 - h) + gamma_1*gamma_2*h))
                                        
        return g
                
    
    def chance_constraint(self, sigma=0.1, scale = 1e3):
        
        g = []
        for n in range(self.n_obs):  
            for k in range(self.N_obs+1):


                nugget = 1e-6
                diff = ca.vertcat(self.X[0,k], self.X[1,k]) - ca.vertcat(self.P[2*self.n_states+4*n*(self.N_obs+1)+4*k],self.P[2*self.n_states+4*n*(self.N_obs+1)+4*k+1])
                kappa_nk = diff/(ca.norm_2(diff) + nugget)

                # Eigen Values to Covariance:
                lam_1 = self.P[2 * self.n_states + 4 * n * (self.N_obs + 1) + 4 * k + 2]
                lam_2 = self.P[2 * self.n_states + 4 * n * (self.N_obs + 1) + 4 * k + 3]
                cov = ca.SX(2, 2)
                cov[0, 0] = 1
                cov[0, 1] = 0
                cov[1, 0] = 0
                cov[1, 1] = 1
                gamma_nk = ca.sqrt(2*kappa_nk.T @ cov  @ kappa_nk) * ca.erfinv( 1- 2*sigma)

                if self.mpc_iter < self.pred_trajs[self.traj_ids[n]]['input'].shape[0]:
                    g = ca.vertcat(g,
                                -kappa_nk.T @ diff + (self.rob_diam/2 + self.obs_diam/2) + gamma_nk)
                else:
                    g = ca.vertcat(g,
                                   -kappa_nk.T @ diff)
                                                        
        return g
    
    
    def affine_chance_constraint(self, sigma =0.1, scale =1e3):

        g = []
        for n in range(self.n_obs):  
            for k in range(self.N_obs+1):


                nugget = ca.DM(1e-6)

                # Approximated Minkowski Sum (ellipsoid)
                a =  self.rob_diam/2 + self.obs_diam/2
                b =  self.rob_diam/2 + self.obs_diam/2

                # Construct chance constraint:
                omega_root = ca.SX(2,2)
                omega_root[0, 0] = 1/a
                omega_root[0, 1] = 0
                omega_root[1, 0] = 0
                omega_root[1, 1] = 1/b    

                # Eigen Values to Covariance:
                lam_1 = self.P[2 * self.n_states + 4 * n * (self.N_obs + 1) + 4 * k + 2]
                lam_2 = self.P[2 * self.n_states + 4 * n * (self.N_obs + 1) + 4 * k + 3]
                cov = ca.SX(2, 2)
                cov[0, 0] = lam_1**2
                cov[0, 1] = 0
                cov[1, 0] = 0
                cov[1, 1] = lam_2**2

                diff = ca.vertcat(self.X[0,k], self.X[1,k]) - ca.vertcat(self.P[2*self.n_states+4*n*(self.N_obs+1)+4*k],self.P[2*self.n_states+4*n*(self.N_obs+1)+4*k+1])

                pos_io = omega_root @ diff
                a_io = diff/ (ca.norm_2(diff) + nugget)
                cov_io = omega_root.T @ cov @ omega_root
                pos_io_norm = ca.sqrt(pos_io.T @ pos_io)
                c_io = ca.sqrt(2*a_io.T @ cov_io  @ a_io) *  ca.erfinv( 1- 2*sigma)

                b_io = self.rob_diam/2 + self.obs_diam/2


                g = ca.vertcat(g,
                            -pos_io_norm + b_io + c_io)


        
        return g


    
    def compute_lk(self, n_step, n_obs):  # Minimum distance from ellipse center to ellipse periphery passing through robot center

        nugget = 1e-4
        lam_1 =  self.P[2 * self.n_states + 4 * n_obs * (self.N_obs + 1) + 4 * (n_step) + 2] # Major Axis
        lam_2 =  self.P[2 * self.n_states + 4 * n_obs * (self.N_obs + 1) + 4 * (n_step) + 3] # Minor Axis
        x_diff = self.X[0,n_step]-self.P[2*self.n_states+4*n_obs*(self.N_obs+1)+4*(n_step)]
        y_diff = self.X[1,n_step]-self.P[2*self.n_states+4*n_obs*(self.N_obs+1)+4*(n_step)+1]


        m = y_diff/(x_diff + nugget)
        l_k = ca.sqrt((lam_1**2 * lam_2**2 *(1 + m**2))/(lam_2**2 + lam_1**2 * m**2))

        return l_k

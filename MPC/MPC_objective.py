import casadi as ca
from casadi import sin, cos, pi
from utils import *
from constraints import *



class MPC_objective:
    def __init__(self, n_states, n_controls, N, N_obs, n_obs, step_horizon, obs_horizon, rob_diam, obs_diam, lambda_, d_safe, scale, mpc_iter, 
                 pred_trajs, traj_ids, constraint = 'HC', prediction_model = 'deteministic', CBF_order = 'first'):
    
        self.n_states = n_states
        self.n_controls = n_controls
        self.N = N
        self.N_obs = N_obs
        self.n_obs = n_obs
        self.step_horizon  = step_horizon
        self.obs_horizon = obs_horizon
        self.rob_diam = rob_diam
        self.obs_diam = obs_diam
        self.lambda_ = lambda_
        self.d_safe = d_safe
        self.scale = scale
        self.mpc_iter = mpc_iter
        self.pred_trajs = pred_trajs
        self.traj_ids = traj_ids
        self.constraint = constraint
        self.prediction_model = prediction_model
        self.CBF_order = CBF_order


    def MPC_cost(self,  X, U,  P, st_lb, st_ub, con_lb, con_ub, f,  Q, Q_N, R, S, start, goal):
       
        # Define the constraints:
        lbx = ca.DM.zeros((self.n_states*(self.N+1) + self.n_controls*self.N, 1))
        ubx = ca.DM.zeros((self.n_states*(self.N+1) + self.n_controls*self.N, 1))


        for i in range(self.n_states):
            lbx[i: self.n_states*(self.N+1): self.n_states] = st_lb[i]    # X lower bound
            ubx[i: self.n_states*(self.N+1): self.n_states] = st_ub[i]

        for i in range(self.n_controls):
            lbx[self.n_states*(self.N+1)+i:self.n_states*(self.N+1)+self.n_controls*self.N:self.n_controls] = con_lb[i]         
            ubx[self.n_states*(self.N+1)+i:self.n_states*(self.N+1)+self.n_controls*self.N:self.n_controls] = con_ub[i]      


        # Define the cost function:
        cost_fn = 0
        g = []
        g = ca.vertcat(g, X[:, 0] - P[:self.n_states])

        # Define Total Collision cost:
        coll_cost_total = 0

        for k in range(self.N):
            st = X[:, k]
            con = U[:, k]

            if k < self.N-1:
                con_next = U[:,k+1]


            # Slack Cost (If HC fails):
            # Collision Cost (Sigmoid Potential):
            for n in range(self.n_obs):
                coll_cost = dynamic_collision_cost( X, P, k, n, self.n_states, self.N_obs, self.obs_horizon, self.step_horizon, self.obs_diam, self.rob_diam, self.lambda_)
                # coll_cost_total += coll_cost


            lenToGoal = (goal-start).T @ (goal-start) 
            
            # Total Cost
            cost_fn = cost_fn \
                + (((st - P[self.n_states:2*self.n_states])).T @ Q @ (st - P[self.n_states:2*self.n_states]))/lenToGoal \
                + (con/con_ub).T @ R @ (con/con_ub) \
                + (con_next/con_ub - con/con_ub).T @ S @ (con_next/con_ub- con/con_ub)  
                
            
            st_new = X[:, k+1]

            # Compute the next step using control with RK4:
            k1 = f(st, con)
            k2 = f(st + self.step_horizon/2*k1, con)
            k3 = f(st + self.step_horizon/2*k2, con)
            k4 = f(st + self.step_horizon*k3, con)

            new_state_RK4 = st + (self.step_horizon/6) * (k1 + 2*k2 + 2*k3 + k4)

            g = ca.vertcat(g, 
                        st_new - new_state_RK4)


        # Terminal Cost:
        cost_fn = cost_fn \
            + (((X[:, self.N] - P[self.n_states:2*self.n_states])).T @ Q_N @ (X[:, self.N] - P[self.n_states:2*self.n_states]))/lenToGoal \
            # + coll_cost_total
        

        robot_con = robot_constraint(X, P, self.n_states, self.N, self.N_obs, self.n_obs, self.step_horizon, 
                                        self.obs_horizon, self.rob_diam, self.obs_diam, self.prediction_model,self.pred_trajs, self.mpc_iter, self.traj_ids)
        
        
        if self.constraint == 'HC':
            g_con = robot_con.hard_constraint(d_safe = self.d_safe, scale = self.scale) 
              


        elif self.constraint == 'CBF':
            order = self.CBF_order
            g_con = robot_con.Control_Barrier( gamma_1 = 0.2, gamma_2 = 0.3,  d_safe = self.d_safe, scale = self.scale, order = order)
            print(g_con.shape) 
           

        elif self.constraint =='chance':
            g_con = robot_con.chance_constraint(sigma = 0.3, scale = self.scale)

        
        elif self.constraint =='chance_affine':
            g_con = robot_con.affine_chance_constraint(sigma = 0.1, scale = self.scale)

        
        elif self.constraint == None:
            g_con, lbg_con, ubg_con = [], [], []
        
        g = ca.vertcat(g,
                       g_con)
    
        
        lbg_con = -ca.DM.inf(g_con.shape)
        ubg_con = ca.DM.zeros(g_con.shape)

        lbg = ca.vertcat(ca.DM.zeros((self.n_states*(self.N+1), 1)),
                         lbg_con)
        ubg = ca.vertcat(ca.DM.zeros((self.n_states*(self.N+1), 1)),
                         ubg_con)
        
        
        return [cost_fn, g, lbx, ubx, lbg, ubg]

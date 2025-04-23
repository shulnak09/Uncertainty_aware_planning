import casadi as ca
import numpy as np
from casadi import sin, cos, pi
import matplotlib.pyplot as plt
import sys
import random
from pathlib import Path


from utils import *
from dynamics import *
from Trajectory_Predictor import *
from Draw_chance_constrained import*
from animate_trajectory import *
from MPC_objective import *
from constraints import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Probabilistic Trajectory Prediction using Deep Ensembles:
look_back = 8
forward_pred= 12
skip = 1
# traj_ids = [2182, 2193, 4074, 5221, 1196, 3135, 2117, 3117, 2401, 5303, 4024.0, 3001.0, 1107.0, 5204.0] # Ensure that do not use the variable name as  "key"  here : Test keys: 2193, 3135, 2401, [2182, 2193, 4074, 5221, 1196]
traj_ids = [2182, 2193, 4074, 5221, 1196, 3135]

prediction = Trajectory_predictor(look_back, forward_pred, skip)
pred_trajs = prediction.predictor(traj_ids = traj_ids)

with open(os.path.join(PATH,'weights.pkl'), 'rb') as f:
    weights = pkl.load(f)

shapes = []
for key, value in pred_trajs.items():
    shapes.append(value['input'].shape[0])
id_list = list(range(max(shapes)))  # Add more IDs as needed


# Evaluate the predcitive uncertainty using ensembles:
cov = {}
for id, value in pred_trajs.items():
    mu_ens = prediction.weighted_sum(weights,  pred_trajs[id]['mean'])
    var_aleatoric = prediction.weighted_sum(weights,  pred_trajs[id]['cov'])
    var_epistemic = prediction.weighted_sum(weights, pred_trajs[id]['mean'][:, :, :, :2] ** 2) - mu_ens[:, :, :2] ** 2
    covar = var_aleatoric + var_epistemic
    fill_array  = np.full((pred_trajs[id]['cov'].shape[1],1,2), 0.05) # Adding noise to last observed state 
    cov[id] = (np.concatenate((fill_array, covar), axis =1))



# Define the initial and target State:
state_init = ca.DM([-7.5, 5.0, 0.0, 0 ,0, 0, 0])
state_target = ca.DM([5.0, -5.0, 0, 0, 0,0,0]) 

# Define the dynamics of the robot:
robot =  RobotDynamics(state_init, state_target)
[f, n_states, n_controls, st_lb, st_ub, con_lb, con_ub, rob_diam, Q, Q_N, R, S] = robot.dynamic_car()


##  Run the MPC main loop
# Define the Parameters:
step_horizon = 0.4  # time between steps (s)
obs_horizon =  0.4  # Pedestrian  Prediction horizon (s) 
N = 8             # number of look ahead steps


# Define the obstacles:
n_obs = len(pred_trajs)
N_obs =  int(N * (step_horizon/obs_horizon))  
obs_diam = 0.6
obs_x = ca.SX.sym('obs_x',n_obs,1)
obs_y = ca.SX.sym('obs_y',n_obs,1)
lambda_ = ca.SX.sym('lambda', 1)
lambda_ = 1.0
d_safe = 0.5


# Define the major and minor ellipse of Prediction uncertainty
rot = ca.SX.sym('rot', 2, 2)

# Define the  states
X = ca.SX.sym('X', n_states, N+1)

# Define all control actions over all time steps
U = ca.SX.sym('U', n_controls, N)

# Define the parameters for start, goal and obstacles
P = ca.SX.sym('P', n_states + n_states + (N_obs+1)*4*n_obs)


# Define the MPC objective: 'HC' : Hard Constraint, 'CBF': Control Barrier Function, 'chance': chance Constraint:


prediction_model = 'deterministic' # Types: 'deterministic', 'stochastic', 'reactive'
CBF_order = 'first'  # Order : 'first', 'second'
filepath = Path("MPC/Results")
subfolder = f"HC_dynamic_car_{prediction_model}"
filepath =  filepath/subfolder
savefile = f'{subfolder}.gif'
filepath.mkdir(parents=True, exist_ok=True)

obj = MPC_objective(n_states, n_controls, N, N_obs, n_obs, step_horizon, obs_horizon, rob_diam, obs_diam, lambda_, d_safe, constraint  = 'HC', 
                    prediction_model= prediction_model, CBF_order=CBF_order)
[cost_fn, g, lbx, ubx, lbg, ubg] = obj.MPC_cost(X, U,  P, st_lb, st_ub, con_lb, con_ub, f, state_init, state_target, Q, Q_N, R, S)


OPT_variables = ca.vertcat(
    X.reshape((-1, 1)),
    U.reshape((-1, 1))
)

nlp_prob = {
    'f': cost_fn,
    'x': OPT_variables,
    'g': g,
    'p': P
}

opts = {
    'ipopt': {
        'max_iter': 100,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}


solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    
args = {
    'lbg':lbg, # constraints lower bound
    'ubg':ubg,
    'lbx':lbx,
    'ubx':ubx
}

#  Define the initial and final state
t0 = 0
t = ca.DM(t0)

u0 = ca.DM.zeros((n_controls, N))  # initial control
X0 = ca.repmat(state_init, 1, N+1) # initial state full
obs0 = ca.DM.zeros((n_obs,4)) # initial state of obstacle with zeros
obsN = ca.repmat(obs0,1,N_obs+1)


mpc_iter = 0
cat_robot_states = DM2Arr(X0[:,0])
cat_states = DM2Arr(X0)
cat_controls = DM2Arr(u0[:, 0].T)

# Initialize Obstacle states:
for n in range(n_obs):
    for k in range(N_obs+1):

        if k == 0 or prediction_model == 'reactive':
            obs_x = pred_trajs[traj_ids[n]]['input'][0,-1,0] + pred_trajs[traj_ids[n]]['initial_pos'][0,0]
            obs_y = pred_trajs[traj_ids[n]]['input'][0,-1,1] + pred_trajs[traj_ids[n]]['initial_pos'][0,1] 
        else:
            obs_x = prediction.weighted_sum(weights,  pred_trajs[traj_ids[n]]['mean'])[0,k-1,0] + pred_trajs[traj_ids[n]]['initial_pos'][0,0]
            obs_y = prediction.weighted_sum(weights,  pred_trajs[traj_ids[n]]['mean'])[0,k-1,1] + pred_trajs[traj_ids[n]]['initial_pos'][0,1]

        obsN[n, 4*k] = obs_x
        obsN[n, 4*k+1] = obs_y


        # Assumed covariance for zeroth state:
        sigma = np.diag(np.squeeze(cov[traj_ids[n]][0,k,:2]))
        lam_1, lam_2, rot = ellipse(sigma)

        obsN[n, 4*k+2] = lam_1
        obsN[n, 4*k+3] = lam_2
    

cat_obstacles = DM2Arr(obsN)

times = np.array([[0]])
tol = 6e-1
sim_time = 30       # simulation time


cost ={
    'translational_error':[],
    'collision_cost':[],
    'control_effort_cost':[],
    'change_control_effort_cost':[],
    'total_cost':[]
}

### Run the Main MPC loop ###

if __name__ == '__main__':
    main_loop = time.time()  # return time in sec
    while (ca.norm_2(state_init[:2] - state_target[:2]) > tol) and (mpc_iter * step_horizon < sim_time):
        t1 = time.time()
        
        # Define the arguements:
        args['p'] = ca.vertcat(
            state_init,     # current state
            state_target,   # target state
            ca.reshape(obsN, -1, 1)
        )


        # optimization variable current state
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(N+1), 1),
            ca.reshape(u0, n_controls*N, 1)
        )

        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )


        u = ca.reshape(sol['x'][n_states * (N + 1):],  n_controls, N)
        X0 = ca.reshape(sol['x'][:n_states * (N + 1)], n_states,   N + 1)

        # Compute The cost Functions (For Plotting):

        translational_cost  = 0
        collision_cost = 0
        control_cost = 0
        change_effort = 0
        total_cost = 0


     
        for k in range(N):
            translational_cost += (X0[:,k] - state_target[:n_states]).T @ Q @ (X0[:,k] - state_target[:n_states]) 
            control_cost += (u[:,k]).T @ R @ (u[:,k]) 
            
            if k < N-1:
                change_effort += ((u[:,k+1]- u[:,k])).T @ S @ ((u[:,k+1]- u[:,k]))
                
            for n in range(n_obs):
                    coll_cost = dynamic_collision_cost(X0, args['p'],  k, n, n_states, N_obs, obs_horizon, step_horizon, obs_diam, rob_diam, lambda_)
                    collision_cost += ca.DM(coll_cost).full()
        
        translational_cost += (X0[:,N] - state_target[:n_states]).T @ Q_N @ (X0[:,N] - state_target[:n_states])
        total_cost = translational_cost + control_cost + change_effort + collision_cost

        cost['translational_error'].append(translational_cost)
        cost['control_effort_cost'].append(control_cost)
        cost['change_control_effort_cost'].append(change_effort)
        cost['collision_cost'].append(collision_cost)
        cost['total_cost'].append(total_cost)


        cat_states = np.dstack((
            cat_states,
            DM2Arr(X0)
        ))

        cat_robot_states = np.hstack((
            cat_robot_states,
            DM2Arr(X0[:,0])
        ))


        cat_controls = np.vstack((
            cat_controls, (
            DM2Arr(u[:, 0].T)
        )))


        t = np.vstack((
            t,
            t0
        ))


        t0, state_init, u0 = shift_timestep(step_horizon, t0, state_init, u, f)
        # vx = state_init[3]
        # vy = state_init[4]
        # v = np.sqrt(vx**2 +vy**2)
        # steer = state_init[6]


        # print(X0)
        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(X0[:, -1], -1, 1)
        )



        # xx ...
        t2 = time.time()
        # print(mpc_iter)
        # print(t2-t1)
        print(ca.norm_2(state_init[:2] - state_target[:2]))
        times = np.vstack((
            times,
            t2-t1   
        ))

        mpc_iter = mpc_iter + 1


        # Update the position of the pedestrian:
        for n in range(n_obs):
            for k in range(N_obs+1):
 

                if mpc_iter < pred_trajs[traj_ids[n]]['input'].shape[0]:
                    factor = int(obs_horizon/step_horizon)
                    if k == 0 or prediction_model == 'reactive':
                        obs_x = pred_trajs[traj_ids[n]]['input'][int(mpc_iter),-1,0] + pred_trajs[traj_ids[n]]['initial_pos'][int(mpc_iter),0]
                        obs_y = pred_trajs[traj_ids[n]]['input'][int(mpc_iter),-1,1] + pred_trajs[traj_ids[n]]['initial_pos'][int(mpc_iter),1] 
                    else:
                        obs_x = prediction.weighted_sum(weights, pred_trajs[traj_ids[n]]['mean'])[int(mpc_iter),k-1,0] + pred_trajs[traj_ids[n]]['initial_pos'][int(mpc_iter),0]
                        obs_y = prediction.weighted_sum(weights, pred_trajs[traj_ids[n]]['mean'])[int(mpc_iter),k-1,1] + pred_trajs[traj_ids[n]]['initial_pos'][int(mpc_iter),1]
                    
                    obsN[n, 4*k] = obs_x
                    obsN[n, 4*k+1] = obs_y
                    
                    # Assumed covariance for zeroth state:
                    sigma = np.diag(np.squeeze(cov[traj_ids[n]][mpc_iter,k,:2]))
                    lam_1, lam_2, rot = ellipse(sigma)

                    obsN[n, 4*k+2] = lam_1
                    obsN[n, 4*k+3] = lam_2
                
                else:

                    obsN[n, 4*k] = 1e-3
                    obsN[n, 4*k+1] = 1e-3

                    obsN[n, 4*k+2] = 0.02
                    obsN[n, 4*k+3] = 0.02




        cat_obstacles = np.dstack((cat_obstacles,
                                  obsN   
        ))   


    cat_obstacles = np.array(cat_obstacles)
    cat_obstacles = np.reshape(cat_obstacles,(n_obs, N_obs+1, 4,  mpc_iter+1))

    main_loop_time = time.time()
    ss_error = ca.norm_2(state_init[:2] - state_target[:2])

    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('final error: ', ss_error)

    print("Cat_robot_states shape:", cat_robot_states.shape)
    print("Cat_states shape:", cat_states.shape)
    print("Cat_obstacles shape:", cat_obstacles.shape)


# Plot Trajectory Length
x_y_coordinates = cat_robot_states[:2, :]  # Extract (x, y) coordinates
differences = np.diff(x_y_coordinates, axis=1)
trajectory_length = np.sum(np.linalg.norm(differences, axis=0))
print("Length of the trajectory:", trajectory_length)
np.set_printoptions(threshold=sys.maxsize)


# Plot the results:
animation_robot =   robot_navigation(pred_trajs, cat_obstacles, weights, id_list, savefile, filepath,  forward_pred = N, prediction_model = prediction_model)
animation_robot.cost_plots(cost, step_horizon)
animation_robot.plot_graphs(t, cat_states, cat_controls)
animation_robot.animate_robot(t, cat_robot_states, cat_states, cat_controls, state_target, N, rob_diam,  cat_obstacles, N_obs, obs_diam,  tol)


       
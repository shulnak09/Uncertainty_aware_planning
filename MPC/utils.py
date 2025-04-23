import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import pandas as pd
import torch
import casadi as ca



def dynamic_collision_cost( X, P, n_step, n, n_states, N_obs, obs_horizon, step_horizon, obs_diam, rob_diam, lambda_):

    # Define the Minkowsky Ellipse:
    eta = ca.SX(2,2)
    eta[0,0] = 1. / (P[2 * n_states + 4 * n * (N_obs + 1) + 4 * n_step + 2] + obs_diam + rob_diam)**2
    eta[0,1] = 0.0
    eta[1,0] = 0.0
    eta[1,1] = 1. / (P[2 * n_states + 4 * n * (N_obs + 1) + 4 * n_step + 3] + obs_diam + rob_diam)**2

    x_diff = X[0,int(obs_horizon/step_horizon)*n_step]-P[2*n_states+4*n*(N_obs+1)+4*n_step]
    y_diff = X[1,int(obs_horizon/step_horizon)*n_step]-P[2*n_states+4*n*(N_obs+1)+4*n_step+1]
    d_buffer = 1.2

    d = ca.sqrt(ca.vertcat(x_diff, y_diff).T   @ eta  @ ca.vertcat(x_diff, y_diff))
    coll_cost = 1./(1 + lambda_ * ca.exp(d - d_buffer))

    return coll_cost



def shift_timestep(step_horizon, t0, state_init, u, f):

    # Adding noise to the control
    con_conv = np.array([[0.05], [np.deg2rad(2)]])**2
    con = u[:,0] + np.sqrt(con_conv)*np.random.randn(2,1)

    
    k1 = f(state_init, con)
    k2 = f(state_init + 0.5 * step_horizon*k1, con)
    k3 = f(state_init + 0.5 * step_horizon*k2, con)
    k4 = f(state_init + step_horizon * k3, con)
    next_state = state_init + 1/6 * step_horizon * (k1 + 2*k2 + 2*k3 + k4)

    next_state = ca.DM.full(next_state)
    t0 = t0 + step_horizon

    u0 = ca.horzcat(
            u[:, 1:],
            ca.reshape(u[:, -1], -1, 1)
        )

    return t0, next_state, u0



def DM2Arr(dm):
    return np.array(dm.full())



def ellipse(cov):
    a = cov[0,0]
    b = cov[0,1]
    c = cov[1,1]

    eig_val, eig_vec = np.linalg.eig(cov)
    
    idx = eig_val.argsort()
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]

    # print("eig_vec : {}".format(eig_vec))

    # Define the ellipse parameters:
    lam_1 = np.sqrt(eig_val[0]) # semi-major axes
    lam_2 = np.sqrt(eig_val[1]) # semi-minor axes

    # Find the rotation matrix:
    R = eig_vec

    return lam_1, lam_2, R



def plot_covariance_ellipse(mean, cov, ax=None, n_std=3.0, facecolor='none', **kwargs):
    """
    Plots an ellipse representing the covariance matrix.

    Parameters:
    mean : array-like, shape (2,)
        The mean (center) of the ellipse.
    cov : array-like, shape (2, 2)
        The covariance matrix.
    ax : matplotlib.axes.Axes, optional
        The axes object to draw the ellipse into. If not provided, a new figure and axes will be created.
    n_std : float, optional
        The number of standard deviations to determine the ellipse's radii. Default is 3.0.
    facecolor : str, optional
        The facecolor of the ellipse. Default is 'none' (transparent).

    **kwargs : additional keyword arguments
        These are passed to the Ellipse patch constructor.
    """
    from matplotlib.patches import Ellipse

    if ax is None:
        fig, ax = plt.subplots()

    # Calculate the eigenvalues and eigenvectors
    vals, vecs = np.linalg.eigh(cov)

    # Sort the eigenvalues and corresponding eigenvectors
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Calculate the angle of the ellipse
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Calculate the width and height of the ellipse
    width, height = 2 * n_std * np.sqrt(vals)

    # Create the ellipse patch
    ellipse = Ellipse(mean, width, height, angle=theta, facecolor=facecolor, **kwargs)

    # Add the ellipse to the plot
    ax.add_patch(ellipse)
    ax.set_xlim(mean[0] - 3*width, mean[0] + 3*width)
    ax.set_ylim(mean[1] - 3*height, mean[1] + 3*height)
    ax.set_aspect('equal', 'box')

    return ax


device = None
def create_dataset(df, agents, look_back = 8, forward_pred = 12, skip=2, device = device):
    traj_test_input = {}
    traj_test_output = {}
    initial_pos = {}
    
    for agent_id in agents:
        sub_df = df[df['agent_id'] == agent_id]
        data_x, data_y = [], []
        start_pos = []
        
        for j in range(0, len(sub_df)-look_back-forward_pred+1, skip):
            pos_lb = sub_df.iloc[j:j+look_back][['pos_x','pos_y']] -\
                        sub_df.iloc[j+look_back-1][['pos_x','pos_y']] 
            vel_lb = sub_df.iloc[j:j+look_back][['vel_x','vel_y']]
            data_x.append(pd.concat([pos_lb,vel_lb], axis=1).values)
            


            pos_fp = sub_df.iloc[j+look_back:j+forward_pred+look_back][['pos_x','pos_y']] -\
                        sub_df.iloc[j+look_back-1][['pos_x','pos_y']] 
            vel_fp = sub_df.iloc[j+look_back:j+forward_pred+look_back][['vel_x','vel_y']]
            data_y.append(pd.concat([pos_fp,vel_fp], axis=1).values)
            
            start_pos.append(sub_df.iloc[j+look_back-1][['pos_x','pos_y']])
        
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        start_pos = np.array(start_pos)
        
        traj_test_input[agent_id] = torch.tensor(data_x, dtype=torch.float).to(device)
        traj_test_output[agent_id] = torch.tensor(data_y, dtype=torch.float).to(device)
        initial_pos[agent_id] = torch.tensor(start_pos,dtype=torch.float).to(device)
    return traj_test_input, traj_test_output, initial_pos


def sigmoid_scaling(value, scale = 1.0):
    return (1.0/(1.0 + ca.exp(- scale *(value - 1))))

def Minkowski_sum():
    pass

# from filterpy.stats import plot_covariance_ellipse
from scipy.linalg import inv
import numpy as np

# Implement Kalman Filter for state estimation:
# X[k+1] = A * X[K] + B * u[K] + w


# Define the state transition matrix:
def get_F(dt):  
    F = np.array([[1., 0, dt, 0],
                  [0,  1., 0, dt],
                  [0,  0, 1., 0],
                  [0,  0, 0, 1]])
    return F

def get_Q(dt, var_wx, var_wy):
    Q = np.array([[0.25*dt**4*var_wx, 0, 0.5*dt**3*var_wx,0],
                  [0, 0.25*dt**4*var_wy, 0, 0.5*dt**3*var_wy],
                  [0.5*dt**3*var_wx, 0, dt**2*var_wx,0],
                  [0, 0.5*dt**3*var_wy,0, dt**2*var_wy]])
    return Q

# Define the process covariance matrix P:
sigma_x = 4
sigma_y = 4
sigma_u = 2
sigma_v = 2
P = np.diag([sigma_x**2, sigma_y**2, sigma_u**2, sigma_v**2]) 


def Kalman_filter(
                    X_prev, 
                    X_measured,
                    vx, vx_dot,
                    vy, vy_dot,
                    dt = 0.4,
                    var_wx = 1,
                    var_wy = 1,
                    P = P
                ):
    
    
#     noise_x = np.random.normal(mu, sigma, [X_prev.shape[0]])
#     X_prev_noise = X_train[:,:] + noise_x
#     x = X_prev_noise[0,:].T

    # Define the state transition matrix:
    F = get_F(dt)


    # Define Process noise:
    Q = get_Q(dt, var_wx, var_wy)
    
    # Define the update parameters:
    H = np.identity(X_prev.shape[0])
    
    # Define the measurement covariance:
    R = np.diag([vx**2, vy**2, vx_dot**2, vy_dot**2])
    
    # measurement 
    z = X_measured
    x = X_prev

    # Predict step:
    x = F @ x 
    P = F @ P @ F.T + Q

    # Update step of Kalman filter:
    # S = H*P*H.T + R + eps ; R is the measurement covariance matrix
    S = H @ P @ H.T + R + 1e-5*np.eye(4)
    K = P @ H.T @ inv(S)
    y = z - H @ x

    x += K @ y
    P = P - K @ H @ P
    return x, P
        
    

def sample_distribution(mean, var, N=1):
    x_sample = np.random.multivariate_normal(mean, var, N)
    if N==1:
        x_sample = x_sample[0,:]
    return x_sample


def get_trajectory(
                    X_seq, # 8, 4
                    f,     
                    mu = 0.,
                    sigma = 0.05,
                    # add other inputs for Kalman filter
                    dt = 0.4,
                    var_wx = 1,
                    var_wy = 1
                  ):
    lookback = X_seq.shape[0]
    vx, vx_dot = f*np.mean([X_seq[:,0]]), f*np.mean([X_seq[:,2]])
    vy, vy_dot = f*np.mean([X_seq[:,1]]), f*np.mean([X_seq[:,3]]) 
    
    # Define the state variance, P:
    sigma_x = 4
    sigma_y = 4
    sigma_u = 2
    sigma_v = 2
    P = np.diag([sigma_x**2, sigma_y**2, sigma_u**2, sigma_v**2]) 
    
    trajectories, mus, covs = [], [], []
    for i in range(X_seq.shape[0]):
        X_measured = X_seq[i,:] 
        
        if i==0:
            X_prev = X_seq[0,:]
            # Add a sample noise to the first point
            noise_x = np.random.normal(mu, sigma, [X_prev.shape[0]])
            X_prev = X_prev + noise_x
#             X_prev = X_prev.T
        
        mu, cov = Kalman_filter(X_prev, 
                                X_measured, 
                                vx, 
                                vx_dot, 
                                vy, 
                                vy_dot,
                                #add other inputs
                                dt = 0.4,
                                var_wx = 1,
                                var_wy = 1,
                                P = P
                                )
        P = cov
        mus.append(mu)
        covs.append(cov.diagonal())
        x_new = sample_distribution(mu, cov, N=1)
        trajectories.append(x_new)
        X_prev = x_new
    
   
    return np.array(trajectories), np.array(mus), np.array(covs)

def get_N_trajectories(
            X_seq,
            num_traj = 100,
            f_list = [0.05]
            ):
  
    f_traj, f_mu, f_cov = [], [], []
    for f in f_list:
        num_trajectories, batch_mu, batch_cov = [], [], []
        for i in range(num_traj):
            traj, mus, covs = get_trajectory(X_seq, f)
            num_trajectories.append(traj)
            batch_mu. append(mus)
            batch_cov.append(covs)
        num_trajectories, batch_mu, batch_cov = np.array(num_trajectories), np.array(batch_mu), np.array(batch_cov)
        f_traj.append(num_trajectories)
        f_mu.append(batch_mu)
        f_cov.append(batch_cov)
    
    return np.array(f_traj), np.array(f_mu), np.array(f_cov)
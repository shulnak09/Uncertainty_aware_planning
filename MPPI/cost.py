import numpy as np

def stage_cost(x_t:np.ndarray,  Q:np.ndarray) -> float:
    ' Define the stage cost '
    x_t[2] = (x_t[2] + 2.0 * np.pi) % (2.0 *np.pi) # Restricting yaw angle between [0, 2*pi]
    cost = x_t.T @ Q @ x_t

    return cost
 

def terminal_cost(x_T:np.ndarray, R:np.ndarray) -> float:
    'Define the terminal cost'
    x_T[2] = (x_T[2] + 2.0 * np.pi) % (2.0 *np.pi)
    terminal_cost = x_T.T @ R @ x_T


    return terminal_cost

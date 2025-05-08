import numpy as np

def dynamics(x_t, v_t, dt) -> np.ndarray:
    'calculate the next state of the robot. Simple bicycle model'

    # Obtain the current state and control:
    x, y, yaw, v = x_t # states  
    steer, accel = v_t # control

    # Robot parameters:
    L = 0.131 
    
    # Updated state:
    x_new = x + v * np.cos(yaw) * dt
    y_new = y + v * np.sin(yaw) * dt
    yaw_new = yaw + v/L *np.tan(steer) * dt 
    v_new =  v + accel * dt

    new_state = np.array([x_new, y_new, yaw_new, v_new])

    return new_state




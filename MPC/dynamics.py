import numpy as np
import casadi as ca


# Create Dynamics model for different systems:

class RobotDynamics:
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal

    
    def dynamic_car(self):

        rob_diam = 0.6      # diameter of the robot
        wheel_radius = 1    # wheel radius
        mass = 4.72         # Mass of the car
        Lr = 0.131          # L in J Matrix (half robot x-axis length)
        Lf = 0.189          # L in J Matrix (half robot y-axis length)

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        psi = ca.SX.sym('psi')
        v_x = ca.SX.sym('v_x')
        v_y = ca.SX.sym('v_y')
        r = ca.SX.sym('r')
        delta = ca.SX.sym('delta')

        # Define the Maximum of Force:
        F_max = 4; F_min = -F_max
        d_delta_max = 0.2 * ca.pi; d_delta_min = -d_delta_max



        states = ca.vertcat(x, y, psi, v_x, v_y, r, delta)
        n_states = states.numel()

        # Define the symbolic control inputs:
        F_x = ca.SX.sym('F_x')
        d_delta = ca.SX.sym('d_delta')

        controls = ca.vertcat(F_x, d_delta)
        n_controls = controls.numel()

        rhs =  ca.vertcat(states[3]*ca.cos(states[2]) - states[4]*ca.sin(states[2]),
                  states[3]*ca.sin(states[2]) + states[4]*ca.cos(states[2]),
                  states[5],
                  controls[0]/mass,
                  (states[3]*controls[1] + states[6]*controls[0]/mass)*(Lr/(Lr+Lf)),
                  (states[4]*controls[1] + states[6]*controls[0]/mass)*(1/(Lr+Lf)),
                  controls[1])

        f = ca.Function('f', [states, controls], [rhs])

        # Define the Weight Matrices:
        Q = 1e-4 * ca.diagcat(2e4, 2e4, 1e3, 1e4, 1e4, 1e-5, 1e-5)
        Q_N = 1e-3 * ca.diagcat(2e5, 2e5, 1e4, 1e5, 1e5, 1e-5, 1e-5)
        R = 1e-1*ca.diagcat(1e-1, 1e-1)
        S = 1e-1*ca.diagcat(1e0, 1e0)

        # Lower and Upper bound of states:
        st_lb = [-20, -20, -ca.pi, -0.5, -0.2, -1*ca.pi, -1*ca.pi]
        st_ub = [20, 20, ca.pi, 2, 0.2, 1*ca.pi, 1*ca.pi]

        # Lower and Upper bound for controls:
        con_lb = [F_min, d_delta_min]
        con_ub = [F_max, d_delta_max]

        return f, n_states, n_controls, st_lb, st_ub, con_lb, con_ub, rob_diam, Q, Q_N, R, S


    def quadrotor_3D(self):

        #Bebop dynamics model

        # Size of the drone:
        quad_dim = {
                'a0' : 0.3,
                'b0' : 0.3,
                'c0' : 0.5
        }
         
        # Define constant parameters:
        g           =   9.81
        kD_x        =   0.25
        kD_y        =   0.33
        k_vz        =   1.2270
        tau_vz      =   0.3367
        k_phi       =   1.1260
        tau_phi     =   0.2368
        k_theta     =   1.1075
        tau_theta   =   0.2318
    

        # Define the states:
        # state:   x, y, z, vx, vy, vz, phi, theta, psi
        # control: phi_c, theta_c, vz_c, psi_rate_c

        state_syms = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi']
        x, y, z, vx, vy, vz, phi, theta, psi = [ca.SX.sym(s) for s in state_syms]
        states = ca.vertcat(x, y, z, vx, vy, vz, phi, theta, psi)
        n_states = states.numel()


        # Define the controls:

        con_syms = ['phi_c', 'theta_c', 'vz_c', 'psi_rate_c']
        phi_c, theta_c, vz_c, psi_rate_c = [ca.SX.sym(s) for s in con_syms]        
        controls = ca.vertcat(phi_c, theta_c, vz_c, psi_rate_c)
        n_controls = controls.numel()


        ax = ca.tan(states[7]) * g - kD_x * states[3] 
        ay = -ca.tan(states[6]) * g - kD_y * states[4]
        az = (k_vz*controls[3] - states[5])/tau_vz

        ## Attitude dynamics:
        dphi    = (k_phi*controls[0] - states[8]) / tau_phi
        dtheta  = (k_theta*controls[1] - states[7]) / tau_theta
        dpsi    = controls[3]


        rhs = ca.vertcat(states[3], states[4], states[5], ax, ay, az, dphi, dtheta, dpsi)

        f = ca.Function('f', [states, controls], [rhs])

        # Lower and Upper bound of states:
        st_lb = [-ca.inf, -ca.inf, -ca.inf, -2.0, -2.0, -1.0, -ca.pi, -ca.pi, -ca.pi ]
        st_ub = [ ca.inf,  ca.inf,  ca.inf,  2.0,  2.0,  1.0,  ca.pi,  ca.pi,  ca.pi]

        # Lower and Upper bound for controls:
        con_lb = [-12*ca.pi/180, -12*ca.pi/180, -1.0, -90*ca.pi/180 ]
        con_ub = [ 12*ca.pi/180,  12*ca.pi/180,  1.0,  90*ca.pi/180]


        return f, n_states, n_controls, st_lb, st_ub, con_lb, con_ub, quad_dim
    
    def quadrotor_2D(self):
        ' 2D Quadrotor dynamics '
        # Define constant parameters:
        g           =   9.81
        m           =   1
        I_xx        =   0.01
        L           =   0.15


        kp_y        =   0.25
        kp_z        =   0.7
        kv_y        =   0.25
        kv_z        =   0.7
        kp_phi      =   18
        kv_phi      =   15

        quad_dim = 2*L

        # Define the states:
        # state:   y, z, phi, vy, vz, phidot
        # control: phi_c, theta_c, vz_c, psi_rate_c

        state_syms = ['y', 'z', 'phi', 'vy', 'vz', 'phi_dot']
        y, z,  vy, vz, phi, phi_dot = [ca.SX.sym(s) for s in state_syms]
        states = ca.vertcat(y, z,  vy, vz, phi, phi_dot )
        n_states = states.numel()


        # Define the controls:

        con_syms = ['u1', 'u2']
        u1, u2 = [ca.SX.sym(s) for s in con_syms]        
        controls = ca.vertcat(u1, u2)
        n_controls = controls.numel()

        y_ddot = -u1 * ca.sin(phi) / m
        z_ddot = -g + u1 * ca.cos(phi) / m
        phi_ddot = u2 / I_xx

        rhs = ca.vertcat(states[3],
                         states[4],
                         states[5],
                         y_ddot,
                         z_ddot,
                         phi_ddot)
        
        f = ca.Function('f', [states, controls], [rhs])

        Q = 1e-5 * ca.diagcat(2e4, 2e4, 1e4, 1e4, 1e4, 1e4)
        Q_N = 1e-5 * ca.diagcat(2e5, 2e5, 5e4, 1e5, 1e5, 1e4)
        R = 1e-1*ca.diagcat(1e-1, 1e-1)
        S = 1e-1*ca.diagcat(1e0, 1e1)

        # Lower and Upper bound of states:
        st_lb = [-20, -20, -ca.pi, -1.0, -1.0,  -ca.pi]
        st_ub = [ 20,  20,  ca.pi,  1.0,  1.0,   ca.pi]

        # Lower and Upper bound for controls:
        con_lb = [ 0,  0]
        con_ub = [ 1.7658,  1.7658]

        return f, n_states, n_controls, st_lb, st_ub, con_lb, con_ub, quad_dim, Q, Q_N, R, S


        

    def dubin_car(self):

        rob_diam = 0.6      # diameter of the robot
        Lr = 0.131          # L in J Matrix (half robot x-axis length)
        Lf = 0.189          # L in J Matrix (half robot y-axis length)


        # Define the States of the car:
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')


        states = ca.vertcat(x, y, theta)
        n_states = states.numel()

        # Define the symbolic control inputs:
        phi_max = 0.1 * ca.pi; phi_min =  -phi_max
        v_max = 2.0; v_min = -v_max 
    
        v = ca.SX.sym('v')
        phi = ca.SX.sym('phi')

        controls = ca.vertcat(v, phi)
        n_controls = controls.numel()

        # Define the system dynamics:
        rhs = ca.vertcat(controls[0] * ca.cos(states[2]),
                         controls[0] * ca.sin(states[2]),
                         controls[0]/(Lf + Lr) * ca.tan(controls[1])
        )

        f = ca.Function('f', [states, controls], [rhs])

        # Define the weight Matrices:
        Q = 1e-5 * ca.diagcat(2e4, 2e4, 1e4)
        Q_N = 1e-5 * ca.diagcat(4e5, 4e5, 5e4)
        R = 1e-1*ca.diagcat(1e-1, 1e-1)
        S = 1e-1*ca.diagcat(1e0, 1e0)

        # Lower and Upper bound of states:
        st_lb = [-20, -20, -2*ca.pi]
        st_ub = [ 20,  20,  2*ca.pi]

        # Lower and Upper bound for controls:
        con_lb = [v_min, phi_min]
        con_ub = [v_max, phi_max]

        return f, n_states, n_controls, st_lb, st_ub, con_lb, con_ub, rob_diam, Q, Q_N, R, S


    def differential_drive(self):

        wheel_rad = 1
        rob_diam =  0.3  

        # Define the states of differential Drive:
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')


        states = ca.vertcat(x, y, theta)
        n_states = states.numel()

        # Define the symbolic control inputs:
        wl_max = 1 * ca.pi;  wl_min =  -wl_max
        wr_max = 1 * ca.pi;  wr_min =  -wr_max
    
        w_l = ca.SX.sym('w_l')
        w_r = ca.SX.sym('w_r')

        controls = ca.vertcat(w_l, w_r)
        n_controls = controls.numel()

        
        # Define the system dynamics:
        rhs = ca.vertcat((wheel_rad/2)*(controls[0] + controls [1]) * ca.cos(states[2]),
                         (wheel_rad/2)*(controls[0] + controls [1]) * ca.sin(states[2]),
                         (wheel_rad/rob_diam) * (controls[0] -  controls[1])
        )

        f = ca.Function('f', [states, controls], [rhs])


        # Define the weight Matrices:
        Q = 1e-5 * ca.diagcat(2e4, 2e4, 1e4)
        Q_N = 1e-5 * ca.diagcat(4e5, 4e5, 5e4)
        R = 1e-4*ca.diagcat(1e-2, 1e-2)
        S = 1e-4*ca.diagcat(1e-1, 1e-1)

        # Lower and Upper bound of states:
        st_lb = [-20, -20, -2*ca.pi]
        st_ub = [ 20,  20,  2*ca.pi]

        # Lower and Upper bound for controls:
        con_lb = [wl_min, wl_max]
        con_ub = [wr_min, wr_max]


        return f, n_states, n_controls, st_lb, st_ub, con_lb, con_ub, rob_diam, Q, Q_N, R, S




        
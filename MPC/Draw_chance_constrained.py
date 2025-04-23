import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle, Ellipse, Circle
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


class robot_navigation:
    def __init__(self, pred_trajs, cat_obstacles,  weights, id_list, savefile, filepath,  forward_pred =12, prediction_model = 'stochastic'):
        self.pred_trajs = pred_trajs
        self.cat_obstacles = cat_obstacles
        self.forward_pred = forward_pred
        self.id_list = id_list
        self.weights = weights
        self.savefile = savefile
        self.filepath = filepath
        self.prediction_model = prediction_model



    # Define the animation function
    def animate_pedestrian(self, ax, id_no):
        # ax.scatter(self.X_test[id_no, -1, 0] + self.initial_pos[id_no,0], self.X_test[id_no, -1, 1] + self.initial_pos[id_no,1], color='g', marker='o', s=5, label='Input')
        # circle = Circle((self.X_test[id_no, -1, 0] + self.initial_pos[id_no,0], self.X_test[id_no, -1, 1] + self.initial_pos[id_no,1]), radius=self.obs_diam/2, color='b', fill=False)
        # ax.add_patch(circle)

        obs_id = 0
        colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'yellow', 'pink','violet']

        for _ , single_ped in self.pred_trajs.items():
            self.X_test = single_ped['input']
            self.y_test = single_ped['output']
            self.mus = single_ped['mean']
            self.sigmas = single_ped['cov']
            self.initial_pos = single_ped['initial_pos']

            if id_no < single_ped['input'].shape[0] and self.cat_obstacles.size > 0:
                # self.cat_obstacles = np.squeeze(self.cat_obstacles)
                color = colors[obs_id % len(colors)]
                ax.scatter(self.cat_obstacles[obs_id,0,0,id_no], self.cat_obstacles[obs_id,0,1,id_no], color = color, marker ='o', s=5)
                circle = Circle((self.cat_obstacles[obs_id,0,0,id_no], self.cat_obstacles[obs_id,0,1,id_no]), radius=self.obs_diam/2, color=color, fill=True)
                ax.add_patch(circle)

                if self.prediction_model == 'deterministic' or self.prediction_model == 'reactive':
                    for i in range(self.forward_pred):
                        circle = Circle((self.cat_obstacles[obs_id,i+1,0,id_no], self.cat_obstacles[obs_id,i+1,1,id_no]), radius=self.obs_diam/2, color='b', linestyle='--', fill=False)
                        ax.add_patch(circle)

                ax.plot(self.cat_obstacles[obs_id,1:, 0, id_no], self.cat_obstacles[obs_id,1:, 1, id_no], color='b', marker='d', alpha=0.85, ms=3)

                if self.prediction_model == 'deterministic' or self.prediction_model == 'stochastic':
                    ax.plot(self.y_test[id_no, :self.forward_pred, 0] + self.initial_pos[id_no,0], self.y_test[id_no, :self.forward_pred, 1] + self.initial_pos[id_no,1], color='r', alpha=0.5, marker='^', ms=3)
                

                mu_ens = self.weighted_sum(self.weights, self.mus)
                # ax.plot(mu_ens[id_no, :self.forward_pred, 0] + self.initial_pos[id_no,0], mu_ens[id_no, :self.forward_pred, 1] + self.initial_pos[id_no,1], color='b', marker='d', alpha=0.85, ms=3, label='NN state Estimate')

                var_aleatoric = self.weighted_sum(self.weights, self.sigmas[:, :, :, :2])
                var_epistemic = self.weighted_sum(self.weights, self.mus[:, :, :, :2] ** 2) - mu_ens[:, :, :2] ** 2
                var_ens = var_aleatoric + var_epistemic
                var_state_unc = self.weighted_sum(self.weights, self.mus[:, :, :, 2:4])
        
                
                
                for pred in range(self.forward_pred):
                    cov_state = np.squeeze(var_state_unc[id_no, pred, :2])
                    cov_state = np.diag(np.squeeze(cov_state))
                    

                    if self.cat_obstacles.size >0 :
                        mean = np.squeeze(self.cat_obstacles[obs_id, pred+1, :2, id_no])
                        lambda_tot = np.squeeze(self.cat_obstacles[obs_id, pred+1, 2:, id_no])
                    else:
                        mean = np.squeeze(mu_ens[id_no, pred, :2]  + self.initial_pos[id_no,:])
                        cov_pred = np.diag(var_ens[id_no, pred, :2])
                        lambda_tot, v_tot = np.linalg.eig(cov_pred)
                        lambda_tot = np.sqrt(lambda_tot)

                    lambda_ale, v_ale = np.linalg.eig(cov_state)
                    lambda_ale = np.sqrt(lambda_ale)

                    if self.prediction_model == 'stochastic':
                        for j in range(1, 2):
                            ell1 = Ellipse(xy=(mean[0], mean[1]),
                                        width=(self.obs_diam/2 + lambda_tot[0]) * j * 2,
                                        height=(self.obs_diam/2 + lambda_tot[1]) * j * 2,
                                        angle=np.rad2deg(np.arccos((v_ale[0, 0]))),
                                        color='none', lw=0.5)
                            ell1.set_facecolor('tab:blue')
                            ell1.set_alpha(0.4 / j)
                            ax.add_artist(ell1)

                            ell2 = Ellipse(xy=(mean[0], mean[1]),
                                        width=(lambda_tot[0]) * j * 2,
                                        height=(lambda_tot[1]) * j * 2,
                                        angle=np.rad2deg(np.arccos((v_ale[0, 0]))),
                                        color='none', linestyle='--', lw=0.25)
                            ell2.set_facecolor('tab:olive')
                            ell2.set_alpha(0.4 / j)
                            ax.add_artist(ell2)
                
            obs_id += 1

    @staticmethod
    def weighted_sum(weights, array):
        avg_ens = np.tensordot(weights, array, axes=([0], [0]))
        return avg_ens
    


    def animate_robot(self, t, cat_robot_states, cat_states, cat_controls, xs, N, rob_diam,  cat_obstacles, N_obs,  obs_diam,   tol):
        plt.rc('font', family='DejaVu Sans')
        plt.rc('font', size=12)
        plt.rcParams['animation.ffmpeg_path'] = "C:\ffmpeg\bin" 

        self.obs_diam = obs_diam

        line_width = 2.5
        fontsize_labels = 14

        x_r_1 = []
        y_r_1 = []


        # n_obstacles = len(obs_x)

        r = rob_diam / 2
        ang = np.arange(0, 2 * np.pi, 0.01)
        xp = r * np.cos(ang)
        yp = r * np.sin(ang)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-0.0, 20.0)
        ax.set_ylim(-0.0, 20.0)
        ax.set_aspect('equal')

        cat_states = cat_states.transpose(1,0,2) 
        trajectory_length = 0
        xy_coord = cat_robot_states[:2, :]
        trajectory_length = np.array(
            [0] + [np.round(np. linalg.norm(xy_coord[:,k] - xy_coord[:,k-1]), 2) for k in range(1, xy_coord.shape[1])]).cumsum()
                


        def update(k):
            ax.clear()
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
            h_t, w_t = 0.14, 0.09

            # Goal Location:
            x1, y1, th1 = xs[0], xs[1], xs[2]
            x1_tri = np.squeeze(np.array([x1 + h_t * np.cos(th1), x1 + (w_t / 2) * np.cos(np.pi / 2 - th1),
                    x1 - (w_t / 2) * np.cos(np.pi / 2 - th1)]))
            y1_tri = np.squeeze(np.array([y1 + h_t * np.sin(th1), y1 - (w_t / 2) * np.sin(np.pi / 2 - th1),
                    y1 + (w_t / 2) * np.sin(np.pi / 2 - th1)]))
            ax.fill(x1_tri, y1_tri, 'g')
            ax.plot(x1 + tol * np.cos(ang), y1 + tol * np.sin(ang), 'k--', linewidth=2, label = 'Goal')


            x1, y1, th1 = cat_robot_states[0, k], cat_robot_states[1, k], cat_robot_states[2, k]
            x_r_1.append(x1)
            y_r_1.append(y1)
            x1_tri = [x1 + h_t * np.cos(th1), x1 + (w_t / 2) * np.cos(np.pi / 2 - th1),
                    x1 - (w_t / 2) * np.cos(np.pi / 2 - th1)]
            y1_tri = [y1 + h_t * np.sin(th1), y1 - (w_t / 2) * np.sin(np.pi / 2 - th1),
                    y1 + (w_t / 2) * np.sin(np.pi / 2 - th1)]
            

            
            
            ax.plot(x_r_1, y_r_1, '-r', linewidth=line_width)
            ax.text(0.05, 0.95, f'Trajectory length :{trajectory_length[k]:.2f}[m]', fontsize = 10, bbox=dict(facecolor='white', alpha=0.5), transform=ax.transAxes)

            if k < cat_robot_states.shape[1]:
                ax.plot(cat_states[:N, 0, k], cat_states[:N, 1, k], 'r-', label = 'Robot Trajectory')

                # Plot the obstacle
                # for n in range(cat_obstacles.shape[0]):
                #     ax.plot(cat_obstacles[n,:N_obs+1,0,k],cat_obstacles[n,:N_obs+1,1,k],'b--*')
                    # ax.plot(cat_obstacles[n,0,k],cat_obstacles[n,1,k],'b--*')
        

                for j in range(1, N + 1):
                    r = (obs_diam / 2) 
                    xp_obs = r * np.cos(ang)
                    yp_obs = r * np.sin(ang)
                    ax.plot(cat_states[j - 1, 0, k] + xp, cat_states[j - 1, 1, k] + yp, '--r', linewidth=line_width * 0.25)
                    # ax.plot(obs_cl[j - 1, 0, k] + xp_obs, obs_cl[j - 1, 1, k] + yp_obs, '--b', linewidth=line_width * 0.5)

                    # for n in range(cat_obstacles.shape[0]):
                    #     ax.plot(cat_obstacles[n,int(j/4),0,k] + xp_obs, cat_obstacles[n,int(j/4),1,k]+yp_obs,'b',  linewidth=line_width)
                    #     ax.plot(cat_obstacles[n,int(j/4),0,k] + xp_obs, cat_obstacles[n,int(j/4),1,k]+yp_obs,'b',  linewidth=line_width*0.25)

            if (k < self.id_list[-1]):
                self.animate_pedestrian(ax, self.id_list[int(k)])



            ax.fill(x1_tri, y1_tri, 'r')
            ax.plot(x1 + xp, y1 + yp, 'r', linewidth=line_width, label = 'robot')

            ax.set_title(f'Time : {np.round(k * 0.4,1)}[s]')
            ax.set_ylabel('y[m]', fontsize=fontsize_labels)
            ax.set_xlabel('x[m]', fontsize=fontsize_labels)
            ax.grid(False)
            ax.legend(loc = 'upper right')

        ani = FuncAnimation(fig, update, frames=len(cat_states[0, 0, :]), repeat=False)
        # writer = FFMpegWriter(fps=10)  # Specify frames per second (fps)
        save_path = Path(self.filepath)/self.savefile
        ani.save(save_path, writer = PillowWriter(fps=3), dpi=300)  # Specify the filename and writer
        plt.show()

    
    @staticmethod
    def cost_plots(cost, step_horizon):
    # Plot the Cost function:
        for key in cost:
            cost[key] = np.squeeze(cost[key])

        fig = plt.figure(figsize=(6,6))
        ax = plt.gca()

        times = np.arange(len(cost['total_cost']))*step_horizon

        for key, value in cost.items():
            plt.plot(times, value, label = key, linewidth = 2.)

        ax.set_yscale('log')

        plt.xlabel('Time (s)')
        plt.ylabel('Cost Value')
        plt.title('Contribution of Cost Factors Over Time')
        plt.legend(prop={'size': 15})
        ax.legend(bbox_to_anchor=(1.1, 1.05))
        plt.grid(True)
        plt.show()
        plt.pause(3)
        plt.close()



    def plot_graphs(self, t, cat_states, cat_controls):
        fig1,axs = plt.subplots(nrows = cat_states.shape[0], ncols =1, figsize = (4,7))
        states = ['x[m]', 'y[m]', r'$\theta[rad]$', r'$v_{x}[m/s]$', r'$v_{y}[m/s]$', r'$\psi$[rad]', r'$\delta[rad]$']
        for i in range(cat_states.shape[0]):
            axs[i].plot(t, np.squeeze(cat_states[i, 0, :]), linewidth=2.5, label = states[i])
            axs[i].set_ylabel(states[i], fontsize=12)
            axs[i].grid(True)
        
        axs[-1].set_xlabel('Time(s)', fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_path = Path(self.filepath)/'states.jpg'
        plt.savefig(save_path, dpi=300) 
    

        fig2, axs = plt.subplots(nrows=2, ncols=1, figsize =(4,2))
        controls = ['F', r'$\dot{\delta}$']
        axs[0].plot(t, cat_controls[:, 0])
        axs[1].plot(t, cat_controls[:, 1])
        axs[0].grid(True)
        axs[1].grid(True)
        axs[-1].set_xlabel('Time(s)', fontsize=12)
        axs[0].set_ylabel(controls[0], fontsize=12)
        axs[1].set_ylabel(controls[1], fontsize=12)
    
        plt.tight_layout(rect=[0, 0, 1, 0.95])  
        
        save_path = Path(self.filepath)/'controls.jpg'
        plt.savefig(save_path, dpi=300) 
        plt.show()
        
    
# You can call the function with your data like this:
# draw_mpc_apf_obstacle_avoidance(t, xx, xx1, u_cl, xs, N, rob_diam, obs_x, obs_y, obs_cl, obs_diam, mass, rx, ry, tol, factor)



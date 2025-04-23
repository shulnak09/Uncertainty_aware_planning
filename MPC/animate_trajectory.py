import matplotlib.animation as animation
from matplotlib import pyplot as plt
import numpy as np

import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse



class animate_trajectory:
    def __init__(self, pred_trajs, id_list, weights, forward_pred,  savefile):
        self.pred_trajs = pred_trajs
        self.forward_pred = forward_pred
        self.savefile = savefile
        self.id_list = id_list
        self.weights = weights
        self.fig, self.ax = plt.subplots(figsize=(6, 6))

    # Define the initialization function to clear the plot
    def init_plot(self):
        self.ax.clear()
        self.ax.set_xlim([-20, 20])
        self.ax.set_ylim([-20, 20])
        self.ax.set_aspect('equal', adjustable='box')
        return self.ax,
    
    # Define the animation function
    def animate(self, id_no):
        self.ax.clear()
        self.ax.set_xlim([-20, 20])
        self.ax.set_ylim([-20, 20])
        self.ax.set_aspect('equal', adjustable='box')

        for _, single_ped in self.pred_trajs.items():
            self.X_test = single_ped['input']
            self.y_test = single_ped['output']
            self.mus = single_ped['mean']
            self.sigmas = single_ped['cov']
            self.initial_pos = single_ped['initial_pos']

            if id_no < single_ped['input'].shape[0]:
        
                self.ax.scatter(self.X_test[id_no, :, 0] + self.initial_pos[id_no,0], self.X_test[id_no, :, 1] + self.initial_pos[id_no,1], color='g', marker='o', s=5, label='Input')
                self.ax.plot(self.y_test[id_no, :, 0] + self.initial_pos[id_no,0], self.y_test[id_no, :, 1] + self.initial_pos[id_no,1], color='r', alpha=0.5, marker='^', ms=3, label='Ground Truth')
                
                mu_ens = self.weighted_sum(self.weights, self.mus)
                self.ax.plot(mu_ens[id_no, :, 0] + self.initial_pos[id_no,0], mu_ens[id_no, :, 1] + self.initial_pos[id_no,1], color='b', marker='d', alpha=0.85, ms=3, label='NN state Estimate')

                var_aleatoric = self.weighted_sum(self.weights, self.sigmas[:, :, :, :2])
                var_epistemic = self.weighted_sum(self.weights, self.mus[:, :, :, :2] ** 2) - mu_ens[:, :, :2] ** 2
                var_ens = var_aleatoric + var_epistemic
                var_state_unc = self.weighted_sum(self.weights, self.mus[:, :, :, 2:4])

                
                for pred in range(self.forward_pred):
                    mean = np.squeeze(mu_ens[id_no, pred, :2]  + self.initial_pos[id_no,:])
                    cov_pred = np.squeeze(np.squeeze(np.diag(var_ens[id_no, pred, :])))
                    cov_state = np.squeeze(var_state_unc[id_no, pred, :2])
                    cov_state = np.diag(np.squeeze(cov_state))
                    

                    lambda_tot, v_tot = np.linalg.eig(cov_pred)
                    lambda_tot = np.sqrt(lambda_tot)
                    lambda_ale, v_ale = np.linalg.eig(cov_state)
                    lambda_ale = np.sqrt(lambda_ale)
                    
                    for j in range(1, 2):
                        ell1 = Ellipse(xy=(mean[0], mean[1]),
                                    width=(1 * lambda_ale[0]) * j * 2,
                                    height=(1 * lambda_ale[1]) * j * 2,
                                    angle=np.rad2deg(np.arccos((v_ale[0, 0]))),
                                    color='none', lw=0.5)
                        ell1.set_facecolor('tab:blue')
                        ell1.set_alpha(0.2 / j)
                        self.ax.add_artist(ell1)

                        ell2 = Ellipse(xy=(mean[0], mean[1]),
                                    width=(lambda_tot[0]) * j * 2,
                                    height=(lambda_tot[1]) * j * 2,
                                    angle=np.rad2deg(np.arccos((v_ale[0, 0]))),
                                    color='none', linestyle='--', lw=0.25)
                        ell2.set_facecolor('tab:olive')
                        ell2.set_alpha(0.4 / j)
                        self.ax.add_artist(ell2)

        # ax.legend()
        return self.ax,

    def animate_trajectories(self, id_list):
        self.id_list = id_list
        ani = animation.FuncAnimation(self.fig, self.animate, frames=self.id_list, init_func= self.init_plot, blit=False, repeat=False)
        anim_writer = animation.PillowWriter(fps=1)
        ani.save(self.savefile, writer=anim_writer)
        plt.show()

    
    def weighted_sum(self, weights, array):
#     mu_ens = np.zeors_like(mu_preds[0])
        avg_ens = np.tensordot(weights, array, axes=([0], [0]))
        return avg_ens
        


# Example usage


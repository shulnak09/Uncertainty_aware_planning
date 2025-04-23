
import os
import torch.nn as nn
import torch
import pickle as pkl
import pandas as pd
import time
from collections import defaultdict
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Load the modules:
from Deep_Ensemble.normalize import Normalization
from Deep_Ensemble.model import *
from utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pdb



PATH = r'/home/anshul/Research/Pred_unc_aware_planning/Deep_Ensemble/Trained_models'
with open(os.path.join(PATH, 'weights.pkl'), 'rb') as f:
    weights = pkl.load(f)

'''
Why test_df.pkl and not test_data.pkl is used?
'''

with open (os.path.join(PATH, 'test_df.pkl'),'rb') as f:
    [test_df, test_agents] = pkl.load(f)


# Load the NN hyperparameters  for Inference:
with open(os.path.join(PATH, "model_hyperparams.pkl"),'rb') as f:
    params = pkl.load(f)


input_size = params['input_size']
hidden_size = params['hidden_size']
output_size = params['output_size']
num_layers = params['num_layers']
dropout = params['dropout']
num_fea = params['input_size']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trajectory_predictor:
    def __init__(self, look_back = 8, forward_pred = 12, skip=1):
        self.look_back = look_back
        self.forward_pred =  forward_pred
        self.skip = skip


    def predictor(self, traj_ids = None):

        # Obtain the Input and output Trajectory:
        input_traj, output_traj, starting_pos = create_dataset(test_df, test_agents, look_back = self.look_back, 
                                                            forward_pred = self.forward_pred, skip=self.skip, device = device)
        

        
        deep_ensemble = EnsembleModel(input_size = num_fea, hidden_size = hidden_size, output_size=output_size, num_layers=num_layers,
                                    dropout=dropout, use_attention = False, device = device, num_models=3, init_weight = xavier_initialize)
        ensemble_predictions = defaultdict(list)


        start_time = time.time()
        for idx, model in enumerate(deep_ensemble.models):
            model_path = f"model_{idx}_parameters.pth"
            model.load_state_dict(torch.load(os.path.join(PATH, model_path)), strict=True)
            model.eval()

            # Temporarily storing predictions for this model
            model_predictions = {}
            for key, value in input_traj.items():
                with torch.no_grad():
                    if value.shape[0] == 0:
                        model_predictions[key] = np.array([])
                    else:
                        prediction = model.predict(value, target_len=self.forward_pred, device=device)
                        model_predictions[key] = prediction

            # Append each key's prediction to the ensemble_predictions
            for key, preds in model_predictions.items():
                ensemble_predictions[key].append(preds)


        # Transform  ensemble_predictions to numpy arrays: [3, n_traj, 12, 6]
        mu_preds = {}
        sigma_preds = {}
        for key in ensemble_predictions:
            min_logvar, max_logvar = -4, 4
            ensemble_predictions[key] = torch.stack(ensemble_predictions[key], axis=0)
            preds = torch.cat((ensemble_predictions[key][:,:,:,:int(num_fea/2)], torch.exp(ensemble_predictions[key][:,:,:,int(num_fea/2):int(num_fea)])),3)
            sigmas = ensemble_predictions[key][:,:,:,int(num_fea):]
            sigmas = torch.clamp(sigmas, min=min_logvar, max=max_logvar)
            mu_preds[key] = torch.Tensor(preds)
            # mu_preds = denormalize_data(mu_preds, train_mean,  train_std)
            sigma_preds[key] = torch.Tensor(torch.exp(sigmas))

        # Computation Time:
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time} seconds")

        # Optionally, convert the defaultdict back to a regular dict if needed
        ensemble_predictions = dict(ensemble_predictions)


        pred_trajs = {}
        for id in traj_ids:
            single_ped = {}
            single_ped['input'] = input_traj[id].detach().cpu().numpy()   # shape (N_traj, 8, 4) : key {3135: (14,8,4)}
            single_ped['output'] = output_traj[id].detach().cpu().numpy()  
            single_ped['mean'] = mu_preds[id].detach().cpu().numpy()  
            single_ped['cov'] = sigma_preds[id].detach().cpu().numpy()  
            single_ped['initial_pos'] = starting_pos[id].detach().cpu().numpy()  

            pred_trajs[id] = single_ped


        return pred_trajs
    
    
    def weighted_sum(self, weights, array):
        avg_ens = np.tensordot(weights, array, axes=([0], [0]))
        return avg_ens



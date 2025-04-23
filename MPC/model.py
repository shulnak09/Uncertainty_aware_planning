import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import trange
import random
import math
from torch.utils.tensorboard import SummaryWriter


class lstm_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, p = 0.2, weight_init = None):
        super(lstm_encoder, self).__init__()
        
        """
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden state in the LSTM.
        num_layers (int): The number of LSTM layers stacked vertically. The output of Layer1 is 
        input to Layer 2 alongwith the hidden and cell unit. 
        Link: https://rb.gy/one06f     
        """
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, dropout = p,  batch_first = False)
        
    
    def forward(self, x):
        lstm_out, (self.hidden, self.cell) = (self.lstm(x.view(x.shape[0], x.shape[1], self.input_size)))
        return lstm_out, (self.hidden, self.cell)
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                  torch.zeros(self.num_layers, batch_size, self.hidden_size))
        
class lstm_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers =2, p =0.2):
        super(lstm_decoder,self).__init__()
        
        
        self.input_size =  input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, 
                            num_layers = num_layers, batch_first = False)
        
        self.linear = nn.Linear(hidden_size,  output_size)
#         self.linear = nn.Linear(hidden_size, input_size)
            
    
    def forward(self, x, hidden, cell):
        
        # Shape of x is (N, features), we want (1,N, features)
        lstm_out, (self.hidden, self.cell) = self.lstm(x.unsqueeze(0), (hidden, cell))
        output = self.linear(lstm_out.squeeze(0))
        
        return output, self.hidden, self.cell
    

'''
Decoder Atttention Class 
'''
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(2*hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias= False)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.shape[0]
        h = hidden[-1].repeat(timestep,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1)
        attn_energies = self.score(h, encoder_outputs)  
        weights = F.softmax(attn_energies, dim=1)
        return weights

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        energy = self.v(energy).squeeze(2)
        return energy

    
class DecoderAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, p=0.2):
        super(DecoderAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers  = num_layers
        self.dropout = nn.Dropout(p)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(input_size + hidden_size, hidden_size, num_layers, batch_first =  False)
        self.linear = nn.Linear(hidden_size*2, output_size)
    
    def forward(self, x, hidden, cell, encoder_outputs):
        x = x.unsqueeze(0)
        attn_weights = self.attention(hidden, encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        lstm_input = torch.cat([x, context.transpose(0,1)], 2)
        lstm_out, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = self.linear(torch.cat([lstm_out.squeeze(0), context.squeeze(1)], 1))
        return output, hidden, cell, attn_weights

    
    
    
class lstm_seq2seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers,  dropout, use_attention,  device):
        
        super(lstm_seq2seq, self).__init__()
        self.input_size =  input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.use_attention = use_attention
        
        self.encoder = lstm_encoder(input_size = self.input_size, hidden_size = hidden_size, num_layers=num_layers, p = dropout).to(device)
        
        if use_attention:
            self.decoder = DecoderAttention(input_size = self.input_size, hidden_size = hidden_size, output_size=output_size, num_layers=num_layers,  p = dropout).to(device)
        else:
            self.decoder = lstm_decoder(input_size = self.input_size, hidden_size = hidden_size, output_size=output_size, num_layers=num_layers,  p = dropout).to(device)    
        
    def train_model( self, device, train_loader, val_loader, n_epochs, target_len, beta =0.5, 
                    training_prediction ='teacher_forcing', teacher_forcing_ratio = 0.7, learning_rate = 0.001, dynamic_tf = False,
                   optimizer_name = 'Adam',  writer = None):
        
        

        
        # Select optimizer
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        elif optimizer_name == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), lr = learning_rate)


        ''' 
        input_tensor = input_data with shape (Batch, seq_length, input_size =4 )
        target_tensor = target_data with shape(Batch, target_length, output_size = 4)
        n_epochs = number of epochs
        training_prediction = type of prediction that the NN model has to perform either 'recursive' or 
                        student-teacher-forcing
        dynamic_tf =  dynamic tecaher forcing reduces the amount of teacher force ratio every epoch
        '''
        # Clip the log-variance to avoid exploding/vanishing gradients:
        min_logvar, max_logvar = -4, 4

        # define optimizer
        optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                        factor=0.25, patience=5, threshold=0.001, threshold_mode='abs')

        # Initialize loss for each epoch
        losses = np.full(n_epochs, np.nan)
        val_losses = np.full(n_epochs, np.nan)

        # Number of batches:
        first_batch = next(iter(train_loader))
        num_fea = first_batch[0].shape[2]  # Assuming batch structure is (batch_x, batch_y)

        lrs = [] # Obtain the LR
        
        # Instantiate EarlyStopping
        early_stopping = EarlyStopping(patience=7, min_delta=0.001)  # Adjust these values as needed

        # Teacher Forcing Decay:
        decay = 10.0


        with trange(n_epochs) as tr:
            for it in tr:

                batch_loss = 0
                self.train()
                
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.permute(1, 0, 2).to(device), batch_y.permute(1, 0, 2).to(device) # shape : seq, batch, input

    #                     print(batch_x.shape, batch_y.shape)
                    # Initialize output tensor:
                    outputs = torch.zeros(target_len, batch_y.shape[1], batch_y.shape[2] + 2).to(device)

                    # Initialize the hidden state 
                    encoder_hidden = self.encoder.init_hidden(batch_y.shape[1])

                    # zero the gradient:
                    optimizer.zero_grad()

                    # Encoder outputs for the entire sequence of lookback:
                    encoder_output, (encoder_hidden, encoder_cell) = self.encoder(batch_x)

                    # Decoder outputs:
                    decoder_input = batch_x[-1,:,:] # d(y(t)) is used to predict ^y(t+1)  
                    decoder_hidden = encoder_hidden # encoder cell and hidden output are input to decoder_hidden 
                    decoder_cell = encoder_cell

                    if training_prediction == 'recursive':
                        # Predict recursively:
                        for t in range(target_len):
                            if self.use_attention:
                                decoder_output, decoder_hidden, decoder_cell, _ = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_output)
                            else:
                                decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
                            outputs[t] = decoder_output
                            decoder_input = decoder_output[:,:num_fea]

                    if training_prediction == 'teacher_forcing':
                        if random.random() < teacher_forcing_ratio:
                            for t in range(target_len):    
                                if self.use_attention:
                                    decoder_output, decoder_hidden, decoder_cell, _ = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_output)
                                else:
                                    decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
                                outputs[t] = decoder_output
                                decoder_input = torch.squeeze(batch_y[t,:,:])

                        else:
                            for t in range(target_len):
                                if self.use_attention:
                                    decoder_output, decoder_hidden, decoder_cell, _ = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_output)
                                else:
                                    decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
                                outputs[t] = decoder_output
                                decoder_input = decoder_output[:,:num_fea]

                    # compute the loss:
    #                     outputs = outputs.to(device, dtype=torch.float64)
                    F, B, _ = outputs.shape

                    #predictive UQ
                    outputs_mean, outputs_var = outputs[:,:,:int(num_fea/2)], outputs[:,:,num_fea:] #target_len, b, 8
    #                     print( outputs_mean.shape, outputs_var.shape )
                    outputs_logvar = torch.clamp(outputs_var, min=min_logvar, max=max_logvar)
                    loss_NLL = 0.5*((outputs_mean - batch_y[:,:,:int(num_fea/2)])**2/torch.exp(outputs_logvar))+ 0.5*outputs_logvar
                    if beta > 0:
                        loss = loss_NLL * torch.exp(outputs_logvar).detach() ** beta
                    loss_NLL = torch.mean(loss)

                    #State UQ
                    state_log_covar = (outputs[:,:,int(num_fea/2):num_fea])
#                     state_log_covar = torch.clamp(state_log_covar, min=min_logvar, max=max_logvar)
                    state_covar = torch.exp(state_log_covar)
                    loss_MSE = torch.mean(0.5*(state_covar - batch_y[:,:,int(num_fea/2):num_fea])**2)

                    loss = loss_NLL  + loss_MSE

    #                     (torch.log(var) + ((y - mean).pow(2))/var).sum()
    #                     loss = gaussian_nll(batch_y,outputs)
                    batch_loss += loss.item() # Compute the loss for entire batch 

                    # Backpropagation:
                    loss.backward()
                    optimizer.step()


                # LR scheduler:
                scheduler.step(loss)
    #                 lrs.append(scheduler.get_last_lr())

                # Loss for epoch
                batch_loss /= len(train_loader)
                losses[it]  = batch_loss
                
                writer.add_scalar('training loss',
                                  losses[it],
                                  it
                                  )

                # Dynamic teacher Forcing:
                if dynamic_tf and teacher_forcing_ratio >0:
                    teacher_forcing_ratio = teacher_forcing_ratio * inverse_sigmoid(decay, it)  # Teacher Forcing Decay is linear (Look For inverse sigmoid)
                    
                self.eval()
                with torch.no_grad():
                    
                    val_loss = 0
                    for batch_x, batch_y in val_loader:
                        
                        batch_x, batch_y = batch_x.permute(1, 0, 2).to(device), batch_y.permute(1, 0, 2).to(device) # shape : seq, batch, input

                        # Initialize output tensor:
                        outputs = torch.zeros(target_len, batch_y.shape[1], batch_y.shape[2] + 2).to(device)

                        # Initialize the hidden state 
                        # encoder_hidden = self.encoder.init_hidden(batch_y.shape[1])


                        # Encoder outputs for the entire sequence of lookback:
                        encoder_output, (encoder_hidden, encoder_cell) = self.encoder(batch_x)

                        # Decoder outputs:
                        decoder_input = batch_x[-1,:,:] # d(y(t)) is used to predict ^y(t+1)  
                        decoder_hidden = encoder_hidden # encoder cell and hidden output are input to decoder_hidden 
                        decoder_cell = encoder_cell

                        for t in range(target_len):
                            if self.use_attention:
                                decoder_output, decoder_hidden, decoder_cell, _ = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_output)
                            else:
                                decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
                            outputs[t] = decoder_output
                            decoder_input = decoder_output[:,:num_fea]
                        
                        #predictive UQ
                        outputs_mean, outputs_var = outputs[:,:,:int(num_fea/2)], outputs[:,:,num_fea:] #target_len, b, 8
                        outputs_logvar = torch.clamp(outputs_var, min=min_logvar, max=max_logvar)
                        loss_NLL = 0.5*((outputs_mean - batch_y[:,:,:int(num_fea/2)])**2/torch.exp(outputs_logvar))+ 0.5*outputs_logvar
                        if beta > 0:
                            loss = loss_NLL * torch.exp(outputs_logvar).detach() ** beta
                        loss_NLL = torch.mean(loss)

                        #State UQ
                        state_log_covar = (outputs[:,:,int(num_fea/2):num_fea])
#                         state_log_covar = torch.clamp(state_log_covar, min=min_logvar, max=max_logvar)
                        state_covar = torch.exp(state_log_covar)
                        loss_MSE = torch.mean(0.5*(state_covar - batch_y[:,:,int(num_fea/2):num_fea])**2)

                        loss = loss_NLL  + loss_MSE
                        
                        val_loss += loss.item()
                        
                    val_loss /= len(val_loader)
                    val_losses[it] = val_loss
                    
                    # Log validation loss
                    writer.add_scalar('Validation Loss', val_loss, it)
                    
                    # Call EarlyStopping instance and check for early stop
                    early_stopping(val_loss, self)
                    if early_stopping.early_stop:
                        print("Early stopping triggered")
                        break

                    # Log the running loss averaged per batch:
#                     writer.add_scalars('Training vs validation Loss',
#                                        {'Training':batch_loss, 'validation':val_loss},
#                                        tr*len(train_loader)+it)
                # progress bar
                tr.set_postfix(loss="{0:.3f}".format(batch_loss), val_loss="{0:.3f}".format(val_loss))
            
            writer.flush()
            return losses, val_losses


    def predict(self, input_tensor, target_len, device):

        '''
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor 
        : param target_len:        number of target values to predict 
        : return np_outputs:       np.array containing predicted values; prediction done recursively 
        '''
        input_tensor = input_tensor.permute(1, 0, 2).to(device) #batch_first=False
        # encode input_tensor
        encoder_output, (encoder_hidden,encoder_cell) = self.encoder(input_tensor.to(self.device))

        # initialize tensor for predictions
        outputs = torch.zeros(target_len, input_tensor.shape[1], input_tensor.shape[2] +2, device=self.device) #target_len, B, 4

        # decode input_tensor
        decoder_input = input_tensor[-1, :, :]
    #         decoder_input_var = torch.ones_like(decoder_input)*min_var
    #         decoder_input = torch.cat([decoder_input, decoder_input_var], dim=1)
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        for t in range(target_len):
            if self.use_attention:
                decoder_output, decoder_hidden, decoder_cell, _ = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_output)
            else:
                decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
            outputs[t] = decoder_output
            decoder_input = decoder_output[:,:input_tensor.shape[2]]

        np_outputs = outputs #.detach() #.numpy()

        np_outputs = np_outputs.permute(1, 0, 2) # batch_first=False
        return np_outputs

class EnsembleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, use_attention,  device, num_models = 5, init_weight = None):
        super(EnsembleModel, self).__init__()
        self.device = device
        self.models = [lstm_seq2seq(input_size, hidden_size, output_size, num_layers, dropout, use_attention, device) for _ in range(num_models)]
        
        if init_weight is not None:
            for model in self.models:
                init_weight(model)

    def train_model(self, *args, **kwargs):
        for model in self.models:
            model.train_model(*args, **kwargs)
    
    def predict(self, input_tensor, target_len, device):
        self.to_eval_mode()
        input_tensor = input_tensor.to(device)
        predictions = [model.predict(input_tensor, target_len, device) for model in self.models]
        ensemble_predictions = torch.stack(predictions)
        return ensemble_predictions
    
    def to_eval_mode(self):
        # Use it if necessary for predict method.
        for model in self.models:
            model.eval()
    
    def to_device(self):
        for model in self.models:
            model.to(self.device)
    
    


    
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
#             print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if val_loss < self.val_loss_min:
#             print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
            torch.save(model.state_dict(), 'checkpoint.pt')
            self.val_loss_min = val_loss


def xavier_initialize(model):
    for m in model.modules():
        # print(f"Initializing module: {m}")
        if isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                # print(f"Initializing LSTM param: {name}")
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'bias' in name and param is not None:
                    param.data.fill_(0)
                else:
                    pass
                    # print(f"No bias found for {name} in {m}")
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)
            else:
                pass
                # print(f"No bias found in {m}")

def he_initialize(model):
    for m in model.modules():
        if isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
                elif 'weight_hh' in name:
                    nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
                elif 'bias' in name and param is not None:
                    param.data.fill_(0)

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.fill_(0)
            else:
                print(f"No bias found in {m}")

# Define Inverse Sigmoid For Tecaher Forcing:
def inverse_sigmoid(decay, idx):
    return decay/(decay + math.exp(idx/decay))


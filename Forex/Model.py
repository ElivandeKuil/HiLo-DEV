import torch
from torch.autograd import Variable
import Globals
import numpy as np


class LSTM(torch.nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.ID = None
        self.ticker= None
        self.device = Globals.device
        self.num_classes = 1
        self.num_layers = Globals.number_layers
        self.input_size = Globals.sequence_input_size
        self.hidden_size = Globals.hidden_layer_size
        self.seq_length = Globals.sequence_length
        self.label_mode = Globals.label_mode
        
        self.lstm = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc_1 =  torch.nn.Linear(self.hidden_size, self.num_classes) 

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.flatten = torch.nn.Flatten()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device) 
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        self.lstm.flatten_parameters()
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) 
        hn = hn[-1].view(-1, self.hidden_size) 
        out = self.relu(hn)
        out = self.fc_1(out) 
        out = self.sigmoid(out) 
        return out


class LINEAR(torch.nn.Module):
    def __init__(self, linear_size):   
        super(LINEAR, self).__init__()
        self.ID = None
        self.fc1 = torch.nn.Linear(in_features=linear_size,out_features=1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        output = self.fc1(x)
        output = self.sigmoid(output)
        return output
        
class NTDM_V0(torch.nn.Module):
    
    def __init__(self, dev):
        super(NTDM_V0, self).__init__()
        self.ID = None
        self.ticker = "None"
        self.name = "First gen"
        self.description = "First attempt tackling Forex"
        
        self.linear_encoder = LINEAR(362)
        
        self.sequence_encoder = LSTM()
        
        self.final_fc = torch.nn.Linear(in_features=2,out_features=1)
        
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, linear, sequence):
        
        linear_output = self.linear_encoder(linear)
        sequence_output = self.sequence_encoder(sequence)
        
        catted = torch.cat([linear_output, sequence_output], dim=1)
        
        final_output = self.final_fc(catted)
        
        final_output = self.sigmoid(final_output)
        return final_output
    

class NTDM_V1(torch.nn.Module):
    
    def __init__(self, dev):
        super(NTDM_V1, self).__init__()
        self.ID = None
        self.ticker = "None"
        self.name = "LSTM_only"
        self.description = "100th attempt at Forexx, this time LSTM only"
        
        self.sequence_encoder = LSTM()
        
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, sequence):
        
        sequence_output = self.sequence_encoder(sequence)
        
        return sequence_output

class GRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = torch.nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h =  torch.nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):

        # Inputs:
        #       input: of shape (batch_size, input_size)
        #       hx: of shape (batch_size, hidden_size)
        # Output:
        #       hy: of shape (batch_size, hidden_size)

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size))

        x_t = self.x2h(input)
        h_t = self.h2h(hx)


        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy
    

class GRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, output_size):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = torch.nn.ModuleList()

        self.rnn_cell_list.append(GRUCell(self.input_size,
                                          self.hidden_size,
                                          self.bias))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(GRUCell(self.hidden_size,
                                              self.hidden_size,
                                              self.bias))
        self.fc = torch.nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input, hx=None):

        # Input of shape (batch_size, seqence length, input_size)
        #
        # Output of shape (batch_size, output_size)

        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))

        else:
             h0 = hx

        outs = []
        last_state_list = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(h0[layer, :, :])

        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](input[:, t, :], hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1],hidden[layer])
                hidden[layer] = hidden_l

                hidden[layer] = hidden_l

            outs.append(hidden_l)
            last_state_list.append(hidden_l)

            # Take only last time step. Modify for seq to seq
        
        last_state_list = last_state_list[-1].squeeze()
        out = torch.stack(outs)
        out = self.fc(out).swapaxes(0, 1)
        return out, last_state_list
    
class ST_AutoEncoder(torch.nn.Module):
    
    def __init__(self, bottle_neck_size):
        super(ST_AutoEncoder, self).__init__()
        
        self.name = "GRU AE"
        self.description = "G8 from master thesis"
        self.bottle_neck_size = bottle_neck_size
        
                
        # Temporal Encoder & Decoder
        self.temporal_encoder_decoder = Temporal_EncDec(bottle_neck_size)
        
    def forward(self, x):
       
        output, bottleneck = self.temporal_encoder_decoder(x)
        
        return output, bottleneck
    
    
class Temporal_EncDec(torch.nn.Module):
    def __init__(self, bottle_neck_size):
        super(Temporal_EncDec, self).__init__()
        
        self.gru1 = GRU(input_size=3, hidden_size=16, output_size=3, num_layers=1, bias=True)
        self.gru2 = GRU(input_size=3, hidden_size=16, output_size=3, num_layers=1, bias=True)

        self.flat = torch.nn.Flatten()
        
        self.fc1 = torch.nn.Linear(in_features=800,out_features=bottle_neck_size)
        
        self.fc2 =  torch.nn.Linear(in_features=bottle_neck_size, out_features=800)
        
        self.unflat = torch.nn.Unflatten(1, unflattened_size=(0,0))
        
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        layer_output_list, hidden_state_1 = self.gru1(x)
        
        unflat_shape_h = hidden_state_1.shape
        
        flat_h = hidden_state_1.reshape(-1)
        
        encoded = self.fc1(flat_h)
        
        decoded = torch.tanh(self.fc2(encoded))
        
        self.unflat.unflattened_size = (unflat_shape_h[0], unflat_shape_h[1])
        
        decoded_hidden_state = decoded.reshape(unflat_shape_h)
        
        layer_output_list, hidden_state_2 = self.gru2(layer_output_list, decoded_hidden_state.unsqueeze(dim=0))
        
        return layer_output_list, encoded
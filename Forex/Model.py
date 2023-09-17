import torch
from torch.autograd import Variable
import Globals


class LSTM(torch.nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.ID = None
        self.ticker= None
        self.device = Globals.device
        self.num_classes = 1
        self.num_layers = Globals.number_layers
        self.input_size = 3 
        self.hidden_size = Globals.hidden_layer_size
        self.seq_length = 8
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
        
       # linear_output = torch.reshape(linear_output, (1, 1))
       # torch.reshape(sequence_output, (1, 1))
        
        catted = torch.cat([linear_output, sequence_output], dim=1)
        
        final_output = self.final_fc(catted)
        
        final_output = self.sigmoid(final_output)
        return final_output
    


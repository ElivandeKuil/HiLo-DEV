import numpy as np
from Order import Order
import torch
from datetime import datetime
import Globals
from torch.utils.data import DataLoader

class TickerPredictor():
    
    def __init__(self, ticker, Id, name, spread, labelmode):
        
        self.ticker = ticker
        self.ID = Id
        self.Name = name
        self.Description = ""
        self.Comments = ""
        self.Max_spread = spread # in %, so 0.1 means 0.1 perent NOT 10
        self.Active = True
        self.labelmode = labelmode
        
        self.model = None
        self.doc = None
        
        
        
    def load_models(self, model, doc):
        
        model.sequence_encoder.device = 'cpu'
        model.sequence_encoder.to('cpu')
        model.linear_encoder.to('cpu')
        
        self.doc = doc
        self.model = model.to('cpu')
        
    def format_batch(self, batch): 
        
        linear_data = []
        sequence_data = []
        labels = []
        
        for sample in batch:
            
            linear_data.append(sample.linear_input)
            sequence_data.append(sample.sequence_input)
            
            
        return linear_data, sequence_data
    
    
    def predict(self, time, data):
        
        order = []
        
        loader = DataLoader([data],
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=lambda batch: self.format_batch(batch))


        with torch.no_grad():
                
            model = model.eval()
            model.to('cpu')
            
            iter = iter(loader)
            
            for v in range(0, len(iter)):
                
                batch_linear_data, batch_sequence_data = next(iter)
                
                linear_data = torch.FloatTensor(np.array(batch_linear_data)).to('cpu').to(dtype=torch.float)
                sequence_data = torch.FloatTensor(np.array(batch_sequence_data)).to('cpu').to(dtype=torch.float)
                
                output = model(linear_data, sequence_data)                    
                
                if Globals.sys_log_mode > 2:
                    Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                            model.ID + " => prediction: " + str(output), self.ticker) 
                
                output = output.round()[0]
                
                if output > 0: 
                    if Globals.sys_log_mode > 0:
                        if self.labelmode == '+':
                            label_mode = 'high'
                            Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                                "Buy order expected", self.ticker)   
                        else:
                            label_mode = 'low'
                            Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                                "Sell order expected", self.ticker)
        
                
                    ID = self.ticker + str(time)
                    
                    order = [Order(ID, self, label_mode, self.ticker, time, model.ID)]
        
        if len(order) > 0:
            if Globals.sys_log_mode > 0:
                Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                "An order was created", self.ticker) 
                print("An order was created")
        
        return order
        

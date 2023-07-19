import numpy as np
from Order import Order
import torch
from datetime import datetime
import Globals
from torch.utils.data import DataLoader

#Globals.log_df.export_df("C:/Users/eli_s/Documents/GitHub/Project S V6/SingleClass/analysis docs")

class TickerPredictor():
    
    def __init__(self, ticker, Id, name, spread):
        
        self.ticker = ticker
        self.ID = Id
        self.Name = name
        self.Description = ""
        self.Comments = ""
        self.Max_spread = spread # in %, so 0.1 means 0.1 perent NOT 10
        self.Active = True
        
        self.low_models = []
        self.high_models = []
        
        self.high_test_prec = 0
        self.high_test_ar = 0
        self.low_test_prec = 0
        self.low_test_ar = 0
        
        
        
    def load_models(self, low_models, high_models, low_ind_documentation, low_comb_documentation,
                    high_ind_documentation, high_comb_documentation, high_test_prec, high_test_ar,
                    low_test_prec, low_test_ar):
        
        for model in low_models:
            
            model.sequence_encoder.device = 'cpu'
            model.sequence_encoder.to('cpu')
            model.linear_encoder.to('cpu')
            
            self.low_models.append(model.to('cpu'))
            
        for model in high_models:
            
            model.sequence_encoder.device = 'cpu'
            model.sequence_encoder.to('cpu')
            model.linear_encoder.to('cpu')
            
            self.high_models.append(model.to('cpu'))
        
        if len(low_models) > 0:
        
            self.low_ind_documentation = low_ind_documentation
            self.low_comb_documentation = low_comb_documentation
            self.low_target_prec = low_test_prec
            self.low_test_ar = low_test_ar
        
        if len(high_models) > 0:
            self.high_ind_documentation = high_ind_documentation
            self.high_comb_documentation = high_comb_documentation
            self.high_target_prec = high_test_prec
            self.high_test_ar = high_test_ar
        
    def format_batch(self, batch): 
        
        linear_data = []
        sequence_data = []
        labels = []
        
        for sample in batch:
            
            linear_data.append(sample.linear_input)
            sequence_data.append(sample.sequence_input)
            
            
        return linear_data, sequence_data
    
    
    def predict(self, time, data, debug=False):
        
        low_prediction = 0
        high_prediction = 0
        
        order = []
        predicted_model = "None"
        
        
        test_loader = DataLoader([data],
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=0,
                                         collate_fn=lambda batch: self.format_batch(batch))
        
        for model in self.low_models:
            
            with torch.no_grad():
                
                model = model.eval()
                model.to('cpu')
                
                test_iter = iter(test_loader)
                
                for v in range(0, len(test_iter)):
                    
                    test_batch_linear_data, test_batch_sequence_data = next(test_iter)
                    
                    
                    linear_data = torch.FloatTensor(np.array(test_batch_linear_data)).to('cpu').to(dtype=torch.float)
                    sequence_data = torch.FloatTensor(np.array(test_batch_sequence_data)).to('cpu').to(dtype=torch.float)
                    
                    output = model(linear_data, sequence_data)
                
                    if debug:
                        Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                               "low prediction: " + str(output), self.ticker) 
                        print("low prediction: " + str(output))
                    
                    
                    output = output.round()[0]
                    
                    if debug:
                        Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                               "low prediction: " + str(output), self.ticker) 
                        print("low prediction (rounded): " + str(output))
                    
                    if output > 0: 
                        Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                               "Sell order expected", self.ticker)
                        low_prediction = 1
                        predicted_model = model.ID
        
        for model in self.high_models:
            
            with torch.no_grad():
                
                model = model.eval()
                model.to('cpu')
                
                test_iter = iter(test_loader)
                
                for v in range(0, len(test_iter)):
                    
                    test_batch_linear_data, test_batch_sequence_data = next(test_iter)
                    
                    
                    linear_data = torch.FloatTensor(np.array(test_batch_linear_data)).to('cpu').to(dtype=torch.float)
                    sequence_data = torch.FloatTensor(np.array(test_batch_sequence_data)).to('cpu').to(dtype=torch.float)
                    
                    output = model(linear_data, sequence_data)
                
                    if debug:
                
                        Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                               "high prediction: " + str(output), self.ticker) 
                        print("high prediction: " + str(output))
                        
                    output = output.round()[0]
                    
                    if debug: 
                    
                        Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                               "high prediction: " + str(output), self.ticker) 
                        print("high prediction (rounded): " + str(output))
                    
                    if output > 0: 
                        Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                               "Buy order expected", self.ticker)
                        high_prediction = 1
                        predicted_model = model.ID
        
        if low_prediction > 0 or high_prediction > 0:
            
            if low_prediction > 0 and high_prediction > 0:
                
                print("double")
                Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                       "Orders cancelled due to a double prediction", self.ticker)
            else:
                
                if low_prediction > 0: label_mode = 'low'
                if high_prediction > 0 : label_mode = 'high'
            
                ID = self.ticker + str(time)
                
                Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                       "An order was created", self.ticker) 
                print("An order was created")
                
                order = [Order(ID, self, label_mode, self.ticker, time, predicted_model)]
        
        return order
        
#Globals.log_df.export_df("C:/Users/eli_s/Documents/GitHub/Project S V6/Forex/analysis docs")
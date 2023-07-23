import numpy as np
from Order import Order
import torch
from datetime import datetime
import Globals
from torch.utils.data import DataLoader

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
    
    
    def predict(self, time, data):
        
        low_prediction = 0
        high_prediction = 0
        highest_predict = 0
        lowest_predict = 0
        
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
                    
                    if Globals.sys_log_mode > 2:
                        Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                               model.ID + " => low prediction: " + str(output), self.ticker) 
                    
                    if output > lowest_predict: lowest_predict = output
                    
                    output = output.round()[0]
                    
                    if output > 0: 
                        
                        if Globals.sys_log_mode > 0:
                            Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                                   "Sell order expected", self.ticker)
                        low_prediction += 1
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
                    
                    if output > highest_predict: highest_predict = output
                
                    if Globals.sys_log_mode > 2:
                
                        Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                               model.ID + " => high prediction: " + str(output), self.ticker) 
                        
                    output = output.round()[0]
                    
                    if output > 0: 
                        if Globals.sys_log_mode > 0:
                            Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                                   "Buy order expected", self.ticker)
                        high_prediction += 1
                        predicted_model = model.ID
        
        if low_prediction > 0 or high_prediction > 0:
            
            if low_prediction > 0 and high_prediction > 0:
                
                if Globals.sys_log_mode > 2:
                    Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                       "Double prediction found", self.ticker)
                
                if lowest_predict < 0.55 and highest_predict < 0.55:
                    
                    if Globals.sys_log_mode > 0:
                        Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                           "Tiebreaking was unsuccesful due to lack of conviction from both models, order was cancelled", self.ticker)
                else:   
                    diff = highest_predict - lowest_predict
                    if diff < -0.05: 
                        ID = self.ticker + str(time)
                        order = [Order(ID, self, 'low', self.ticker, time, predicted_model, "double")]
                        
                        if Globals.sys_log_mode > 2:
                            Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                               "Difference in prediction output was > .05 in favour of low (" + str(diff) + ")", self.ticker)
                        
                    else:
                        if diff > 0.05:
                            ID = self.ticker + str(time)
                            order = [Order(ID, self, 'high', self.ticker, time, predicted_model, "double")]
                            
                            if Globals.sys_log_mode > 2:
                                Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                                   "Difference in prediction output was > .05 in favour of high (" + str(diff) + ")", self.ticker)
                            
                        else:
                            
                            if Globals.sys_log_mode > 2:
                                Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                                   "No large difference in prediction output was found", self.ticker)
                            
                            bias = 0
                            low_predict_percentage = low_prediction / len(self.low_models)
                            high_predict_percentage = high_prediction / len(self.high_models)
                            
                            if low_predict_percentage == high_predict_percentage:
                                if Globals.sys_log_mode > 2:
                                    Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                                       "High and Low predict percentage where equal (h=" + str(high_predict_percentage) + " vs. l=" + str(low_predict_percentage) + ")", self.ticker)
                            else:
                                if low_predict_percentage < high_predict_percentage:
                                    bias += 1
                                    
                                    if Globals.sys_log_mode > 2:
                                        Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                                           "High models had higher predict percentage (h=" + str(high_predict_percentage) + " vs. l=" + str(low_predict_percentage) + ")", self.ticker)
                                else:
                                    bias -= 1
                                    
                                    if Globals.sys_log_mode > 2:
                                        Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                                           "low models had higher predict percentage (l=" + str(low_predict_percentage) + " vs. h=" + str(high_predict_percentage) + ")", self.ticker)
                                
                            if lowest_predict < highest_predict:
                                bias += 1
                                
                                if Globals.sys_log_mode > 2:
                                    Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                                       "High models had highest prediction score (h=" + str(highest_predict) + " vs. l=" + str(lowest_predict) + ")", self.ticker)
                            else:
                                bias -= 1
                                
                                if Globals.sys_log_mode > 2:
                                    Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                                       "Low models had highest prediction score (l=" + str(lowest_predict) + " vs. h=" + str(highest_predict) + ")", self.ticker)
                            
                            if self.low_target_prec < self.high_target_prec:
                                bias += 1
                                
                                if Globals.sys_log_mode > 2:
                                    Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                                       "High models had highest target precision (h=" + str(self.high_target_prec) + " vs. l=" + str(self.low_target_prec) + ")", self.ticker)
                            else:
                                bias -= 1
                                
                                if Globals.sys_log_mode > 2:
                                    Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                                       "Low models had highest target precision (l=" + str(self.low_target_prec) + " vs. h=" + str(self.high_target_prec) + ")", self.ticker)
                            
                            if low_prediction == high_prediction:
                                if Globals.sys_log_mode > 2:
                                    Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                                       "High and low models had equal amount of predictions (h=" + str(high_prediction) + " vs. l=" + str(low_prediction) + ")", self.ticker)
                            else:
                                if low_prediction < high_prediction:
                                    bias += 1
                                    
                                    if Globals.sys_log_mode > 2:
                                        Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                                           "High models had most predictions (h=" + str(high_prediction) + " vs. l=" + str(low_prediction) + ")", self.ticker)
                                else:
                                    bias -= 1
                                    
                                    if Globals.sys_log_mode > 2:
                                        Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                                           "Low models had most predictions (l=" + str(low_prediction) + " vs. h=" + str(high_prediction) + ")", self.ticker)
                                    
                            if bias > 0:
                                ID = self.ticker + str(time)
                                order = [Order(ID, self, 'high', self.ticker, time, predicted_model, "double")]
                                
                                if Globals.sys_log_mode > 1:
                                    Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                                       "High prediction won the tiebreaker with a score of " + str(bias), self.ticker)
                                
                            if bias < 0:
                                ID = self.ticker + str(time)
                                order = [Order(ID, self, 'low', self.ticker, time, predicted_model, "double")]
                                
                                if Globals.sys_log_mode > 1:
                                    Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                                       "Low prediction won the tiebreaker with a score of " + str(bias), self.ticker)
                                    
                            if bias == 0:
                                if Globals.sys_log_mode > 0:
                                    Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                                                       "Tiebreaking was unsuccesful, order was cancelled", self.ticker)
            else:
                
                if low_prediction > 0: label_mode = 'low'
                if high_prediction > 0 : label_mode = 'high'
            
                ID = self.ticker + str(time)
                
                order = [Order(ID, self, label_mode, self.ticker, time, predicted_model)]
        
        if len(order) > 0:
            if Globals.sys_log_mode > 0:
                Globals.log_df.add_log(datetime.now(), "TickerPredictor", "TickerPredictor", "predict",
                               "An order was created", self.ticker) 
                print("An order was created")
        
        return order
        
#Globals.log_df.export_df("C:/Users/eli_s/Documents/GitHub/HiLo-DEV/Forex/analysis docs")

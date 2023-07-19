import torch
import pickle
import Model
from torch.utils.data import DataLoader
from datetime import datetime
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import os
import shutil
from sklearn.metrics import f1_score
from alive_progress import alive_bar
import matplotlib.pyplot as plt
import Globals
import string
import random
import pandas as pd


class SingleTester():
    
    def __init__(self, model, TestData, verbose=False):
        
        self.TestData = TestData
        self.model = model
        self.device = Globals.device
        self.verbose = verbose
        
        if Globals.label_mode == "low": self.label_mode = "(-)"
        if Globals.label_mode == "high": self.label_mode = "(+)"
        
        self.folder_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Models")
        
    def format_batch(self, batch): 
        
        linear_data = []
        sequence_data = []
        labels = []
        
        for sample in batch:
            
            linear_data.append(sample.linear_input)
            sequence_data.append(sample.sequence_input)
            
            labels.append(sample.label)
                
        return linear_data, sequence_data, labels
    
    def test(self):
    
        if self.verbose == True: print("Started testing model")
        
        met = TestMetrics(self.model.ID)
        
        test_loader = DataLoader(self.TestData,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=0,
                                         collate_fn=lambda batch: self.format_batch(batch))
        
        with torch.no_grad():
            
            self.model = self.model.eval()
            
            test_iter = iter(test_loader)
            
            test_labels = []
            test_predictions = []
              
            for v in range(0, len(test_iter)):
                
                test_batch_linear_data, test_batch_sequence_data, test_batch_labels = next(test_iter)
                
                for sample in test_batch_labels:
                    test_labels.append(sample)
                
                test_batch_linear_data = torch.FloatTensor(np.array(test_batch_linear_data)).to(self.device).to(dtype=torch.float)
                test_batch_sequence_data = torch.FloatTensor(np.array(test_batch_sequence_data)).to(self.device).to(dtype=torch.float)
                test_batch_labels = torch.FloatTensor(test_batch_labels).reshape([len(test_batch_labels),1]).to(self.device).to(dtype=torch.float)
                
                output = self.model(test_batch_linear_data, test_batch_sequence_data)
                
                for sample in output.cpu().detach().numpy():
                    test_predictions.append(sample.round()[0])
            
            test_performance, prec, rec, AR, hits, _ = met.get_metrics(test_predictions, test_labels, False)
            
            return self.model.ID, test_performance, prec, rec, AR, hits, test_predictions, test_labels
        
class ModelTester():
    
    def __init__(self, TestData):
        
        self.TestData = TestData
        
        self.device = Globals.device
        
        if Globals.label_mode == "low": self.label_mode = "(-)"
        if Globals.label_mode == "high": self.label_mode = "(+)"
        
        self.folder_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Models")
        
    def test_by_txtfile(self, txt_path, show_combined_result = True):
        
        models = []
        
        met = TestMetrics("Combined")
        
        with open(txt_path) as f:
            model_IDs = f.readlines()
            
        for line in model_IDs:
            line.replace('\n', "")
            
        for ID in model_IDs:
            with open(self.folder_root + "/temp/" + Globals.ticker + ";" + ID.replace('\n', "") + self.label_mode, "rb") as fp:   
                model = pickle.load(fp)
                models.append(model)
         
        metric_results = []
        all_predictions = []
        
        with alive_bar(len(models)) as bar3:     
            for model in models:
                tester = SingleTester(model, self.TestData)
                ID, f1, prec, rec, AR, hits, predictions, test_labels = tester.test()
                result_line = [ID.replace('\n', ""), f1, prec, rec, AR, hits]
                metric_results.append(result_line)
                all_predictions.append(predictions)
                bar3()
        
        print("Individual results:")
        ind_df = pd.DataFrame(metric_results, columns = ['ID','f1','precision', 'recall', 'AR', 'hits'])
        print(ind_df)
        
        ind_df.to_excel(r'C:/Users/eli_s/Documents/GitHub/Project S V6/Forex/Data/' + Globals.ticker + "_all_" + Globals.label_mode + ".xlsx", index = False)
        
        profitable_models = []
        profitable_predictions = []
        
        threshold = 0.62
        
        profitable_models_IDs = []
        
        for u in (range(0, len(model_IDs))):
            model_result = ind_df.iloc[[u]]
            model_prec = model_result['precision'].values[0]
            if model_prec >= threshold and model_result['hits'].values[0] > 1:
                profitable_models.append(models[u])
                profitable_predictions.append(all_predictions[u])
                profitable_models_IDs.append(models[u].ID)
        
        selected_models_df = ind_df.loc[ind_df['ID'].isin(profitable_models_IDs)]
        ind_df = selected_models_df
        
        print("Selected models:")
        print(ind_df)
            
        if show_combined_result == True:
            
            combined_predictions = []
            predictions = np.array(profitable_predictions)
            
            if len(predictions) < 1:
                
                print("WARNING; no predictions were made")
                
                return None, None, None
            
            for u in range(0, len(predictions[0])):
                
                column = predictions[:,u]
                if np.sum(column) > 0:
                    combined_predictions.append(1)
                else:
                    combined_predictions.append(0)
            
            _, _, _, _, _, comb_df = met.get_metrics(combined_predictions, test_labels)    
            comb_df.to_excel(r'C:/Users/eli_s/Documents/GitHub/Project S V6/Forex/Data/' + Globals.ticker + "_combined_" + Globals.label_mode + ".xlsx", index = False)
            
            for model in profitable_models:
                
                with open("C:/Users/eli_s/Documents/GitHub/Project S V6/Forex/Models/prio/" + model.ticker + ";" + str(model.ID) + "_" + Globals.label_mode, "wb") as fp: 
                    pickle.dump(model, fp)
            
            return ind_df, comb_df, profitable_models
        
        
class TestMetrics():
    
    def __init__(self, ID):
        self.ID = ID
    
    def get_metrics(self, test_predictions, test_labels, verbose=True):
        
        test_performance = round(f1_score(test_labels, test_predictions), 3)
        prec = round(precision_score(test_labels, test_predictions), 3)
        rec = round(recall_score(test_labels, test_predictions), 3)
        AR = round(sum(test_predictions) / len(test_predictions), 3)
        hits = sum(test_predictions)
        df = pd.DataFrame([[self.ID, test_performance, prec, rec, AR, hits]], columns = ['ID','f1','precision', 'recall', 'AR', 'hits'])
        if verbose == True:
            print(df)
        
        return test_performance, prec, rec, AR, hits, df
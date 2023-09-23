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
        
        sequence_data = []
        labels = []
        
        for sample in batch:
            
            sequence_data.append(sample.sequence_input.astype(float))
            
            labels.append(sample.label)
                
        return sequence_data, labels
    
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
                
                test_batch_sequence_data, test_batch_labels = next(test_iter)
                
                for sample in test_batch_labels:
                    test_labels.append(sample)
                
                test_batch_sequence_data = torch.FloatTensor(np.array(test_batch_sequence_data)).to(self.device).to(dtype=torch.float)
                test_batch_labels = torch.FloatTensor(test_batch_labels).reshape([len(test_batch_labels),1]).to(self.device).to(dtype=torch.float)
                
                output = self.model(test_batch_sequence_data)
                
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
        
    def test_by_txtfile(self, txt_path):
        
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
         
        metric_results = ['ID', 'f1', 'precision', 'recall', 'AR', 'hits']
        
        for model in models:
            tester = SingleTester(model, self.TestData)
            ID, f1, prec, rec, AR, hits, predictions, test_labels = tester.test()
            result_line = [ID.replace('\n', ""), f1, prec, rec, AR, hits]
            metric_results.append(result_line)
        
        return metric_results, models[0]
        
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
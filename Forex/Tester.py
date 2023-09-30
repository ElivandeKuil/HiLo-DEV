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
import Globals
import string
import random
import pandas as pd
import math
from DataLoader import AE_train_dataloader, AE_test_dataloader

class SingleTester():
    
    def __init__(self, model, indices, verbose=False):
        
        self.test_loader = AE_test_dataloader(1, Globals.total_chunks, Globals.look_back, Globals.look_ahead, indices)
        self.model = model
        self.device = Globals.device
        self.verbose = verbose
        self.loss_function = torch.nn.MSELoss()
        
        self.folder_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Models")
    
    def test(self, error_threshold):
        
        Globals.phase = 'test'

        if self.verbose == True: print("Started testing model")
        
        met = TestMetrics(self.model.ID)
        
        with torch.no_grad():
            
            self.model = self.model.eval()
            
            test_labels = []
            test_predictions = []
              
            term = False

            with alive_bar(math.floor(Globals.total_chunks * 10000/2)) as bar:
                while term != True:
                    bar()

                    list_sequence_data, list_labels = self.test_loader.get_batch()
                    
                    if len(list_sequence_data) == 0:
                        term = True
                        break
                    
                    test_labels.append(list_labels[0])
                    
                    test_sequence_data = torch.FloatTensor(np.array(list_sequence_data)).to(self.device).to(dtype=torch.float)
                    
                    output, _ = self.model(test_sequence_data)

                    loss = self.loss_function(output, test_sequence_data)

                    prediction = 0
                    if loss > error_threshold:
                        prediction = 1

                    test_predictions.append(prediction)

            
            test_performance, prec, rec, AR, hits, df = met.get_metrics(test_predictions, test_labels, False)
            
            return self.model.ID, test_performance, prec, rec, AR, hits, test_predictions, test_labels, df
        
class ModelTester():
    
    def __init__(self, num_chunks):
        
        if Globals.label_mode == "low": self.label_mode = "(-)"
        if Globals.label_mode == "high": self.label_mode = "(+)"

        indices = self.get_random_indices(num_chunks, Globals.total_chunks)
        self.test_indices = indices[0:(math.floor(len(indices) / 2))]
        self.ultra_test_indices = indices[(math.floor(len(indices) / 2)):]
        
        self.folder_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Models")

    def cross_validate(self, model):

        error_thresholds = [100, 200, 300, 500]

        best_prec = 0
        best_thresh = 0

        print("Starting cross validation for error thresholds:")
        print(error_thresholds)

        for threshold in error_thresholds:
            ST = SingleTester(model, self.test_indices)
            _, performance, prec, rec, AR, hits, _, _, df = ST.test(threshold)

            print("Threshold of " + str(threshold) + " performed as such:")
            print(df)

            if prec > best_prec:

                print("This is the best performing threshold so far")
                best_prec = prec
                best_thresh = threshold
        

        print("Starting ultra test on model " + str(model.ID) + " with optimal threshold of " + str(best_thresh))

        ST = SingleTester(model, self.ultra_test_indices)
        _, performance, prec, rec, AR, hits, _, _, df = ST.test(best_thresh)

        print("Results:")
        print(df)



    def get_random_indices(self, num_chunks, total_chunks):

            random_indices = []
            while len(random_indices) < num_chunks:
                random_int = random.randint(0, total_chunks)
                if random_int in random_indices:
                    do = 'nothing'
                else:
                    random_indices.append(random_int)

            return random_indices

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
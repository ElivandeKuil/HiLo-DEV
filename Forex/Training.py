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
import math
import copy
import random
from DataLoader import AE_train_dataloader, AE_test_dataloader


class ModelTrainer:
    
    def __init__(self, model, num_chunks, verb = True):
        self.model = model
        self.device = Globals.device
        self.model.to(self.device)
        self.batch_size = Globals.batch_size
        self.learning_rate = Globals.learning_rate
        self.weight_decay = Globals.weight_decay
        self.n_epochs = Globals.num_epochs
        self.num_chunks = num_chunks
        self.verbose = verb
        self.folder_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Models")
        self.num_train_batches = (80000 / Globals.batch_size) * num_chunks
        self.num_val_batches = (10000 / Globals.batch_size) * num_chunks
        indices = self.get_random_indices(num_chunks, Globals.total_chunks)
        self.train_indeces = copy.deepcopy(indices)
        self.val_indices = copy.deepcopy(indices)
        
        if Globals.flush_temp == True:
            shutil.rmtree(os.path.join(self.folder_root, "temp"))
            os.mkdir(os.path.join(self.folder_root, "temp"))
        
        
        if Globals.label_mode == "low": self.label_mode = "(-)"
        if Globals.label_mode == "high": self.label_mode = "(+)"
        
        self.loss_function = torch.nn.MSELoss()
    
    def get_random_indices(self, num_chunks, total_chunks):

        random_indices = []
        while len(random_indices) < num_chunks:
            random_int = random.randint(0, total_chunks)
            if random_int in random_indices:
                do = 'nothing'
            else:
                random_indices.append(random_int)

        return random_indices

    def train_model(self, verbose=False):
        
        Globals.logger.log_and_print_line("Started training")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        highest_val_performance = -10
        best_epoch = 0   
        train_losses = []
        
        for epoch in range(0,self.n_epochs):

            train_loader = AE_train_dataloader(Globals.batch_size, self.num_chunks, Globals.total_chunks, Globals.look_back, Globals.look_ahead, 0, copy.deepcopy(self.train_indeces))
            val_loader = AE_train_dataloader(Globals.batch_size, self.num_chunks, Globals.total_chunks, Globals.look_back, Globals.look_ahead, 1, copy.deepcopy(self.val_indices))
            
            with alive_bar(math.floor(self.num_train_batches + self.num_val_batches)) as bar2:
            
                Globals.logger.log_and_print_line("Training epoch " + str(epoch + 1) + "...")
                
                ################################TRAINING################################
                
                self.model = self.model.train()
                
                epoch_train_total = 0
                epoch_val_total = 0
                term = False
                while term != True:
                    
                    list_batch_sequence_data = train_loader.get_batch()
                    if len(list_batch_sequence_data) == 0:
                        term = True
                        break

                    batch_sequence_data = torch.FloatTensor(np.array(list_batch_sequence_data)).to(self.device).to(dtype=torch.float)
                    
                    bar2()
                    
                    reconstructed, _ = self.model(batch_sequence_data)
                    
                    loss = self.loss_function(reconstructed, batch_sequence_data)
                    
                    epoch_train_total += loss.item()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
                
                ###############################Validating#################################
                
                with torch.no_grad():
                    
                    self.model = self.model.eval()
                    term2 = False
                    while term2 != True:
                        
                        val_batch_sequence_data = val_loader.get_batch()
                        if len(val_batch_sequence_data) == 0:
                            term2 = True
                            break

                        bar2()

                        val_batch_sequence_data = torch.FloatTensor(np.array(val_batch_sequence_data)).to(self.device).to(dtype=torch.float)
                        
                        val_reconstructed, _ = self.model(val_batch_sequence_data)
                        
                        val_loss = loss = self.loss_function(val_reconstructed, val_batch_sequence_data)
                        
                        epoch_val_total += val_loss.item()
                   
            current_loss = epoch_val_total / self.num_val_batches
            
            dt = datetime.now()
            
            time = str(dt.microsecond)
            chars = ''.join(random.choices(string.ascii_lowercase, k=3))

            if epoch > 3 and round(epoch_train_total / self.num_train_batches, 6) > train_losses[-4]:
                print('Training got stuck, aborting...')
                break
            train_losses.append(round(epoch_train_total / self.num_train_batches, 6))
            
            self.model.ID = Globals.ticker + ";" + str(chars) + str(time)
            line = "Finished epoch " + str(epoch + 1), " out of " + str(self.n_epochs) + ". modelID=" + str(self.model.ID) + "; Train loss = " + str(round(epoch_train_total / self.num_train_batches * 1000, 6)) + ", Validation loss = " + str(round(current_loss * 1000, 6)) 
            Globals.logger.log_and_print_line(line)
            
            with open(self.folder_root + "/temp/" + self.model.ticker + ";" + str(self.model.ID) + self.label_mode, "wb") as fp: 
                pickle.dump(self.model, fp)
            
            if current_loss > highest_val_performance:
                best_epoch_model = self.model.ID

        if len(best_epoch_model) > 0:       
            with open(self.folder_root + "/temp/" + self.model.ticker + ";" + str(best_epoch_model) + self.label_mode, "rb") as fp:   
                best_model = pickle.load(fp)
        else:
             best_model = None

        return best_model, best_epoch
    
    def test_model(self, model):
        
        if model == None:
            
            Globals.logger.log_and_print_line("No profitable model was found this run")
            Globals.logger.prio_log("No profitable model was found this run")
            return None, None
        else:
        
            Globals.logger.log_and_print_line("Started testing model")
            
            test_loader = DataLoader(self.TestData,
                                             batch_size=1,
                                             shuffle=True,
                                             num_workers=0,
                                             collate_fn=lambda batch: self.format_batch(batch))
            
            with torch.no_grad():
                
                model = model.eval()
                
                test_iter = iter(test_loader)
                
                test_labels = []
                test_predictions = []
                
                with alive_bar(len(self.TestData)) as bar3:
                                    
                    for v in range(0, len(test_iter)):
                        
                        bar3()
                        
                        test_batch_sequence_data, test_batch_labels = next(test_iter)
                        
                        for sample in test_batch_labels:
                            test_labels.append(sample)
                        
                        test_batch_sequence_data = torch.FloatTensor(np.array(test_batch_sequence_data)).to(self.device).to(dtype=torch.float)
                        test_batch_labels = torch.FloatTensor(test_batch_labels).reshape([len(test_batch_labels),1]).to(self.device).to(dtype=torch.float)
                        
                        output = model(test_batch_sequence_data)
                        
                        for sample in output.cpu().detach().numpy():
                            test_predictions.append(sample.round())
                
                test_performance = f1_score(test_labels, test_predictions)
                Globals.logger.log_and_print_line("Results:")
                Globals.logger.log_and_print_line("Model ID: " +str(model.ID))
                Globals.logger.log_and_print_line("F1:")
                Globals.logger.log_and_print_line(test_performance)
                Globals.logger.log_and_print_line("Precision:")
                Globals.logger.log_and_print_line(precision_score(test_labels, test_predictions))
                Globals.logger.log_and_print_line("Recall:")
                Globals.logger.log_and_print_line(recall_score(test_labels, test_predictions))
                Globals.logger.log_and_print_line("Action ratio:")
                Globals.logger.log_and_print_line(str(sum(test_predictions) / len(self.TestData)))
                
                
                return test_performance, model
   


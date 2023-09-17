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


class ModelTrainer:
    
    def __init__(self, model, traindata, valdata, testdata, verb = True):
        self.model = model
        self.device = Globals.device
        self.model.to(self.device)
        self.TrainData = traindata
        self.ValData = valdata
        self.TestData = testdata
        self.batch_size = Globals.batch_size
        self.learning_rate = Globals.learning_rate
        self.weight_decay = Globals.weight_decay
        self.n_epochs = Globals.num_epochs
        self.verbose = verb
        self.folder_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Models")
        
        if Globals.flush_temp == True:
            shutil.rmtree(os.path.join(self.folder_root, "temp"))
            os.mkdir(os.path.join(self.folder_root, "temp"))
        
        
        if Globals.label_mode == "low": self.label_mode = "(-)"
        if Globals.label_mode == "high": self.label_mode = "(+)"
        
        self.loss_function = torch.nn.BCELoss()
       
        self.plot_x = []
        self.plot_y_f1 = []
        self.plot_y_prec = []
        self.plot_y_rec = []
        self.plot_y_AR = []
    
    def format_batch(self, batch): 
        
        linear_data = []
        sequence_data = []
        labels = []
        
        for sample in batch:
            
            linear_data.append(sample.linear_input)
            sequence_data.append(sample.sequence_input)
            
            labels.append(sample.label)
                
        return linear_data, sequence_data, labels
    
    def train_model(self, verbose=False):
        
        Globals.logger.log_and_print_line("Started training")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        highest_val_performance = -10
        best_epoch = 0   
        best_epoch_model = None
        
        train_loader = DataLoader(self.TrainData,
                                         batch_size=self.batch_size,
                                         shuffle=True,
                                         num_workers=0,
                                         drop_last=True,
                                         collate_fn=lambda batch: self.format_batch(batch))
        val_loader = DataLoader(self.ValData,
                                         batch_size=self.batch_size,
                                         shuffle=True,
                                         num_workers=0,
                                         drop_last=True,
                                         collate_fn=lambda batch: self.format_batch(batch))
        
        for epoch in range(0,self.n_epochs):
            
            self.plot_x.append(epoch)
            
            with alive_bar(int((len(self.TrainData) + len(self.ValData))/self.batch_size)) as bar2:
            
                Globals.logger.log_and_print_line("Training epoch " + str(epoch + 1) + "...")
                
                ################################TRAINING################################
                
                self.model = self.model.train()
                train_iter = iter(train_loader)
                
                epoch_train_total = 0
                epoch_val_total = 0
                
                train_labels = []
                train_predictions = []
                               
                for u in range(0, len(train_iter)):
                    
                    list_batch_linear_data, list_batch_sequence_data, list_batch_labels = next(train_iter)
                    
                    for sample in list_batch_labels:
                        train_labels.append(sample)
                    
                    batch_linear_data = torch.FloatTensor(np.array(list_batch_linear_data)).to(self.device).to(dtype=torch.float)
                    
                    batch_sequence_data = torch.FloatTensor(np.array(list_batch_sequence_data)).to(self.device).to(dtype=torch.float)
                    
                    batch_labels = torch.FloatTensor(list_batch_labels).reshape([len(list_batch_labels),1]).to(self.device).to(dtype=torch.float)
                    
                    bar2()
                    
                    prediction = self.model(batch_linear_data, batch_sequence_data)
                    
                    for sample in prediction.cpu().detach().numpy():
                        train_predictions.append(sample.round().astype(int))
                    
                    loss = self.loss_function(prediction, batch_labels)
                        
                    epoch_train_total += loss.item()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
                
                ###############################Validating#################################
                
                with torch.no_grad():
                    
                    self.model = self.model.eval()
                    
                    val_iter = iter(val_loader)
                    
                    val_labels = []
                    val_predictions = []
                                        
                    for v in range(0, len(val_iter)):
                        
                        val_batch_linear_data, val_batch_sequence_data, val_batch_labels = next(val_iter)
                        
                        for sample in val_batch_labels:
                            val_labels.append(sample)
                        
                        bar2()
                        
                        val_batch_linear_data = torch.FloatTensor(np.array(val_batch_linear_data)).to(self.device).to(dtype=torch.float)
                        val_batch_sequence_data = torch.FloatTensor(np.array(val_batch_sequence_data)).to(self.device).to(dtype=torch.float)
                        val_batch_labels = torch.FloatTensor(val_batch_labels).reshape([len(val_batch_labels),1]).to(self.device).to(dtype=torch.float)
                        
                        output = self.model(val_batch_linear_data, val_batch_sequence_data)
                        
                        for sample in output.cpu().detach().numpy():
                            val_predictions.append(sample.round().astype(int))
                        
                        val_loss = self.loss_function(output, val_batch_labels.to(self.device).to(dtype=torch.float))
                        
                        epoch_val_total += val_loss.item()
                   
            current_loss = epoch_val_total / len(val_loader)
            
            val_performance = f1_score(val_labels, val_predictions)
            val_prec = precision_score(val_labels, val_predictions)
            train_performance = f1_score(train_labels, train_predictions)
        
            dt = datetime.now()
            
            time = str(dt.microsecond)
            chars = ''.join(random.choices(string.ascii_lowercase, k=3))
            
            self.model.ID = Globals.ticker + ";" + str(chars) + str(time)
            line = "Finished epoch " + str(epoch + 1), " out of " + str(self.n_epochs) + ". modelID=" + str(self.model.ID) + "; Train loss = " + str(round(epoch_train_total / len(train_loader), 3)) + " (f1: " + str(round(train_performance, 3)) + "), Validation loss = " + str(round(current_loss, 3)) + " (f1: " + str(round(val_performance, 3)) + ", Prec: "+ str(round(precision_score(val_labels, val_predictions), 2)) + ", Rec: " + str(round(recall_score(val_labels, val_predictions),2)) + ", AR: " + str(sum(val_predictions) / (len(val_loader) * self.batch_size) * 100) + "% => " + str(sum(val_predictions)) + " predictions)"
            Globals.logger.log_and_print_line(line)
            if precision_score(val_labels, val_predictions) > 0.50:
                Globals.logger.prio_log(line)
            
            self.plot_y_f1.append(val_performance)
            self.plot_y_prec.append(precision_score(val_labels, val_predictions))
            self.plot_y_rec.append(recall_score(val_labels, val_predictions))
            self.plot_y_AR.append(sum(val_predictions) / (len(val_loader) * self.batch_size))
            
            with open(self.folder_root + "/temp/" + self.model.ticker + ";" + str(self.model.ID) + self.label_mode, "wb") as fp: 
                pickle.dump(self.model, fp)
            
            best_epoch_model = ""
            
            if val_performance > highest_val_performance and val_prec > 0.6:
                
                highest_val_performance = val_performance
                best_epoch_model = self.model.ID 
                best_epoch = epoch + 1
        if len(best_epoch_model) > 0:       
            with open(self.folder_root + "/temp/" + self.model.ticker + ";" + str(best_epoch_model) + self.label_mode, "rb") as fp:   
                best_model = pickle.load(fp)
        else:
             best_model = None


        """    
        Globals.logger.log_plot(self.plot_x, self.plot_y_f1, "plot_" + self.model.ID + "_f1")
        Globals.logger.log_plot(self.plot_x, self.plot_y_prec, "plot_" + self.model.ID + "_precision")
        Globals.logger.log_plot(self.plot_x, self.plot_y_rec, "plot_" + self.model.ID + "_recall")
        Globals.logger.log_plot(self.plot_x, self.plot_y_AR, "plot_" + self.model.ID + "_action ratio")
        
        
        plt.plot(self.plot_x, self.plot_y_f1)
        plt.title("f1")
        plt.show()
        plt.clf()
        plt.plot(self.plot_x, self.plot_y_prec)
        plt.title("precision")
        plt.show()
        plt.clf()
        plt.plot(self.plot_x, self.plot_y_rec)
        plt.title("recall")
        plt.show()
        plt.clf()
        plt.plot(self.plot_x, self.plot_y_AR)
        plt.title("Action ratio")
        plt.show()
        plt.clf()
        """
        
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
                        
                        test_batch_linear_data, test_batch_sequence_data, test_batch_labels = next(test_iter)
                        
                        for sample in test_batch_labels:
                            test_labels.append(sample)
                        
                        test_batch_linear_data = torch.FloatTensor(np.array(test_batch_linear_data)).to(self.device).to(dtype=torch.float)
                        test_batch_sequence_data = torch.FloatTensor(np.array(test_batch_sequence_data)).to(self.device).to(dtype=torch.float)
                        test_batch_labels = torch.FloatTensor(test_batch_labels).reshape([len(test_batch_labels),1]).to(self.device).to(dtype=torch.float)
                        
                        output = model(test_batch_linear_data, test_batch_sequence_data)
                        
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
   


import os
from datetime import datetime
import Globals

class Logger:
    
    def __init__(self, verbose=True):
        
        self.ticker = ''
    
        self.Queue = []
        self.prioQueue = []
        self.verbose = verbose
        self.errorQueue = []
        
        self.plot_x = []
        self.plot_y = []
        self.legends = []
        self.plot_names = []
        
        self.save_plots = False
    
    def log_error(self, error):
        self.errorQueue.append(error)
    
    def log_and_print_line(self, line):
        
        print(line)
        
        self.Queue.append(line)
        
    def log_line(self, line):
        
        self.Queue.append(line)

    def log_globals(self):
        
        self.log_line("------------------------------------------------------------------------------")
        self.log_line("device: " + Globals.device)
        self.log_line("flush_temp: " + str(Globals.flush_temp))
        self.log_line("batch_size: " + str(Globals.batch_size))
        self.log_line("learning rate: " + str(Globals.learning_rate))
        self.log_line("weight decay: " + str(Globals.weight_decay))
        self.log_line("num_epochs: " + str(Globals.num_epochs))
        self.log_line("hdden layer size: " + str(Globals.hidden_layer_size))
        self.log_line("number layers: " + str(Globals.number_layers))
        self.log_line("balance data: " + str(Globals.balance_data))
        self.log_line("val split: " + str(Globals.val_split))
        self.log_line("test split: " + str(Globals.test_split))
        self.log_line("label_mode: " + str(Globals.label_mode))
        self.log_line("fine_tuning: " + str(Globals.fine_tuning))
        self.log_line("normalization: " + str(Globals.normalization))
        
        self.log_line("------------------------------------------------------------------------------")
        
        self.prio_log("------------------------------------------------------------------------------")
        self.prio_log("device: " + Globals.device)
        self.prio_log("flush_temp: " + str(Globals.flush_temp))
        self.prio_log("batch_size: " + str(Globals.batch_size))
        self.prio_log("learning rate: " + str(Globals.learning_rate))
        self.prio_log("weight decay: " + str(Globals.weight_decay))
        self.prio_log("num_epochs: " + str(Globals.num_epochs))
        self.prio_log("hdden layer size: " + str(Globals.hidden_layer_size))
        self.prio_log("number layers: " + str(Globals.number_layers))
        self.prio_log("balance data: " + str(Globals.balance_data))
        self.prio_log("val split: " + str(Globals.val_split))
        self.prio_log("test split: " + str(Globals.test_split))
        self.prio_log("label_mode: " + str(Globals.label_mode))
        self.prio_log("fine_tuning: " + str(Globals.fine_tuning))
        self.log_line("normalization: " + str(Globals.normalization))
        
        self.prio_log("------------------------------------------------------------------------------")
        
    
    def reset(self):
        
        self.ticker = ''
    
        self.Queue = []
        self.prioQueue = []
        self.simQueue = []
        self.single_dict = {}
        self.dual_dict = {}
        
        self.plot_x = []
        self.plot_y = []
        self.legends = []
        self.plot_names = []
        
    def prio_log(self, line):
        
        print(line)
        self.prioQueue.append(line)
    
    def Dump(self):
        
        print("Dumping logs...")
        
        log_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Logs")
        if Globals.label_mode == "low": self.log_folder_root = os.path.join(log_folder_path, Globals.ticker + ";" + str(datetime.now()).replace(':', '') + "; (-)") 
        if Globals.label_mode == "high": self.log_folder_root = os.path.join(log_folder_path, Globals.ticker + ";" + str(datetime.now()).replace(':', '') + "; (+)") 
        os.makedirs(self.log_folder_root)
        
        self.full_log_path = os.path.join(self.log_folder_root, "full_log.txt")
        self.prio_log_path = os.path.join(self.log_folder_root, "prio_log.txt")
        
        f = open(self.full_log_path, 'w')
        f = open(self.prio_log_path, 'w')
        
        with open(self.full_log_path, 'a') as file:
            
            if len(self.errorQueue) > 0:
                file.write(self.errorQueue[0])
            
            for line in self.Queue:
                file.write(str(line) + "\n")
                
        with open(self.prio_log_path, 'a') as file:
            
            if len(self.errorQueue) > 0:
                file.write(self.errorQueue[0])
            
            for line in self.prioQueue:
                file.write(str(line) + "\n")
        
    def log_plot(self, x_values, y_values, name, legend=[]):
        self.plot_x.append(x_values)
        self.plot_y.append(y_values)
        self.plot_names.append(name)
        self.legends.append(legend)
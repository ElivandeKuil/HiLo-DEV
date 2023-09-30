from Model import NTDM_V0, NTDM_V1, LSTM, ST_AutoEncoder
from Training import ModelTrainer
from Tester import ModelTester
import torch
import warnings 
import Globals
import time
import numpy
from TickerPredictor import TickerPredictor
import pickle
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
from Miner import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
folder_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Data/Pickled")

def run(num_chunks):
    warnings.filterwarnings("ignore")
    
    program_start_time = time.time()
    
    Globals.logger.log_globals()
    
    if Globals.fine_tuning == False:

        model = ST_AutoEncoder(100)
        model.ticker = Globals.ticker

    else:
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename()
        with open(file_path, "rb") as fp:   
            ticker_predictor = pickle.load(fp)
            model = ticker_predictor.model

        model.sequence_encoder.device = Globals.device
        model.sequence_encoder.to(Globals.device)
        model.linear_encoder.to(Globals.device)
        
        model = model.to(Globals.device)
    
    trainer = ModelTrainer(model, num_chunks)
    
    best_model, _ = trainer.train_model(True)
    
    MT = ModelTester(1)

    MT.cross_validate(best_model)

    program_end_time = time.time()
    total_time = program_end_time - program_start_time
    
    Globals.logger.log_and_print_line("Total program took " + str(round(total_time, 2)) + "s")
    
    Globals.logger.Dump()


def create_ticker_predictor(ticker, Id, name, max_spread, model, doc, labelmode):
    path = "C:/Users/eliva/OneDrive/Documents/GitHub/HiLo-DEV/Forex/tickerpredictors/"
    
    new_ticker_predictor = TickerPredictor(
        ticker, 
        Id, 
        name, 
        max_spread,
        labelmode)
    
    new_ticker_predictor.load_models(model, doc)
    
    with open(path + ticker.replace('.', '_') + ".tp", "wb") as fp: 
        pickle.dump(new_ticker_predictor, fp)

Globals.fine_tuning = False
Globals.logger.reset()
Globals.balance_data = True
Globals.balance_percent = .3
Globals.device = device
Globals.batch_size = 1024
Globals.learning_rate = 0.01
Globals.weight_decay = 1e-8
Globals.num_epochs = 1          
Globals.val_split = .1
Globals.test_split = .1
Globals.normalization = True
Globals.total_chunks = 38

Globals.look_ahead = 60
Globals.look_back = 120
Globals.TP = 0.03
Globals.SL = 0.01
Globals.label_mode = "high"
Globals.ticker = "XAUUSD"

run(1)


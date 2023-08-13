import Preprocessing
from Model import NTDM_V0, LSTM
from Training import ModelTrainer
import torch
import warnings 
import Globals
import Tester
import time
import numpy
from TickerPredictor import TickerPredictor
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run():
    warnings.filterwarnings("ignore")
    
    program_start_time = time.time()
    
    counter = 0
    
    for var_num_layers in [2]:
        for var_hidden_size in [6, 8, 10, 12]:
            
            counter += 1
            
            Globals.device = device
            Globals.batch_size = 24
            Globals.learning_rate = 0.01
            Globals.weight_decay = 1e-8
            Globals.num_epochs = 30
            
            Globals.hidden_layer_size = var_hidden_size
            Globals.number_layers = var_num_layers
            Globals.balance_data = False
            
            Globals.val_split = .1
            Globals.test_split = .1
            
            Globals.logger.log_globals()
            
            for u in range(0, 1):
                
                start_time= time.time()
                
                Globals.logger.log_and_print_line("(" + str(counter) + "); Parameters run "  + str(u + 1) + " out of 1")
                
                traindata, valdata, testdata = Preprocessing.get_split_data(Globals.ticker, Globals.val_split, Globals.test_split)
                
                model = NTDM_V0(Globals.device)
                
                model.ticker = Globals.ticker
                
                trainer = ModelTrainer(model, traindata, valdata, testdata)
                
                test1, test2 = trainer.train_model(True)
                inter_time = time.time()
                ellapsed_time = inter_time - start_time
                
                Globals.logger.log_and_print_line("Run took " + str(round(ellapsed_time, 2)) + "s")
    
    program_end_time = time.time()
    total_time = program_end_time - program_start_time
    
    Globals.logger.log_and_print_line("Total program took " + str(round(total_time, 2)) + "s")
    
    Globals.logger.Dump()

def safe_run():
    try:
        run()
        
    except Exception as e: 
        error = 'Program was terminated due to following error: '+ str(e)
        Globals.logger.log_error(error)
        Globals.logger.Dump()


def dual_label_mode_run():
    Globals.logger.reset()
    Globals.label_mode = "low"
    run()
    Globals.logger.reset()
    Globals.label_mode = "high"
    run()
    

def test_models(ticker, txtfolder_name, spread, create_predictor=False):
    
    Globals.device = device
    Globals.val_split = .1
    Globals.test_split = .1
    Globals.ticker = ticker
    
    print("##################" + Globals.ticker + "##################") 
    
    Globals.label_mode = 'high'
    txt_path = "C:/Users/eliva/OneDrive/Documents/GitHub/HiLo-DEV/Forex/analysis docs/Candidates/" + txtfolder_name + "/high_candidate.txt"
    
    _, _, high_testdata = Preprocessing.get_split_data(Globals.ticker, Globals.val_split, Globals.test_split)
    
    print("Started testing high models...")
    
    tester = Tester.ModelTester(high_testdata)
    high_doc, high_model = tester.test_by_txtfile(txt_path)
    
    
    Globals.label_mode = 'low'
    txt_path = "C:/Users/eliva/OneDrive/Documents/GitHub/HiLo-DEV/Forex/analysis docs/Candidates/" + txtfolder_name + "/low_candidate.txt"
    
    _, _, low_testdata = Preprocessing.get_split_data(Globals.ticker, Globals.val_split, Globals.test_split)
    
    print("Started testing low models...")
    
    tester = Tester.ModelTester(low_testdata)
    low_doc, low_model = tester.test_by_txtfile(txt_path)
    
    print("High model score:")
    print(high_doc)
    print("low modelscore:")
    print(low_doc)

    if create_predictor == True:
        choice = input("You have to choose one of the two tested models, type 'h' for high model and l for low model")

        if choice == 'h':

            model = high_model
            doc = high_doc
            labelmode = '+'

        if choice == 'l':
            
            model = low_model
            doc = low_doc
            labelmode = '-'

        if model != None:
        
            create_ticker_predictor(ticker, ticker + "(1)", ticker + "(simple three feaute LSTM + Linear vector)", spread, 
                                model, doc, labelmode)
        

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


def generate_ticket_predictors():
    
    symbols = [ "XAUUSD", "EURUSD", "USDJPY", "GBPUSD"]
    folders = [ "XAUUSD", "EURUSD", "USDJPY", "GBPUSD"]
    spreads = [  0.00012, 0.00012,0.00012, 0.00012]
    
    for u in range (0, len(symbols)):
        test_models(symbols[u], folders[u], spreads[u], create_predictor=True)

generate_ticket_predictors()

"""
Globals.ticker = "EURUSD"
dual_label_mode_run()
Globals.ticker = "USDJPY"
dual_label_mode_run()
Globals.ticker = "GBPUSD"
dual_label_mode_run()
Globals.ticker = "XAUUSD"
dual_label_mode_run()
"""

from datetime import datetime
import MetaTrader5 as mt5
import pickle
import numpy as np
import datetime
from dateutil import parser
import random
import os
import Globals
from alive_progress import alive_bar
from datetime import datetime, timedelta

print("MetaTrader5 package author: ",mt5.__author__)
print("MetaTrader5 package version: ",mt5.__version__)
 
import pandas as pd
import pytz
 
if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()
 
folder_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Data/Pickled")

class Dataset:
    def get_raw_data(self, symbol):
        Globals.ticker = symbol
        print("Downloading data...")
        # set time zone to UTC
        timezone = pytz.timezone("Etc/UTC")
        # create 'datetime' object in UTC time zone to avoid the implementation of a local time zone offset
        utc_from = datetime(2020, 1, 10, tzinfo=timezone)
        ticks = mt5.copy_ticks_from(symbol, utc_from, 300000, mt5.COPY_TICKS_ALL)
        print("Ticks received:",len(ticks))
        
        mt5.shutdown()
        return ticks

    def remove_duplicate_dates(self, raw_data):

        print("Removing duplicates from data..")

        previous_date = 0
        unique_dates_data = []
        with alive_bar(len(raw_data)) as bar:
            for row in raw_data:
                bar()
                if row[0] == previous_date:
                    do = 'nothing'
                else:
                    unique_dates_data.append(row)
                    previous_date = row[0]

        return unique_dates_data
    
    def process_data(self, unique_dates_data, lookahead):
        ticks = []
        print("Trimming data...")
        with alive_bar(len(unique_dates_data)) as bar:
            for i in range(0, len(unique_dates_data)):
                bar()
                ticks.append(tick(unique_dates_data[i], unique_dates_data[i][0] - unique_dates_data[i - 1][0]))

        processed_data= []
        current_day = datetime.fromtimestamp(ticks[0].ID).day
        
        with alive_bar(len(ticks)) as bar:
            print("Converting data to model input...")
            for u in range(130, len(ticks) - 310):
                bar()
                if datetime.fromtimestamp(ticks[u].ID).day == current_day:
                    
                    if datetime.fromtimestamp(ticks[u].ID).hour < 1 and datetime.fromtimestamp(ticks[u].ID).isoweekday() == 1:
                
                        do = 'nothing'
                        
                    else:
                        new_input = model_input(ticks[u], ticks[u - 120: u], ticks[u:u + 300], lookahead)
                        if len(new_input.sequence_input) > 0:
                            processed_data.append(new_input)
                else:
                    current_day = datetime.fromtimestamp(ticks[u].ID).day
        return processed_data

    def analyse_data(self, data, TP, SL):
        
        pos = 0
        neg = 0
        empty = 0
        
        for point in data:
            if point.high_label >= TP and point.low_label > (-1 * SL):
                pos += 1
            if point.low_label <= (-1 * TP) and point.high_label < SL:
                neg += 1
            else:
                empty += 1
            
        Globals.logger.log_and_print_line("number datapoints: " + str(len(data)))
        Globals.logger.log_and_print_line("high label >= 0.03 perecentage: " + str(pos / len(data)))
        Globals.logger.log_and_print_line("low label <= 0.03 perecentage: " + str(neg / len(data)))

    def save_data(self, all_data, look_back, look_ahead):

        with open(folder_root + "/" + Globals.ticker + "_" + Globals.label_mode + ",lb=" + str(look_back) + "s,la=" + str(look_ahead) + "f,norm=" + str(Globals.normalization) + ".Data", "wb") as fp: 
            pickle.dump(all_data, fp)

    def save_split_data(self, all_data, val_size, test_size, look_back, look_ahead):
        np.random.shuffle(all_data)
        val_count = int(len(all_data) * val_size)
        test_count = int(len(all_data) * test_size)
        
        traindata = all_data[:len(all_data) - val_count - test_count]
        
        valdata = all_data[len(all_data) - val_count - test_count: len(all_data) - test_count]
        testdata = all_data[len(all_data) - test_count:]

        with open(folder_root + "/" + Globals.ticker + "_" + Globals.label_mode + ",lb=" + str(look_back) + "s,la=" + str(look_ahead) + "f,norm=" + str(Globals.normalization) + "_TrainData.ProDa", "wb") as fp: 
            pickle.dump(testdata, fp)
        with open(folder_root + "/" + Globals.ticker + "_" + Globals.label_mode + ",lb=" + str(look_back) + "s,la=" + str(look_ahead) + "f,norm=" + str(Globals.normalization) + "_ValData.ProDa", "wb") as fp: 
            pickle.dump(valdata, fp)
        with open(folder_root + "/" + Globals.ticker + "_" + Globals.label_mode + ",lb=" + str(look_back) + "s,la=" + str(look_ahead) + "f,norm=" + str(Globals.normalization) + "_TestData.ProDa", "wb") as fp: 
            pickle.dump(traindata, fp)

        return traindata, valdata, testdata
    
    
    def label_gun(self, data, label_mode, TP, SL):
        
        counter = 0
        with alive_bar(len(data)) as bar:
            print("Labeling...")
            for datum in data:
                bar()
                if label_mode == 'high':
                    if datum.high_label >= TP and datum.low_label > (-1 * SL):
                        datum.label = 1
                        counter += 1
                if label_mode == 'low':
                    if datum.low_label <= (-1 * TP) and datum.high < SL:
                        datum.label = 1
                        counter += 1

        Globals.logger.log_and_print_line("Label 1 percentage = " + str(round(counter/len(data) ,3)))

        return data
        
    def get_data(self, symbol, label_mode, look_back, look_ahead, TP=0.05, SL=0.05, val_size = .1, test_size = .1): 
        
        Globals.ticker = symbol 
        Globals.sequence_length = look_back
        
        try:
            Globals.logger.log_and_print_line("Found pickled data, loading...")

            with open(folder_root + "/" + Globals.ticker + "_" + Globals.label_mode + ",lb=" + str(look_back) + "s,la=" + str(look_ahead) + "f,norm=" + str(Globals.normalization) + "_TrainData.ProDa", "rb") as fp:   
                testdata = pickle.load(fp)
            with open(folder_root + "/" + Globals.ticker + "_" + Globals.label_mode + ",lb=" + str(look_back) + "s,la=" + str(look_ahead) + "f,norm=" + str(Globals.normalization) + "_ValData.ProDa", "rb") as fp:   
                valdata = pickle.load(fp)
            with open(folder_root + "/" + Globals.ticker + "_" + Globals.label_mode + ",lb=" + str(look_back) + "s,la=" + str(look_ahead) + "f,norm=" + str(Globals.normalization) + "_TestData.ProDa", "rb") as fp:   
                traindata = pickle.load(fp)
            
        except:   

            Globals.logger.log_and_print_line("Mining and processing data...")
            raw_data = self.get_raw_data(symbol)
            unique_dates_data = self.remove_duplicate_dates(raw_data)
            del(raw_data)
            processed_data = self.process_data(unique_dates_data, look_ahead)
            del(unique_dates_data)
            self.analyse_data(processed_data, TP, SL)
            self.save_data(processed_data, look_back, look_ahead)
            traindata, valdata, testdata = self.save_split_data(processed_data, val_size, test_size, look_back, look_ahead)
            del(processed_data)
        
        traindata = self.label_gun(traindata, label_mode, TP, SL)
        valdata = self.label_gun(valdata, label_mode, TP, SL)
        testdata = self.label_gun(testdata, label_mode, TP, SL)

        return traindata, valdata, testdata

        
            
def build_sequence_data(prequel_ticks):
        
    sequence = []
    bid_window = []
    ask_window = []
    time_gap_window = []

    for u in range(1, len(prequel_ticks)):
        
        bid_window.append(get_percentage(prequel_ticks[u].bid, prequel_ticks[u - 1].bid, 'string'))
        ask_window.append(get_percentage(prequel_ticks[u].ask, prequel_ticks[u - 1].ask, 'string'))
        xmax = 20
        xmin = 1
        if Globals.normalization == False:
            time_gap_window.append(str(prequel_ticks[u].time_gap)[0:5])
        else:
            time_gap_window.append(str((prequel_ticks[u].time_gap - xmin) / (xmax - xmin))[0:5])

    sequence = []
    for w in range(0, len(bid_window)):
        sequence.append([bid_window[w], ask_window[w], time_gap_window[w]])
        
    np.flip(np.array(sequence), 0)
    
    return sequence

def get_percentage(new, old, type):
    if type == 'float':
        return ((new - old)/old) * 100
    if type == 'string':
        xmax = 0.1
        xmin = -0.1
        if Globals.normalization == True:
            return str(((((new - old)/old) * 100) - xmin) / (xmax - xmin))[0:6]
        else:
            return str(((new - old)/old) * 100)[0:6]

class tick():
    
    def __init__(self, datum, time_gap):
        
        self.ID = datum[0]
        self.bid = datum[1]
        self.ask = datum[2]
        self.time_gap = time_gap
    
class model_input():
    def __init__(self, current_tick, prequel_ticks, sequel_ticks, look_ahead):
        
        self.high_label = 0
        self.low_label = 0
        self.label = 0

        for seq_tick in sequel_ticks:

            if datetime.fromtimestamp(seq_tick.ID) - timedelta(seconds=look_ahead) <= datetime.fromtimestamp(current_tick.ID):
                if get_percentage(seq_tick.bid, current_tick.bid, 'float') > self.high_label:
                    self.high_label = get_percentage(seq_tick.bid, current_tick.bid, 'float')
                else:
                    if get_percentage(seq_tick.bid, current_tick.bid, 'float') < self.low_label:
                        self.low_label = get_percentage(seq_tick.bid, current_tick.bid, 'float')
                
        self.ID = current_tick.ID
        self.sequence_input = build_sequence_data(prequel_ticks)
            








"""

import tkinter as tk
from tkinter import filedialog
import pickle

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()

with open(file_path, "rb") as fp:   
                best_model = pickle.load(fp)

debug1 = best_model[0]
debug2 = best_model[1]
de = 0
"""
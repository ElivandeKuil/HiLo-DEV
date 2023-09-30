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
import math

print("MetaTrader5 package author: ",mt5.__author__)
print("MetaTrader5 package version: ",mt5.__version__)
 
import pandas as pd
import pytz
 
if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()
 
class Dataset:
    def get_raw_data(self, symbol):
        Globals.ticker = symbol
        print("Downloading data...")
        # set time zone to UTC
        timezone = pytz.timezone("Etc/UTC")
        # create 'datetime' object in UTC time zone to avoid the implementation of a local time zone offset
        utc_from = datetime(2020, 1, 10, tzinfo=timezone)
        ticks = mt5.copy_ticks_from(symbol, utc_from, 100000000, mt5.COPY_TICKS_ALL)
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

    def save_split_data(self, all_data, val_size, test_size, look_back, look_ahead, index, folder_root):
        np.random.shuffle(all_data)
        val_count = int(len(all_data) * val_size)
        test_count = int(len(all_data) * test_size)
        
        traindata = all_data[:len(all_data) - val_count - test_count]
        
        valdata = all_data[len(all_data) - val_count - test_count: len(all_data) - test_count]
        testdata = all_data[len(all_data) - test_count:]

        with open(folder_root + "/" + Globals.ticker + str(index) + "_" + Globals.label_mode + ",lb=" + str(look_back) + "s,la=" + str(look_ahead) + "f,norm=" + str(Globals.normalization) + "_TrainData.ProDa", "wb") as fp: 
            pickle.dump(testdata, fp)
        with open(folder_root + "/" + Globals.ticker + str(index) + "_" + Globals.label_mode + ",lb=" + str(look_back) + "s,la=" + str(look_ahead) + "f,norm=" + str(Globals.normalization) + "_ValData.ProDa", "wb") as fp: 
            pickle.dump(valdata, fp)
        with open(folder_root + "/" + Globals.ticker + str(index) + "_" + Globals.label_mode + ",lb=" + str(look_back) + "s,la=" + str(look_ahead) + "f,norm=" + str(Globals.normalization) + "_TestData.ProDa", "wb") as fp: 
            pickle.dump(traindata, fp)

        return traindata, valdata, testdata
    
    
    def label_gun(self, data, label_mode, TP, SL):
        
        if data != None: 

            counter = 0
            print("Labeling...")
            for datum in data:
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
        return False
      
    def get_data_chunk_by_index(self, look_back, look_ahead, TP, SL, data_index, specifier = 'all'):
        folder_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Data")
        folder_root = folder_root + "/" + Globals.ticker + "_" + Globals.label_mode + ",lb=" + str(look_back) + "s,la=" + str(look_ahead) + "f,norm=" + str(Globals.normalization)
           
        try:

            Globals.logger.log_and_print_line("Found pickled data, loading...")

            if specifier == 'all' or specifier == 'test':
                with open(folder_root + "/" + Globals.ticker + str(data_index) + "_" + Globals.label_mode + ",lb=" + str(look_back) + "s,la=" + str(look_ahead) + "f,norm=" + str(Globals.normalization) + "_TrainData.ProDa", "rb") as fp:   
                    testdata = pickle.load(fp)
            else:
                testdata = None

            if specifier == 'all' or specifier == 'val':
                with open(folder_root + "/" + Globals.ticker + str(data_index) + "_" + Globals.label_mode + ",lb=" + str(look_back) + "s,la=" + str(look_ahead) + "f,norm=" + str(Globals.normalization) + "_ValData.ProDa", "rb") as fp:   
                    valdata = pickle.load(fp)
            else:
                valdata = None

            if specifier == 'all' or specifier == 'train':
                with open(folder_root + "/" + Globals.ticker + str(data_index) + "_" + Globals.label_mode + ",lb=" + str(look_back) + "s,la=" + str(look_ahead) + "f,norm=" + str(Globals.normalization) + "_TestData.ProDa", "rb") as fp:   
                    traindata = pickle.load(fp)
            else:
                traindata = None
            
            traindata = self.label_gun(traindata, Globals.label_mode, TP, SL)
            valdata = self.label_gun(valdata, Globals.label_mode, TP, SL)
            testdata = self.label_gun(testdata, Globals.label_mode, TP, SL)

            return traindata, valdata, testdata
        except:
            return False, False, False

    def load_data_chunks(self, num_chunks, look_back, look_ahead, val_size=0.1, test_size=0.1): 
        folder_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Data")
        if os.path.isdir(folder_root+ "/" + Globals.ticker + "_" + Globals.label_mode + ",lb=" + str(look_back) + "s,la=" + str(look_ahead) + "f,norm=" + str(Globals.normalization)):
            count = 0
            folder_root = folder_root+ "/" + Globals.ticker + "_" + Globals.label_mode + ",lb=" + str(look_back) + "s,la=" + str(look_ahead) + "f,norm=" + str(Globals.normalization)
            for path in os.listdir(folder_root):
                if os.path.isfile(os.path.join(folder_root, path)):
                    count += 1
            count = math.ceil(count/3)
            if count < num_chunks:
                raw_data = self.get_raw_data(Globals.ticker)
                unique_dates_data = self.remove_duplicate_dates(raw_data)
                del(raw_data)
                print("Found " + str(count) + " data chunks. Started mining " + str(num_chunks - count) + " more (100.000 items each)...")
                for i in range(count, math.ceil(len(unique_dates_data)/100000)):
                    if i + 1 <= num_chunks:
                        print("Processing chunk " + str(i))
                        processed_data = self.process_data(unique_dates_data[(i * 100000): (i+1) * 100000], look_ahead)
                        self.save_split_data(processed_data, val_size, test_size, look_back, look_ahead, i, folder_root)
                        del(processed_data)
                print("Finished loading all chunks")
            else:
                print("All requested datachunks were already present")
        else:
            os.mkdir(folder_root+ "/" + Globals.ticker + "_" + Globals.label_mode + ",lb=" + str(look_back) + "s,la=" + str(look_ahead) + "f,norm=" + str(Globals.normalization))
            folder_root = folder_root+ "/" + Globals.ticker + "_" + Globals.label_mode + ",lb=" + str(look_back) + "s,la=" + str(look_ahead) + "f,norm=" + str(Globals.normalization)
            raw_data = self.get_raw_data(Globals.ticker)
            unique_dates_data = self.remove_duplicate_dates(raw_data)
            del(raw_data)
            print("Started mining " + str(num_chunks) + " datachunks (100.000 items each)...")
            for i in range(0, math.ceil(len(unique_dates_data)/100000)):
                if i + 1 <= num_chunks:
                    print("Processing chunk " + str(i))
                    processed_data = self.process_data(unique_dates_data[(i * 100000): (i+1) * 100000], look_ahead)
                    self.save_split_data(processed_data, val_size, test_size, look_back, look_ahead, i, folder_root)
                    del(processed_data)
            print("Finished loading all chunks")

        


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
            

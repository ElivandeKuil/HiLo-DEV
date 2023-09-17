import pickle
import numpy as np
import datetime
from dateutil import parser
import time
import random
import os
import Globals
from alive_progress import alive_bar
from datetime import datetime, timedelta

TP = ".04"
SL = ".03"

folder_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Data/Pickled/Raw/" + TP + "-" + SL)

def balance_data(train_data, bal_factor):
    balanced_data = []
    for datum in train_data:
        if datum.label == 0 and random.randint(0, 1000) < bal_factor * 1000:
            do = 'nothing'
        else:
            balanced_data.append(datum)
    return balanced_data


def analyse_data(data):
    
    pos = 0
    empty = 0
    
    for point in data:
        if point.label == 1:
            pos += 1
        else:
            empty += 1
        
    Globals.logger.log_and_print_line("number datapoints: " + str(len(data)))
    Globals.logger.log_and_print_line("label=1 perecentage: " + str(pos / len(data)))
    Globals.logger.log_and_print_line("Label=0 perecentage: " + str(empty / len(data)))

    return empty / len(data)

def get_split_data(ticker, val_size, test_size):
    
    """
    Sizes are in percentage of the total (going from 0 to 1)
    """
    
    try:
        
        with open(folder_root + "/" + Globals.ticker + ";" + Globals.label_mode + "_TestData.ProDa", "rb") as fp:   
            testdata = pickle.load(fp)
        with open(folder_root + "/" + Globals.ticker + ";" + Globals.label_mode + "_ValData.ProDa", "rb") as fp:   
            valdata = pickle.load(fp)
        with open(folder_root + "/" + Globals.ticker + ";" + Globals.label_mode + "_TrainData.ProDa", "rb") as fp:   
            traindata = pickle.load(fp)
        
        Globals.logger.log_and_print_line("Found pickled data, loading...")
        
    except:
        
        Globals.logger.log_and_print_line("Loading and formatting data...")
        
        all_data = np.array(format_data(ticker))
        np.random.shuffle(all_data)
        val_count = int(len(all_data) * val_size)
        test_count = int(len(all_data) * test_size)
        
        traindata = all_data[:len(all_data) - val_count - test_count]
        
        valdata = all_data[len(all_data) - val_count - test_count: len(all_data) - test_count]
        testdata = all_data[len(all_data) - test_count:]

        with open(folder_root + "/" + Globals.ticker + ";" + Globals.label_mode + "_TestData.ProDa", "wb") as fp: 
            pickle.dump(testdata, fp)
        with open(folder_root + "/" + Globals.ticker + ";" + Globals.label_mode + "_ValData.ProDa", "wb") as fp: 
            pickle.dump(valdata, fp)
        with open(folder_root + "/" + Globals.ticker + ";" + Globals.label_mode + "_TrainData.ProDa", "wb") as fp: 
            pickle.dump(traindata, fp)

    Globals.logger.prio_log("traindata properties before balancing:")
    analyse_data(traindata)

    if (Globals.balance_data == True):
        traindata = balance_data(traindata, Globals.balance_percent)
        
    Globals.logger.prio_log("traindata properties after balancing:")
    analyse_data(traindata)
    return traindata, valdata, testdata

def format_data(ticker):

    with open(folder_root + "/" + ticker + ".data", "rb") as fp:   # Unpickling
        raw_data = pickle.load(fp)
        
    dic = list(raw_data.values())[0]
    data = list(dic.items())

    time_frames = []
    for point in data:
        time_frames.append(time_frame(point))

    processed_data= []
    current_day = time_frames[0].ID[0:10]
    
    with alive_bar(len(data)) as bar:
    
        for u in range(0, len(time_frames)):
            bar()
            if time_frames[u].ID[0:10] == current_day:
                
                if int(time_frames[u].ID[11:13]) < 1 and datetime.strptime(time_frames[u].ID[0:10].replace('.', '/'), '%Y/%m/%d').isoweekday() == 1:
               
                    do = 'nothng'
                    
                else:
                    new_input = model_input(time_frames[u], time_frames[u - 8: u])
                    if len(new_input.sequence_input) > 0:
                        processed_data.append(new_input)
            else:
                current_day = time_frames[u].ID[0:10]
        

    return processed_data



class time_frame():
    
    def __init__(self, tup):
        
        self.ID = tup[0]
        self.high = self.get_percentage(tup[1]['high'], tup[1]['open'])
        self.low = self.get_percentage(tup[1]['low'], tup[1]['open'])
        self.close = self.get_percentage(tup[1]['close'], tup[1]['open'])
        self.volume = tup[1]['tick_volume']
        
        
    def get_percentage(self, new, old):
        return (new - old) / old * 100
    

class compact_input():
    def __init__(self, sequence_input, linear_input, label):
        self.linear_input = linear_input
        self.sequence_input = sequence_input
        self.label = label

class model_input():
    def __init__(self, tf, lookback_frames):
        
        self.coming_high = tf.high
        self.coming_low = tf.low
        self.coming_close = tf.close
        
        self.ID = tf.ID
        self.month = tf.ID[5:7]
        self.day = tf.ID[8:10]
        self.sequence_input = self.build_sequence_data(lookback_frames)
        
        self.weekday = parser.parse(self.ID).strftime("%a").lower()
        
        if self.weekday == 'mon':
           self.weekday = 0
        if self.weekday == 'tue':
            self.weekday = 1
        if self.weekday == 'wed':
            self.weekday = 2
        if self.weekday == 'thu':
            self.weekday = 3
        if self.weekday == 'fri':
            self.weekday = 4
        if self.weekday == 'sat':
            self.weekday = 5
        if self.weekday == 'sun':
            self.weekday = 6

        self.time_slot = int(self.get_time_slot_index(tf.ID[11:]))
        self.linear_input = self.build_linear_vector()
        
        self.label = 0
        if Globals.label_mode == 'low':
            if tf.low <= -1 * float(TP) and tf.high <= float(SL):
                self.label = 1
        if Globals.label_mode == 'high':
            if tf.high >= float(TP) and tf.low >= -1 * float(SL):
                self.label = 1
            
    def build_linear_vector(self):
        
        time_slot_vector = self.ohe_builder(self.time_slot, 312)
        month_vector = self.ohe_builder(int(self.month) - 1, 12)
        weekday_vector = self.ohe_builder(int(self.weekday) - 1, 7)
        day_vector = self.ohe_builder(int(self.day) - 1, 31)
        
        linear_input = np.concatenate((time_slot_vector, month_vector, weekday_vector, day_vector))
        
        return linear_input
        
    def build_sequence_data(self, lookback_frames):
        
        sequence = []
        
        open_close_window = []
        high_window = []
        low_window = []
        vol_window = []
        time_window = []
        
        for u in range(0, len(lookback_frames)):
            
            open_close_window.append(lookback_frames[u].close)
            high_window.append(lookback_frames[u].high)
            low_window.append(lookback_frames[u].low)
            vol_window.append(lookback_frames[u].volume)
            time_window.append(lookback_frames[u].ID)
    
        
        sequence = []
        for w in range(0, len(open_close_window)):
            sequence.append([open_close_window[w], high_window[w], low_window[w]])
           
        np.flip(np.array(sequence), 0)
        
        return sequence
    
    def ohe_builder(self, index, max_len):
        vector = [0] * max_len
        vector[int(index)] = 1
        return np.array(vector)
        
    def get_time_slot_index(self, time):
        
        time = datetime.strptime(time, '%H:%M')
        
        time = time.strftime("%H:%M")
        hours = time[0:2]
        minutes= time[3:5]
        
        hourindex = int(hours)
        minutes_index = int(minutes) / 5
        
        time_slot_index = hourindex + ((hourindex + 1) * minutes_index)
        
        return time_slot_index

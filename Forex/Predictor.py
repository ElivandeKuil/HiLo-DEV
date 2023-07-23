from TickerPredictor import TickerPredictor
import os
import pickle
from dateutil import parser
import numpy as np
import Globals
from datetime import datetime, timedelta
import time

class Predictor():
    
    def __init__(self, ticker_predictors_path):
        
        self.predictors_list = self.get_ticker_predictors(ticker_predictors_path)
        
    def get_ticker_predictors(self, path):
        
        ticker_predictors = []
        
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), "rb") as fp:   
                ticker_predictor = pickle.load(fp)
                ticker_predictors.append(ticker_predictor)
        return ticker_predictors
    
    def predict(self, data):
        
        Globals.device ='cpu'
        orders = []
        
        for tickerpredictor in self.predictors_list:
            if tickerpredictor.ticker + "_M15" in data.keys():
                ticker_data = data[tickerpredictor.ticker+ "_M15"]
                if len(ticker_data) == 8:
                    
                    formatted_data = self.format_data(ticker_data)
                    
                    if len(formatted_data) > 0:
                        
                        input_log_string = self.convert_formatted_data_to_string(formatted_data)
                        
                        if Globals.sys_log_mode > 2:
                            Globals.log_df.add_log(datetime.now(), "Predictor", "Predictor", "predict",
                                               "formatted data: \n" + input_log_string, tickerpredictor.ticker)
                        
                        formatted_data = formatted_data[0]
                        order = tickerpredictor.predict(time.time(), formatted_data)
                        if len(order) > 0:
                            orders.append(order[0])
                else:
                    print("retrieved data was not of the correct size for ticker: " + tickerpredictor.ticker + 
                          ", excpected 24, got: " + str(len(ticker_data)) )  
                    Globals.log_df.add_warning(datetime.now(), "Predictor", "Predictor", "predict", 
                                               "Retrieved data was not of correct size, expected 24 but got " + str(len(ticker_data)), tickerpredictor.ticker, 1)
        
        return orders

    def format_data(self, raw_data):
    
        data = list(raw_data.items())
    
        time_frames = []
        for point in data:
            time_frames.append(time_frame(point))
    
        processed_data= []
        current_day = time_frames[0].ID[0:10]
        
        for u in range(0, len(time_frames)):
            if time_frames[u].ID[0:10] == current_day:
                
                debug = time_frames[u].ID[0:10]
                
                if u == 7:
                    if int(time_frames[u].ID[11:13]) < 3 and datetime.strptime(time_frames[u].ID[0:10].replace('.', '/'), '%Y/%m/%d').isoweekday() == 1:
                        do = 'nothing'
                    else:
                        new_input = model_input(time_frames[u], time_frames)
                        processed_data.append(new_input)
            else:
                current_day = time_frames[u].ID[0:10]
            
        return processed_data

    def convert_formatted_data_to_string(self, formatted_data):
        sequence = formatted_data[0].sequence_input
        linear = formatted_data[0].linear_input
        
        linear_string = ' '.join([str(item) for item in linear])
        
        output_string = "linear: " + linear_string + '\n' + "sequence: \n"
        
        for row in sequence:
            
            output_string += ' '.join([str(item) for item in row]) + "\n"
            
        return output_string
                                 

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
            if tf.low <= -0.06:
                self.label = 1
        if Globals.label_mode == 'high':
            if tf.high >= 0.06:
                self.label = 1
            
    def build_linear_vector(self):
        
        time_slot_vector = self.ohe_builder(self.time_slot, 92)
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
        minutes_index = int(minutes) / 15
        
        time_slot_index = hourindex + minutes_index
        
        return time_slot_index
    
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
# display data on the MetaTrader 5 package
print("MetaTrader5 package author: ",mt5.__author__)
print("MetaTrader5 package version: ",mt5.__version__)
 
# import the 'pandas' module for displaying data obtained in the tabular form
import pandas as pd
pd.set_option('display.max_columns', 500) # number of columns to be displayed
pd.set_option('display.width', 1500)      # max table width to display
# import pytz module for working with time zone
import pytz
 
# establish connection to MetaTrader 5 terminal
if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()
 
TP = 0.05
SL = 0.05

def get_raw_data(symbol):
    print("Downloading data...")
    # set time zone to UTC
    timezone = pytz.timezone("Etc/UTC")
    # create 'datetime' object in UTC time zone to avoid the implementation of a local time zone offset
    utc_from = datetime(2020, 1, 10, tzinfo=timezone)
    # request 100 000 EURUSD ticks starting from 10.01.2019 in UTC time zone
    ticks = mt5.copy_ticks_from(symbol, utc_from, 100000000, mt5.COPY_TICKS_ALL)
    print("Ticks received:",len(ticks))
    
    # shut down connection to the MetaTrader 5 terminal
    mt5.shutdown()
    
    # create DataFrame out of the obtained data
    ticks_frame = pd.DataFrame(ticks)
    # convert time in seconds into the datetime format
    ticks_frame['time']=pd.to_datetime(ticks_frame['time'], unit='s')

    return ticks

def remove_duplicate_dates(raw_data):

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




class tick():
    
    def __init__(self, datum):
        
        self.ID = datum[0]
        self.bid = datum[1]
        self.ask = datum[2]
    
class model_input():
    def __init__(self, current_tick, prequel_ticks, sequel_ticks):
        
        self.coming_high = 0
        self.coming_low = 0
        self.label = 0

        for seq_tick in sequel_ticks:

            if datetime.fromtimestamp(seq_tick.ID) - timedelta(minutes=5) <= datetime.fromtimestamp(current_tick.ID):
                if self.get_percentage(seq_tick.bid, current_tick.bid) > self.coming_high:
                    self.coming_high = self.get_percentage(seq_tick.bid, current_tick.bid)
                else:
                    if self.get_percentage(seq_tick.bid, current_tick.bid) < self.coming_low:
                        self.coming_low = self.get_percentage(seq_tick.bid, current_tick.bid)
                
        self.ID = current_tick.ID
        self.month = datetime.fromtimestamp(current_tick.ID).month
        self.day = datetime.fromtimestamp(current_tick.ID).day
        self.weekday = datetime.fromtimestamp(current_tick.ID).weekday
        
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
            print("WARNING: saturday tick found")
        if self.weekday == 'sun':
            self.weekday = 6
            print("WARNING: sunday tick found")

        #self.linear_input = self.build_linear_vector()
        self.sequence_input = self.build_sequence_data(prequel_ticks)

        self.label = 0
        if Globals.label_mode == 'low':
            if self.coming_low <= -1 * float(TP) and self.coming_high <= float(SL):
                self.label = 1
        if Globals.label_mode == 'high':
            if self.coming_high >= float(TP) and self.coming_low >= -1 * float(SL):
                self.label = 1
            
    def build_linear_vector(self):
        month_vector = self.ohe_builder(int(self.month) - 1, 12)
        weekday_vector = self.ohe_builder(int(self.weekday) - 1, 7)
        day_vector = self.ohe_builder(int(self.day) - 1, 31)
        
        linear_input = np.concatenate((month_vector, weekday_vector, day_vector))
        
        return linear_input
        
    def build_sequence_data(self, prequel_ticks):
        
        sequence = []
        bid_window = []
        ask_window = []

        for u in range(1, len(prequel_ticks)):
            
            bid_window.append(self.get_percentage(prequel_ticks[u].bid, prequel_ticks[u - 1].bid))
            ask_window.append(self.get_percentage(prequel_ticks[u].ask, prequel_ticks[u - 1].ask))
    
        sequence = []
        for w in range(0, len(bid_window)):
            sequence.append([bid_window[w], ask_window[w]])
           
        np.flip(np.array(sequence), 0)
        
        return sequence
    
    def ohe_builder(self, index, max_len):
        vector = [0] * max_len
        vector[int(index)] = 1
        return np.array(vector)

    def get_percentage(self, new, old):
        return ((new - old)/old) * 100
    


def process_data(unique_dates_data):
    ticks = []
    print("Trimming data...")
    with alive_bar(len(unique_dates_data)) as bar:
        for point in unique_dates_data:
            bar()
            ticks.append(tick(point))

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
                    new_input = model_input(ticks[u], ticks[u - 120: u], ticks[u:u + 300])
                    if len(new_input.sequence_input) > 0:
                        processed_data.append(new_input)
            else:
                current_day = datetime.fromtimestamp(ticks[u].ID).day
        



raw_data = get_raw_data('USDJPY')
unique_dates_data = remove_duplicate_dates(raw_data)
processed_data = process_data(unique_dates_data)
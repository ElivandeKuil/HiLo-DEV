import os
import sys
from datetime import datetime
import numpy as np
import json
from time import sleep
from threading import Thread
from os.path import join, exists
from traceback import print_exc
from random import random
from datetime import datetime, timedelta
import pickle

from api.dwx_client import dwx_client

class tick_processor():

    def __init__(self, symbol, MT4_directory_path, 
                 sleep_delay=0.005,             # 5 ms for time.sleep()
                 max_retry_command_seconds=10,  # retry to send the commend for 10 seconds if not successful. 
                 verbose=True
                 ):

        self.symbol = symbol
    

        self.last_open_time = datetime.utcnow()
        self.last_modification_time = datetime.utcnow()

        self.dwx = dwx_client(self, MT4_directory_path, sleep_delay, 
                              max_retry_command_seconds, verbose=verbose)
        sleep(1)
        
        self.dwx.start()
        
        print("Account info:", self.dwx.account_info)
        
        print("Started scraping")
        
        now = datetime.now()
        
        # Timezone Name.
        
        start = "22/Jun/2020:23:15:26 UTC +0900"
        end = "26/Jun/2023:09:15:26 UTC +0900"
        
        start_dt_format = datetime.strptime(start, '%d/%b/%Y:%H:%M:%S %Z %z')
        end_dt_format = datetime.strptime(end, '%d/%b/%Y:%H:%M:%S %Z %z')
        
        # Timestamp
        start_timestamp = start_dt_format.timestamp()
        end_timestamp = end_dt_format.timestamp()
        
        self.dwx.get_historic_data(self.symbol, 'M15', start_timestamp, end_timestamp)
        
        

        
        
        
    def on_bar_data(self, symbol, time_frame, time, open_price, high, low, close_price, tick_volume):
        
        print('on_bar_data:', symbol, time_frame, datetime.utcnow(), time, open_price, high, low, close_price)

    
    def on_historic_data(self, symbol, time_frame, data):
        
        print('historic_data:', symbol, time_frame, f'{len(data)} bars')


    def on_historic_trades(self):

        print(f'historic_trades: {len(self.dwx.historic_trades)}')
    

    def on_message(self, message):
        
        if message['type'] == 'ERROR':
            print(message['type'], '|', message['error_type'], '|', message['description'])
        elif message['type'] == 'INFO':
            print(message['type'], '|', message['message'])


    # triggers when an order is added or removed, not when only modified. 
    def on_order_event(self):
        
        print(f'on_order_event. open_orders: {len(self.dwx.open_orders)} open orders')


def activate_dwx(symbol, stop_percentage):
    MT4_files_dir = 'C:/Users/eli_s/AppData/Roaming/MetaQuotes/Terminal/16D9C17040576AD13C62C316983027D5/MQL5/Files/'
    processor = tick_processor(symbol, MT4_files_dir)

    while (len(processor.dwx.historic_data) < 1):
        sleep(2)
        print("Loading...")
    
    
    print("Result:")
    result = processor.dwx.historic_data
    
    with open("C:/Users/eli_s/Documents/GitHub/HiLo-DEV/Forex/Data/Pickled/" + symbol + ".data", "wb") as fp: 
                pickle.dump(result, fp)
    
    
    dic = list(result.values())[0]
    dic_items = list(dic.items())
    
    double_count = 0
    high_over_stop_percentage = 0
    low_under_stop_percentage = 0
    non_volatile_count = 0
    
    current_day = dic_items[0][0][0:10]
    open_price = dic_items[0][1]['open']
    
    total_days = 0
    
    
    for tup in dic_items:
        
        if tup[0][:10] == current_day:
            open_price = (tup[1]['open'])
            high_percent = (tup[1]['high'] - open_price) / open_price * 100
            low_percent = (tup[1]['low'] - open_price) / open_price * 100
            
            if high_percent >= stop_percentage and low_percent <= (-1 * stop_percentage):
                double_count += 1
            else:
                if high_percent >= stop_percentage:
                    high_over_stop_percentage += 1
                else:
                    if low_percent <= (-1 * stop_percentage):
                        low_under_stop_percentage += 1
                    else:
                        non_volatile_count += 1
                
        else:
            
            total_days += 1
            
            current_day = tup[0][:10]
            open_price = tup[1]['open']
        
    
    total_intervals = double_count + high_over_stop_percentage + low_under_stop_percentage + non_volatile_count
    
    print("double percentage: " + str(double_count / total_intervals) )
    print("high hit percentage: " + str(high_over_stop_percentage / total_intervals) )
    print("low hit percentage: " + str(low_under_stop_percentage / total_intervals) )
    print("no hit percentage: " + str(non_volatile_count / total_intervals) )
    debug = 0


activate_dwx("EURSGD", .06)
















debug = 0

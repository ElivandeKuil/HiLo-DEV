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
import pandas as pd
from Order import Order
import time
import Globals
import random

from api.dwx_client import dwx_client

from Predictor import Predictor
from TickerPredictor import TickerPredictor

class tick_processor():

    def __init__(self, MT4_directory_path, 
                 sleep_delay=0.005,             # 5 ms for time.sleep()
                 max_retry_command_seconds=10,  # retry to send the commend for 10 seconds if not successful. 
                 verbose=True
                 ):

        self.last_open_time = datetime.utcnow()
        self.last_modification_time = datetime.utcnow()

        self.dwx = dwx_client(self, MT4_directory_path, sleep_delay, 
                              max_retry_command_seconds, verbose=verbose)
        sleep(1)
        
        self.dwx.start()
        
        self.verbose=False
        
        
        
        
        
    def on_bar_data(self, symbol, time_frame, time, open_price, high, low, close_price, tick_volume):
        
       if self.verbose == True: print('on_bar_data:', symbol, time_frame, datetime.utcnow(), time, open_price, high, low, close_price)

    
    def on_historic_data(self, symbol, time_frame, data):
        
        if self.verbose == True: print('historic_data:', symbol, time_frame, f'{len(data)} bars')


    def on_historic_trades(self):

        if self.verbose == True: print(f'historic_trades: {len(self.dwx.historic_trades)}')
    
    def on_tick(self, dummy, summy, lummy):
        do = 'nothing'
        
    def on_message(self, message):
        
        
        if message['type'] == 'ERROR':
            print(message['type'], '|', message['error_type'], '|', message['description'])
            Globals.log_df.add_error(datetime.now(), "dwx_client", "dwx_client", None, message['type'] + '|' + message['error_type'] + '|' + message['description'], None, "MT5-ERROR")
        elif message['type'] == 'INFO':
            if self.verbose == True: print(message['type'], '|', message['message'])


    # triggers when an order is added or removed, not when only modified. 
    def on_order_event(self):
        
        print(f'on_order_event. open_orders: {len(self.dwx.open_orders)} open orders')
        

class program():
    
    def simulate(self):
        MT4_files_dir = 'C:/Users/eli_s/AppData/Roaming/MetaQuotes/Terminal/16D9C17040576AD13C62C316983027D5/MQL5/Files/'
        self.processor = tick_processor(MT4_files_dir)
        self.predictor = Predictor(os.path.join(os.path.dirname(os.path.realpath(__file__)), "tickerpredictors"))
        sleep(3)
        
        for u in range(1, 124):
            for v in range(0, 4):
                open_market = True
                if open_market:
                    self.update_tp_status()
                    market_data, historic_data, account_info = self.get_data(Globals.tp_df.get_active_tickers_in_list(), [u, v])
                    orders = self.get_orders(historic_data, market_data)
                    if len(orders) > 0:
                        self.place_orders(orders, account_info, market_data)
        
        Globals.log_df.export_df("C:/Users/eli_s/Documents/GitHub/HiLo-DEV/Forex/analysis docs")                
        
                        
    def run(self, debug=False):
        MT4_files_dir = 'C:/Users/eli_s/AppData/Roaming/MetaQuotes/Terminal/16D9C17040576AD13C62C316983027D5/MQL5/Files/'
        self.processor = tick_processor(MT4_files_dir)
        self.predictor = Predictor(os.path.join(os.path.dirname(os.path.realpath(__file__)), "tickerpredictors"))
        sleep(3)
        
        open_market = False
        
        run = True
        
        while (run == True):
            
            # try:
             
                 now = datetime.now()
                 
                 now = now + timedelta(hours=1, minutes=0)
            
                 if now.minute % 15 == 14 and now.second == 50:
                     
                     print("Started cycle...")
                     
                     open_market = False
                     
                     try:
                     
                        open_market = self.get_market_open()
                          
                     except Exception as e:
                         
                         print("Something went wrong getting open symbols; ", e)
                         Globals.log_df.add_error(datetime.now(), "main", "program", "get_open_market", "Something went wrong extracting date information", None, e)
                         
                     try:
                         
                         if open_market:
                    
                             self.update_tp_status()
                         
                     except Exception as e:
                        
                         print("Something went wrong updating tp status; ", e)
                         Globals.log_df.add_error(datetime.now(), "main", "program", "update_tp_status", "Something went wrong updating the statrus of the ticker predictors", None, e)
                         
                     
                         
                     sleep(.5)
                 
                 if open_market:   
                 
                     if now.minute % 15 == 14 and now.second == 58:
                         
                         self.processor.dwx.close_all_orders()
                             
                     if now.minute % 15 == 0 and now.second == 1:
                         
                         try:
                         
                             market_data, historic_data, account_info = self.get_data(Globals.tp_df.get_active_tps_in_list())
                              
                         except Exception as e:
                             
                             print("Something went wrong getting data; ", e)
                             Globals.log_df.add_error(datetime.now(), "main", "program", "get_data", "Something went wrong extracting data", None, e)
                             market_data = {}
                             historic_data = {}
                         
                         try:
                             orders = self.get_orders(historic_data, market_data)
                                
                         except Exception as e:
                             print("Something went wrong getting order; ", e)
                             Globals.log_df.add_error(datetime.now(), "main", "program", "get_orders", "Something went wrong getting orders", None, e)
                             orders = []
                         if len(orders) > 0:
                             
                             try:
                                 self.place_orders(orders, account_info, market_data)
                                 
                             except Exception as e:
                                 print("Something went wrong placing orders; ", e)
                                 Globals.log_df.add_error(datetime.now(), "main", "program", "place_orders", "Something went wrong placing orders", None, e)
                            
                         sleep(.5)
                         
                         print("Ended cycle.")
                         
                     if now.minute % 5 == 1 and now.second == 0:
                         
                         open_market = False
                         
                         try:
                             self.log_closed_orders()
                         except Exception as e:
                             
                             print("Something went wrong logging closed orders; ", e)
                             Globals.log_df.add_error(datetime.now(), "main", "program", "log_closed_orders", "Something went wrong logging closed orders", None, e)
                         
                         sleep(.5)
         #    except Exception as e:
         #        print("A fatal error occurec in the running thread, starting up new cycle, Error: ", e)
         #        Globals.log_df.add_error(datetime.now(), "main", "program", "run", 
        #                                  "A fatal error occurec in the running thread, starting up new cycle", None, e)
             
                 sleep(.5)
    
    
    def update_tp_status(self):
        
        self.processor.dwx.subscribe_symbols([])
        
        all_symbols = Globals.tp_df.get_all_symbols_in_list()
        
        random.shuffle(all_symbols)
        
        self.processor.dwx.subscribe_symbols(all_symbols)
        
        self.processor.dwx.check_market_data_once()
        
        sleep(1)
        
        market_data = self.processor.dwx.market_data
        
        datatable = Globals.tp_df.df
        
        for key in market_data.keys():
            
            order_market_data = market_data[key]
            
            current_spread = self.get_spread(order_market_data['bid'], order_market_data['ask'])
            
            if current_spread > datatable.loc[datatable['Ticker'] == key]['Max_spread'].tolist()[0]:
                
                if datatable.loc[datatable['Ticker'] == key, 'Status'].tolist()[0] == 2:
                    
                    do = "nothing"
                    
                else:
                    do = 0 # Globals.tp_df.update_status_by_id(datatable[datatable['Ticker'] == key]['ID'], 2)
               
                    Globals.log_df.add_warning(datetime.now(), "main", "program", "update_tp_status", key + " was deactivated due to an excessivly high spread: " + str(current_spread), key, 1)
               
            else:
                
                if datatable.loc[datatable['Ticker'] == key, 'Status'].tolist()[0] == 1:
                    
                    do = "nothing"
                    
                else:
                    Globals.tp_df.update_status_by_id(datatable.loc[datatable['Ticker'] == key]['ID'], 1)
                    Globals.log_df.add_warning(datetime.now(), "main", "program", "update_tp_status", key + " was reactivated due to an acceptable spread: " + str(current_spread), key, 1)
               
        
    
    def get_market_open(self):
        
        stopwatchstart = time.time()
        
        next_frame = datetime.now() + timedelta(hours=2, minutes=5) 
        
        day_index = next_frame.weekday()
        
        stopwatchend = time.time()
        opensymboltime = stopwatchend - stopwatchstart
        
        if opensymboltime >= 1:
            Globals.log_df.add_warning(datetime.now(), "main", "program", "get_market_open", "Retrieving date info took over 1 seconds, namely " + str(opensymboltime), None, 2)

        if day_index < 7: #< 5:
            
            return True
        
        else:
            
            return False
        
    def get_lot_size(self, desired_margin_percent, leverage, current_value, equity):
        
        desired_margin = (equity / (desired_margin_percent)) * 100
        
        nominal_lot_size = (desired_margin * leverage) / current_value
        
        lot_size = nominal_lot_size / 100000
        
        return lot_size
    
    def place_orders(self, orders, account_info, market_data):
        
        stopwatchstart = time.time()
        
        datatable = Globals.tp_df.df
        
        
        random.shuffle(orders) # in case of multiple orders
        
        order = orders[0]
            
        bid = market_data[order.ticker]['bid']
        ask = market_data[order.ticker]['ask']
        
        lot = self.get_lot_size(125, account_info['leverage'] , market_data[order.ticker]['bid'], account_info['equity'])
        
        sl = 0
        tp = 0
        ordertype = ""
        
        if order.label_mode == 'high':
            sl = bid - (datatable.loc[datatable['Ticker'] == order.ticker, 'SL_percent'].tolist()[0] * bid)
            tp = bid + (datatable.loc[datatable['Ticker'] == order.ticker, 'TP_percent'].tolist()[0] * bid)
            ordertype = 'buy'
        if order.label_mode == 'low':
            sl = ask + (datatable.loc[datatable['Ticker'] == order.ticker, 'SL_percent'].tolist()[0] * ask)
            tp = ask - (datatable.loc[datatable['Ticker'] == order.ticker, 'TP_percent'].tolist()[0] * ask)
            ordertype = 'sell'
            
        seq_type= type(order.ID)
        magicID = seq_type().join(filter(seq_type.isdigit, order.ID))
        
        self.processor.dwx.open_order(order.ticker, ordertype, lot, 0, sl, tp, magicID, "open-order")
        
        order.status = 2
        
        Globals.order_df.add_new_order(magicID, order.creator, datetime.now(), order.ticker, ordertype, order.comment)
        
        if Globals.sys_log_mode > 0:
            Globals.log_df.add_log(datetime.now(), "main", "program", "place_orders", "Send (" + order.label_mode + ") order to MT5 for ticker: " +
                               order.ticker + "; current_bid= " + str(bid) + ", current_ask=" + str(ask) + ", current_spread=" + str(round(self.get_spread(bid, ask),4)) + "%, lot=" + str(lot) + 
                               ", sl=" + str(sl) + ", tp=" + str(tp) + ", magicID= " + str(magicID), order.ticker)
    
        stopwatchend = time.time()
        ordertime = stopwatchend - stopwatchstart
        
        if ordertime >= 1:
            Globals.log_df.add_warning(datetime.now(), "main", "program", "place_orders", "Placing orders took over 1 seconds, namely " + str(ordertime), order.ticker, 2)
    
    def log_closed_orders(self):
        """
        self.processor.dwx.get_historic_trades(10)
        sleep(1)
        historic_trades = self.processor.dwx.historic_trades
        historic_orders_df = Globals.order_df.df
        """
        debug = 0
        
    def get_data(self, open_symbols, time_delta=[0,0]):
        
        stopwatchstart = time.time()
        
        random.shuffle(open_symbols)
        """
        now = datetime.now() + timedelta(hours=3, minutes=0)
        
        start = now - timedelta(hours=2, minutes=5)
        end = now - timedelta(hours=0, minutes=0, seconds=10)
        
        
        start_timestamp = start.timestamp()
        end_timestamp = end.timestamp()
        
        
        """
        start = "17/Jul/2023:01:00:00 UTC +0100" 
        end = "17/Jul/2023:02:59:51 UTC +0100"
        
        start_dt_format = datetime.strptime(start, '%d/%b/%Y:%H:%M:%S %Z %z') + timedelta(hours=1 * time_delta[0], minutes=15 * time_delta[1])
        end_dt_format = datetime.strptime(end, '%d/%b/%Y:%H:%M:%S %Z %z') + timedelta(hours=1 * time_delta[0], minutes=15 * time_delta[1])
       
        start_timestamp = start_dt_format.timestamp()
        end_timestamp = end_dt_format.timestamp()
        
        
        self.processor.dwx.subscribe_symbols(open_symbols)
        
        for symbol in open_symbols:
            self.processor.dwx.get_historic_data(symbol, 'M15', start_timestamp, end_timestamp)
            sleep(.5)
        
        account_info = self.processor.dwx.account_info
        self.processor.dwx.check_market_data_once()
        
        sleep(2)
        
        market_data = self.processor.dwx.market_data
        
        historic_data = self.processor.dwx.historic_data
    
        ##Reset##
        self.processor.dwx.subscribe_symbols([])
        #########   
        
        stopwatchend = time.time()
        getdatatime = stopwatchend - stopwatchstart
        print("Data retrieval took " + str(getdatatime) + "s")
        
        if getdatatime >= 10:
            Globals.log_df.add_warning(datetime.now(), "main", "program", "get_data", "Retrieving data took over 10 seconds, namely " + str(getdatatime), None, 2)
    
        if len(open_symbols) != len(historic_data):
            
            print("Did not retrieve all symbol data, asked for " + str(len(open_symbols)) + ", and got " + str(len(historic_data)))
            Globals.log_df.add_warning(datetime.now(), "main", "program", "get_data", 
                                       "Did not retrieve all symbol data, asked for " + str(len(open_symbols)) + ", and got " + str(len(historic_data)), None, 1)
            
        if len(open_symbols) != len(market_data):
            
            print("Did not retrieve all market data, asked for " + str(len(open_symbols)) + ", and got " + str(len(market_data)))
            Globals.log_df.add_warning(datetime.now(), "main", "program", "get_data", 
                                       "Did not retrieve all market data, asked for " + str(len(open_symbols)) + ", and got " + str(len(market_data)), None, 2)
    
        return market_data, historic_data, account_info
    
    def get_spread(self, bid, ask):
        return ((ask - bid) / bid) * 100
    
    def get_orders(self, data, market_data):
        
        stopwatchstart = time.time()
        
        orders = self.predictor.predict(data)
        
        stopwatchend = time.time()
        predictiontime = stopwatchend - stopwatchstart
        print("Predictions took " + str(predictiontime) + "s")
        
        if predictiontime >= 3:
            Globals.log_df.add_warning(datetime.now(), "main", "program", "get_orders", "Getting orders took over 3 seconds, namely " + str(predictiontime), None, 2)
    
        if len(orders) == 0:
            print("no orders were made")
            
        return orders

prog = program()
prog.simulate()

debug = 0



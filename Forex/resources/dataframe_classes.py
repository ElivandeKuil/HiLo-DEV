import pandas as pd
import os
import pickle


class tp_df():
    
    def __init__(self, folder_path):
        
        self.path = folder_path
        if os.path.isfile(folder_path + "/tp.df"):
            
            with open(folder_path + "/tp.df", "rb") as fp:   
                self.df = pickle.load(fp)
            
        else:
            columns = ['ID', 'Ticker', 'Name', 'Max_spread', 'Aim_precision', 'Aim_AR', 'Status', 'TP_percent', 'SL_percent', 'Comment', 'Interval']
            self.df = pd.DataFrame(columns=columns)
    
    
    def add_row(self, ID, ticker, name, max_spread, aim_precision, aim_ar, status, tp_percent, sl_percent, interval, comment=""):
        new_row = {'ID':ID, 'Ticker':ticker, 'Name':name, 'Max_spread':max_spread, 'Aim_precision':aim_precision, 'Aim_AR':aim_ar,
                   'Status':status, 'TP_percent':tp_percent, 'SL_percent':sl_percent, 'Comment':comment, 'Interval':interval}
        self.df = self.df.append(new_row, ignore_index=True)
        self.save_df()
    
    def update_status_by_id(self, Id, status):
        Id = Id.tolist()[0]
        self.df.loc[self.df['ID'] == Id, 'Status'] = status
        self.save_df()
        
    def place_comment_by_id(self, ID, comment):
        self.df.loc[self.df['ID'] == ID,['Comment']] = self.df.loc[self.df['ID'] == ID,['Comment']] + "\n " + comment
        self.save_df()
    
    def get_active_tickers_in_list(self):
        sub_df = self.df.loc[self.df['Status'] == 1]
        return sub_df['Ticker'].tolist()
    
    def get_all_symbols_in_list(self):
        return self.df['Ticker'].tolist()
    
    def delete_row(self, index):
        self.df.drop([index])
        self.save_df()
        
    def save_df(self):
        with open(self.path + "/tp.df", "wb") as fp: 
            pickle.dump(self.df, fp)
    
    def export_df(self, folder_path):
        self.df.to_excel(folder_path + "/tp_df_export.xlsx")
            
class log_df():
    
    def __init__(self, folder_path):
        
        self.path = folder_path
        if os.path.isfile(folder_path + "/system_log.df"):
            
            with open(folder_path + "/system_log.df", "rb") as fp:   
                self.df = pickle.load(fp)
            
        else:
            columns = ['Date', 'File', 'Class', 'Method', 'Ticker', 'Threat', 'Message', 'Error']
            self.df = pd.DataFrame(columns=columns)
    
    
    def add_log(self, date, file, _class, method, message, ticker):
        new_row = {'Date':date, 'File':file, 'Class':_class, 'Method':method, 'Threat':0, 'Message':message, 'Ticker':ticker}
        self.df = self.df.append(new_row, ignore_index=True)
        self.save_df()
        
    def add_warning(self, date, file, _class, method, message, ticker, threat):
        new_row = {'Date':date, 'File':file, 'Class':_class, 'Method':method, 'Threat':threat, 'Message': "WARNING: " + message, 'Ticker':ticker}
        self.df = self.df.append(new_row, ignore_index=True)
        self.save_df()
    
    def add_error(self, date, file, _class, method, message, ticker, error):
        new_row = {'Date':date, 'File':file, 'Class':_class, 'Method':method, 'Threat':3, 'Message': "ERROR: " + message, 'Ticker':ticker, 'Error':error}
        self.df = self.df.append(new_row, ignore_index=True)
        self.save_df()
        
    def delete_row(self, index):
        self.df.drop([index])
        self.save_df()
        
    def save_df(self):
        with open(self.path + "/system_log.df", "wb") as fp: 
            pickle.dump(self.df, fp)
            
    def export_df(self, folder_path):
        self.df.to_excel(folder_path + "/log_df_export.xlsx")
            
class order_df():
    
    def __init__(self, folder_path):
        
        self.path = folder_path
        if os.path.isfile(folder_path + "/order.df"):
            
            with open(folder_path + "/order.df", "rb") as fp:   
                self.df = pickle.load(fp)
            
        else:
            columns = ['ID', 'Status', 'Creator', 'OpenDate', 'CloseDate' 'Ticker', 'LabelMode',
                       'PurchasePrice', 'SellPrice', 'ClosedPercentage', 'TotalWinLoss', 'Spread', 
                       'TotalSpreadCost', 'MadeProfit', 'Comment']
            
            
            self.df = pd.DataFrame(columns=columns)
    
    
    def add_new_order(self, Id, creator, date, ticker, labelmode, comment=""):
        new_row = {'ID':Id, 'Status':1, 'Creator':creator, 'OpenDate':date, 'Ticker':ticker, 'Labelmode':labelmode, 'Comment':comment}
        self.df = self.df.append(new_row, ignore_index=True)
        self.save_df()
    
    def place_order(self, Id, purchaseprice):
        self.df.loc[self.df['ID'] == Id,['Status']] = [2]
        self.df.loc[self.df['ID'] == Id,['PurchasePrice']] = purchaseprice
        self.save_df()
    
    def close_order(self, Id, closedate, sellprice, closedpercentage, totalwinloss, spread, totalspreadcost, madeprofit):
        self.df.loc[self.df['ID'] == Id,['Status', 'CloseDate', 'SellPrice','ClosedPercentage', 'TotalWinLoss', 'Spread', 'TotalSpreadCost', 'MadeProfit']] = [3, closedate, sellprice, closedpercentage, totalwinloss, spread, totalspreadcost, madeprofit]
        self.save_df()
        
    def delete_row(self, index):
        self.df.drop([index])
        
    def save_df(self):
        with open(self.path + "/order.df", "wb") as fp: 
            pickle.dump(self.df, fp)
            
    def export_df(self, folder_path):
        self.df.to_excel(folder_path + "/order_df_export.xlsx")



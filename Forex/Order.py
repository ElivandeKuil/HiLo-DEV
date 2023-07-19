
class Order():
    
    def __init__(self, ID, ticker_predictor, label_mode, ticker, time, creator):
        
        self.ID = ID
        self.ticker_predictor = ticker_predictor
        self.label_mode = label_mode
        self.ticker = ticker
        self.time = time
        self.creator = creator
        self.status = 1  # 1 = Created, 2 = Placed, 3 = Succesfully closed, 4 = Closed due to Error
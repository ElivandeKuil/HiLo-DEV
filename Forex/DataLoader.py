from Miner import Dataset
import random
import numpy as np
import Globals 
import math

class AE_test_dataloader():

    def __init__(self, batch_size, total_chunks, look_back, look_ahead, indices):
        self.ds = Dataset()
        self.ds.load_data_chunks(total_chunks, look_back, look_ahead)
        self.indices = indices
        self.current_chunk = []
        self.batch_size = batch_size
        self.total_chunks = total_chunks
        self.look_ahead = look_ahead
        self.look_back = look_back

    def set_next_chunk(self):

        if len(self.indices) > 0:
            print("Setting next chunk (index = " + str(self.indices[-1]) + ")")
            _, _, self.current_chunk = self.ds.get_data_chunk_by_index(Globals.look_back ,Globals.look_ahead, Globals.TP, Globals.SL ,self.indices[-1], 'test')
            
            random.shuffle(self.current_chunk)

            self.indices.remove(self.indices[-1])
            return True
        else:
            return False
        
        
    def get_batch(self):

        if len(self.current_chunk) < self.batch_size:
            succes = self.set_next_chunk()
            if succes == False:
                return [] 
            
        batch = self.current_chunk[-self.batch_size:]
        del self.current_chunk[-self.batch_size:]

        data = []
        labels = []

        for sample in batch:
            
            data.append(np.array(sample.sequence_input).astype(float))
            labels.append(sample.label)

        return data, labels


class AE_train_dataloader():

    def __init__(self, batch_size, num_chunks, total_chunks, look_back, look_ahead, train_or_val, indices):
        self.ds = Dataset()
        self.ds.load_data_chunks(num_chunks, look_back, look_ahead)
        self.indices = indices
        self.current_chunk = []
        self.batch_size = batch_size
        self.num_chunks = num_chunks
        self.total_chunks = total_chunks
        self.look_ahead = look_ahead
        self.look_back = look_back
        self.train_or_val = train_or_val

    def remove_rare_event(self, data):

        clean_data = []
        for u in range(0, len(data)):
            if data[u].label == 0:
                clean_data.append(data[u])

        return clean_data

    def reset(self):
        self = AE_train_dataloader(self.batch_size, self.num_chunks, self.total_chunks, self.look_back, self.look_ahead)

    def set_next_chunk(self):
        if len(self.indices) > 0:
            print("Setting next chunk (index = " + str(self.indices[-1]) + ")")
            if (self.train_or_val == 0):
                self.current_chunk, _, _ = self.ds.get_data_chunk_by_index(Globals.look_back ,Globals.look_ahead, Globals.TP, Globals.SL ,self.indices[-1], 'train')
            else:
                _, self.current_chunk, _ = self.ds.get_data_chunk_by_index(Globals.look_back ,Globals.look_ahead, Globals.TP, Globals.SL ,self.indices[-1], 'val')
            random.shuffle(self.current_chunk)

            self.indices.remove(self.indices[-1])

            self.current_chunk = self.remove_rare_event(self.current_chunk)
            return True
        else:
            return False
        
    def get_batch(self):

        if len(self.current_chunk) < self.batch_size:
            succes = self.set_next_chunk()
            if succes == False:
                return [] 
            
        batch = self.current_chunk[-self.batch_size:]
        del self.current_chunk[-self.batch_size:]

        data = []

        for sample in batch:
            
            data.append(np.array(sample.sequence_input).astype(float))

        return data
    

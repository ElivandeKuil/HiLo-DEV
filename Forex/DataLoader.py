from Miner import Dataset
import random
import numpy as np

class AE_test_dataloader():

    def __init__(self, num_chunks, total_chunks, look_back, look_ahead):
        self.ds = Dataset()
        self.ds.load_data_chunks(num_chunks, look_back, look_ahead)
        self.indices = self.get_random_indices(num_chunks, total_chunks)
        self.current_chunk = None

    def get_random_indices(self, num_chunks, total_chunks):

        random_indices = []
        while len(random_indices < num_chunks):
            random_int = random.randint(0, total_chunks)
            if random_int in random_indices:
                do = 'nothing'
            else:
                random_indices.append(random_int)

        return random_indices
    
    def set_next_chunk(self):

        if len(self.indices > 0):
            _, _, self.current_chunk = self.ds.get_data_chunk_by_index(self.indices[-1])
            random.shuffle(self.current_chunk)
            self.indices.remove(self.indices[-1])
            return True
        else:
            return False

    
    def get_batch(self, batch_size):

        if len(self.current_chunk) < batch_size:
            succes = self.set_next_chunk()
            if succes == False:
                return [], []    
        
        batch = self.current_chunk[-batch_size:]
        del self.current_chunk[-batch_size:]

        data = []
        labels = []

        for sample in batch:
            
            data.append(np.array(sample.sequence_input).astype(float))
            labels.append(sample.label)

        return data, labels



class AE_train_dataloader():

    def __init__(self, num_chunks, total_chunks, look_back, look_ahead):
        self.ds = Dataset()
        self.ds.load_data_chunks(num_chunks, look_back, look_ahead)
        self.indices = self.get_random_indices(num_chunks, total_chunks)
        self.current_chunk_train = []
        self.current_chunk_val = []

    def get_random_indices(self, num_chunks, total_chunks):

        random_indices = []
        while len(random_indices < num_chunks):
            random_int = random.randint(0, total_chunks)
            if random_int in random_indices:
                do = 'nothing'
            else:
                random_indices.append(random_int)

        return random_indices
    
    def remove_rare_event(data):

        clean_data = []
        for u in range(0, len(data)):
            if data[u].label == 0:
                clean_data.append(data[u])
        return clean_data


    def set_next_chunk(self):
        if len(self.indices > 0):
            print("Setting next chunk")
            self.current_chunk_train, self.current_chunk_val, _ = self.ds.get_data_chunk_by_index(self.indices[-1])

            random.shuffle(self.current_chunk_train)
            random.shuffle(self.current_chunk_val)
            self.indices.remove(self.indices[-1])

            self.current_chunk_train = self.remove_rare_event(self.current_chunk_train)
            self.current_chunk_val = self.remove_rare_event(self.current_chunk_val)
            return True
        else:
            return False
        
    
    def get_train_batch(self, batch_size):

        if len(self.current_chunk_train) < batch_size:
            succes = self.set_next_chunk()
            if succes == False:
                return [] 
            
        batch = self.current_chunk_train[-batch_size:]
        del self.current_chunk_train[-batch_size:]

        data = []

        for sample in batch:
            
            data.append(np.array(sample.sequence_input).astype(float))

        return data
    
    def get_val_batch(self, batch_size):

        if len(self.current_chunk_val) < batch_size:
            self.set_next_chunk()
        
        batch = self.current_chunk_val[-batch_size:]
        del self.current_chunk_val[-batch_size:]

        data = []

        for sample in batch:
            
            data.append(np.array(sample.sequence_input).astype(float))

        return data

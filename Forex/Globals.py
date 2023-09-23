from resources.dataframe_classes import tp_df, log_df, order_df
from Logger import Logger
import os

logger = Logger(True)

ticker = ""
flush_temp = False
device = ''
batch_size = 0
learning_rate = 0
weight_decay = 0
num_epochs = 0
normalization = False

hidden_layer_size = 0
number_layers = 0
sequence_length = 0
sequence_input_size = 3

balance_data = False
balance_percent = 0
val_split = 0
test_split = 0
sys_log_mode = 3  # 3 = all
fine_tuning = False

label_mode = 'high' # or 'low'

df_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")

tp_df = tp_df(df_folder_path)
log_df = log_df(df_folder_path)
order_df = order_df(df_folder_path)

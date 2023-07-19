from prod_test_resources.dataframe_classes import tp_df, log_df, order_df
from Logger import Logger

logger = Logger(True)

flush_temp = False
device = ''
batch_size = 0
learning_rate = 0
weight_decay = 0
num_epochs = 0
hidden_layer_size = 0
number_layers = 0
balance_data = False
val_split = 0
test_split = 0

label_mode = 'high' # or 'low'

tp_df = tp_df("C:/Users/eli_s/Documents/GitHub/Project S V6/Forex/prod_test_resources")
log_df = log_df("C:/Users/eli_s/Documents/GitHub/Project S V6/Forex/prod_test_resources")
order_df = order_df("C:/Users/eli_s/Documents/GitHub/Project S V6/Forex/prod_test_resources")

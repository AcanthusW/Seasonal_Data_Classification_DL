# my_config.py

dataset_path = 'data/dataset.csv'


window_size = 5 
lstm_units = 16
cnn_filters = 32 
dropout_rate = 0.01
attention_units = 16


epochs = 200
batch_size = 1000
test_size = 0.2 
shuffle_data = False  # Whether to shuffle data during train-test split


learning_rate = 0.001


class_weight_0 = 1.0
class_weight_1 = 2.0  # Adjust as needed to give more weight to positive class


model_save_path = 'outputs/cnn_lstm_attention_model.keras'


early_stopping_patience = 10


seed = 42
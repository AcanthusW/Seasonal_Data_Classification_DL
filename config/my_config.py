# my_config.py

# Configuration for dataset paths and model training

# Dataset settings
dataset_path = 'path/to/your/merged_data.csv'  # Path to the dataset

# Model parameters
window_size = 5  # Time window size for LSTM input
lstm_units = 16  # Number of LSTM units
cnn_filters = 32  # Number of CNN filters
dropout_rate = 0.01  # Dropout rate to avoid overfitting
attention_units = 16  # Units for the Attention layer

# Training parameters
epochs = 200  # Number of training epochs
batch_size = 1000  # Batch size for training
test_size = 0.2  # Test split ratio
shuffle_data = False  # Whether to shuffle data during train-test split

# Optimizer and learning rate
learning_rate = 0.001  # Learning rate for the optimizer

# Class weights (for handling imbalance in binary classification)
class_weight_0 = 1.0
class_weight_1 = 2.0  # Adjust as needed to give more weight to positive class

# Output settings
model_save_path = 'outputs/cnn_lstm_attention_model.keras'  # Path to save the trained model

# Callbacks
early_stopping_patience = 10  # Number of epochs with no improvement after which training will be stopped

# Miscellaneous
seed = 42  # Random seed for reproducibility
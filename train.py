import os
import datetime
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from model import CNNLSTMAttentionModel
from config.my_config import *

# Load
df = pd.read_csv(dataset_path)
min_max_scaler = preprocessing.MinMaxScaler()
df_scaled = min_max_scaler.fit_transform(df)
df = pd.DataFrame(df_scaled, columns=df.columns)

# Prepare
seq_len = window_size
data = df.values
sequence_length = seq_len + 1
result = []

for index in range(len(data) - sequence_length):
    result.append(data[index: index + sequence_length])
result = np.array(result)


X = result[:, :-1]
y = result[:, -1][:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle_data, random_state=seed)

# training
model_wrapper = CNNLSTMAttentionModel(window_size=window_size, amount_of_features=X.shape[2])
model = model_wrapper.get_model()

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# save
output_folder = 'outputs'
os.makedirs(output_folder, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_path = os.path.join(output_folder, f'cnn_lstm_attention_model_{timestamp}.keras')
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
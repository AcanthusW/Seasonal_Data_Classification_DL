import os
import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from config.my_config import *

# Load and preprocess dataset
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

_, X_test, _, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle_data, random_state=seed)


output_folder = 'outputs'
latest_model = sorted([f for f in os.listdir(output_folder) if f.endswith('.keras')])[-1]
model_path = os.path.join(output_folder, latest_model)
model = load_model(model_path)
print(f"Loaded model from {model_path}")


loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')


y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Save evaluation results
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
evaluation_save_path = os.path.join(output_folder, f'evaluation_results_{timestamp}.txt')
with open(evaluation_save_path, 'w') as f:
    f.write(f'Test Loss: {loss:.4f}\n')
    f.write(f'Test Accuracy: {accuracy:.4f}\n\n')
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(cm))
print(f"Evaluation results saved to {evaluation_save_path}")
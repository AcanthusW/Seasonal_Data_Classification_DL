import tensorflow as tf
from tensorflow.keras import layers, models, Input

class CNNLSTMAttentionModel:
    def __init__(self, window_size, amount_of_features, lstm_units=16, cnn_filters=32, attention_units=16, dropout_rate=0.01):
        self.window_size = window_size
        self.amount_of_features = amount_of_features
        self.lstm_units = lstm_units
        self.cnn_filters = cnn_filters
        self.attention_units = attention_units
        self.dropout_rate = dropout_rate
        self.model = self._build_model()

    def _build_model(self):
        input_layer = Input(shape=(self.window_size, self.amount_of_features))

        # CNN 
        x = layers.Conv1D(filters=self.cnn_filters, kernel_size=3, padding='same', activation='relu')(input_layer)
        
        # LSTM
        x = layers.LSTM(units=self.lstm_units, return_sequences=True)(x)
        
        # Attention Mechanism
        attention = layers.Dense(self.attention_units, activation='tanh')(x)
        attention = layers.Dense(1, activation='softmax')(attention)
        attention = layers.Flatten()(attention)
        attention = layers.RepeatVector(self.lstm_units)(attention)
        attention = layers.Permute([2, 1])(attention)
        x = layers.multiply([x, attention])

        x = layers.Flatten()(x)
        
        # Dropout layer to avoid overfitting
        x = layers.Dropout(rate=self.dropout_rate)(x)
        
        output_layer = layers.Dense(1, activation='sigmoid')(x)
        model = models.Model(inputs=input_layer, outputs=output_layer)
        
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        
        return model

    def get_model(self):
        return self.model
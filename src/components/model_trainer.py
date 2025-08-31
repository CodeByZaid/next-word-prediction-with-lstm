import os
import sys
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from src.logger import logging
from src.exception import CustomException


class ModelTrainer:
    def __init__(self, model_dir: str = "artifacts/models"):
        
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "next_word_lstm.h5")

    def build_model(self, vocab_size, max_seq_len):
        """
        Build LSTM model for next word prediction.
        """
        model = Sequential()
        model.add(Embedding(vocab_size, 128, input_length=max_seq_len-1))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(256))
        model.add(Dense(vocab_size, activation="softmax"))

        model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
        return model

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test, tokenizer, max_seq_len, vocab_size):
        """
        Train the LSTM model, save it, and return trained model.
        """
        try:
            logging.info("Starting model training...")

            # 1. Build model
            model = self.build_model(vocab_size, max_seq_len)

            # 2. Define callbacks
            early_stopping = EarlyStopping(monitor="val_loss", patience=60, restore_best_weights=True)
            checkpoint = ModelCheckpoint(self.model_path, save_best_only=True, monitor="val_loss")

            # 3. Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=60,            
                callbacks=[early_stopping, checkpoint],
                verbose=1
            )

            logging.info(f"Model training completed. Model saved at {self.model_path}")

            return model

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)

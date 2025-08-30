import os
import sys
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle

from src.logger import logging
from src.exception import CustomException


class DataTransformation:
    def __init__(self, processed_dir: str = "artifacts/processed_data"):
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

    def initiate_data_transformation(self, corpus_path: str):
        """
        Reads corpus, tokenizes text, generates n-gram sequences,
        splits into train/test sets, and saves tokenizer & arrays.
        """
        try:
            logging.info("Starting data transformation...")

            # 1. Load corpus
            with open(corpus_path, "r") as file:
                text = file.read().lower()

            # 2. Tokenization
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts([text])
            total_words = len(tokenizer.word_index) + 1

            # Save tokenizer
            tokenizer_path = os.path.join(self.processed_dir, "tokenizer.pickle")
            with open(tokenizer_path, "wb") as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # 3. Generate input sequences
            input_sequences = []
            for line in text.split("\n"):
                token_list = tokenizer.texts_to_sequences([line])[0]
                for i in range(1, len(token_list)):
                    n_gram_sequence = token_list[:i+1]
                    input_sequences.append(n_gram_sequence)

            # 4. Pad sequences
            max_sequence_len = max([len(x) for x in input_sequences])
            input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre"))

            # 5. Split into X (predictors) and y (labels)
            X, y = input_sequences[:, :-1], input_sequences[:, -1]

            # One-hot encode y
            y = to_categorical(y, num_classes=total_words)

            # 6. Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Save arrays
            np.save(os.path.join(self.processed_dir, "X_train.npy"), X_train)
            np.save(os.path.join(self.processed_dir, "X_test.npy"), X_test)
            np.save(os.path.join(self.processed_dir, "y_train.npy"), y_train)
            np.save(os.path.join(self.processed_dir, "y_test.npy"), y_test)

            logging.info("Data transformation completed successfully")

            return X_train, X_test, y_train, y_test, tokenizer, max_sequence_len, total_words

        except Exception as e:
            logging.error("Error in data transformation step")
            raise CustomException(e, sys)

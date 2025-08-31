import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


MODEL_PATH = "artifacts/models/next_word_lstm.h5"
TOKENIZER_PATH = "artifacts/processed_data/tokenizer.pickle"

print(" Loading model and tokenizer...")
model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)

# Build reverse lookup (index → word)
index_word = {idx: word for word, idx in tokenizer.word_index.items()}



def predict_next_word(model, tokenizer, text, max_sequence_len, top_k=5):
    
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Ensure sequence length matches training setup
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding="pre")

    predictions = model.predict(token_list, verbose=0)[0]

    # Get top-k predictions
    predicted_indices = predictions.argsort()[-top_k:][::-1]
    predicted_words = [(index_word.get(idx, ""), predictions[idx]) for idx in predicted_indices]

    return predicted_words



📘 Next Word Prediction using LSTM (Deep Learning)

📖 Project Overview

This project implements a Next Word Prediction model using LSTM (Long Short-Term Memory) neural networks.
It is trained on Shakespeare’s Hamlet text and deployed as a Streamlit web application, where users can input a sequence of words and get the predicted next word in real-time.
The model also uses Early Stopping to prevent overfitting during training.

next-word-prediction-lstm/
│── Data/                  # raw / processed text files (hamlet.txt, etc.)
│── notebook/              # Jupyter notebooks for experiments
│   └── experiemnts.ipynb
│── src/
│   ├── components/        # Core modules
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/          # Training and prediction pipelines
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   └── __init__.py
│── artifacts/             # saved models, tokenizer, logs (gitignored)
│── app.py                 # Streamlit app for predictions
│── requirements.txt       # Python dependencies
│── README.md
│── .gitignore

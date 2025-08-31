from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    print(" Starting Training Pipeline...")

    # 1. Data Ingestion
    ingestion = DataIngestion()
    corpus_path = ingestion.initiate_data_ingestion()
    print(f" Corpus file saved at: {corpus_path}")

    # 2. Data Transformation
    transformation = DataTransformation()
    X_train, X_test, y_train, y_test, tokenizer, max_seq_len, vocab_size = (
        transformation.initiate_data_transformation(corpus_path)
    )
    print(" Data Transformation Completed")
    print(f" X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f" X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"Max sequence length: {max_seq_len},  Vocabulary size: {vocab_size}")

    # 3. Model Training
    trainer = ModelTrainer()
    model = trainer.initiate_model_trainer(X_train, X_test, y_train, y_test, tokenizer, max_seq_len, vocab_size)
    print(" Model Training Completed")
    print(" Training Pipeline Finished Successfully")

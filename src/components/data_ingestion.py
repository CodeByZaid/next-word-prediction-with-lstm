import os
import sys
import nltk
from nltk.corpus import gutenberg
from src.logger import logging
from src.exception import CustomException

class DataIngestion:
    def __init__(self, output_dir: str = "artifacts/raw_data"):
        
        self.output_dir = output_dir
        self.output_file = os.path.join(self.output_dir, "hamlet.txt")

    def initiate_data_ingestion(self) -> str:
        
        logging.info("Starting data ingestion process...")

        try:
            # Download gutenberg dataset if not already present
            nltk.download('gutenberg')

            # Load Hamlet text
            data = gutenberg.raw('shakespeare-hamlet.txt')
            logging.info("Hamlet dataset loaded successfully from Gutenberg corpus")

            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)

            # Save the dataset as hamlet.txt
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(data.lower())  # lowercasing for consistency

            logging.info(f"Data ingestion completed. Corpus saved at {self.output_file}")
            return self.output_file

        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    try:
        obj = DataIngestion()
        output_path = obj.initiate_data_ingestion()
        print(f"Hamlet dataset saved at: {output_path}")
    except Exception as e:
        print(f"Error: {e}")

import logging
import os

import pandas as pd


def load_data(file_path, logger):
    """Load and validate the comment dataset."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file {file_path} not found")
        df = pd.read_csv(file_path)
        required_columns = ['timestamp', 'media_id', 'media_caption', 'comment_text']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns in dataset")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logger.info("Data loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

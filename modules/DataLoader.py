# modules/DataLoader.py

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple

class DataLoader:
    """
    Loads your data.csv files and turns text into
    numeric features using TF-IDF with N-gram features.
    """

    def __init__(
        self,
        filepath: str,
        text_column: str = "question_body",
        label_column: str = "specialty_id",
        ngram_range: Tuple[int, int] = (1, 2), 
    ):
        self.filepath = filepath
        self.text_column = text_column
        self.label_column = label_column
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            preprocessor=self.preprocess_text
        )
        self.feature_names = None

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by:
        1. Converting to lowercase
        2. Removing special characters
        3. Removing extra whitespace
        4. Basic normalization
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep word boundaries
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def load(self) -> pd.DataFrame:
        """Load and clean the dataset."""
        df = pd.read_csv(self.filepath)
        df = df.dropna(subset=[self.text_column, self.label_column])
        return df

    def embed(self, texts: List[str]):
        """
        Fit (or transform) the vectorizer on your text data.
        Returns a sparse feature matrix using TF-IDF.
        """
        features = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return features

    def get_features_and_labels(self) -> Tuple:
        """Get processed features and labels."""
        df = self.load()
        X = self.embed(df[self.text_column].tolist())
        y = df[self.label_column].values
        return X, y

    def get_feature_names(self) -> List[str]:
        """Return list of feature names (n-grams)."""
        if self.feature_names is None:
            raise ValueError("You must call get_features_and_labels() first")
        return self.feature_names

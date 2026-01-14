"""Traditional ML baseline models for chemical property prediction."""

from typing import List, Dict, Union, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score
import sys
from pathlib import Path

# Add parent directory to path to allow imports when models module is imported from scripts
if __name__ != '__main__':
    parent = Path(__file__).parent.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

from utils.smiles_utils import (
    extract_all_ngrams,
    build_ngram_vocabulary,
    smiles_to_ngram_vector
)


class BaselineModel:
    """Base class for baseline models."""
    
    def __init__(self, task_type: str = 'classification'):
        """
        Initialize baseline model.
        
        Args:
            task_type: 'classification' or 'regression'
        """
        self.task_type = task_type
        self.model = None
        self.ngram_vocab = None
        self.max_n = 3
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model.
        
        Args:
            X: Feature matrix
            y: Target labels or values
        """
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True labels or values
            
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X)
        
        if self.task_type == 'classification':
            return {
                'accuracy': accuracy_score(y, predictions),
                'f1_score': f1_score(y, predictions, average='weighted')
            }
        else:
            return {
                'mse': mean_squared_error(y, predictions),
                'rmse': np.sqrt(mean_squared_error(y, predictions)),
                'r2': r2_score(y, predictions)
            }


class LogisticRegressionBaseline(BaselineModel):
    """Logistic Regression baseline."""
    
    def __init__(self, **kwargs):
        """
        Initialize Logistic Regression model.
        
        Args:
            **kwargs: Arguments for LogisticRegression
        """
        super().__init__(task_type='classification')
        self.model = LogisticRegression(max_iter=1000, **kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Logistic Regression model."""
        self.model.fit(X, y)
        return self


class SVMClassifierBaseline(BaselineModel):
    """SVM Classifier baseline."""
    
    def __init__(self, kernel: str = 'rbf', **kwargs):
        """
        Initialize SVM Classifier.
        
        Args:
            kernel: Kernel type ('linear', 'rbf', 'poly')
            **kwargs: Arguments for SVC
        """
        super().__init__(task_type='classification')
        self.model = SVC(kernel=kernel, **kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit SVM Classifier."""
        self.model.fit(X, y)
        return self


class SVRBaseline(BaselineModel):
    """Support Vector Regression baseline."""
    
    def __init__(self, kernel: str = 'rbf', **kwargs):
        """
        Initialize SVR.
        
        Args:
            kernel: Kernel type ('linear', 'rbf', 'poly')
            **kwargs: Arguments for SVR
        """
        super().__init__(task_type='regression')
        self.model = SVR(kernel=kernel, **kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit SVR."""
        self.model.fit(X, y)
        return self


class RandomForestBaseline(BaselineModel):
    """Random Forest baseline."""
    
    def __init__(self, task_type: str = 'classification', **kwargs):
        """
        Initialize Random Forest.
        
        Args:
            task_type: 'classification' or 'regression'
            **kwargs: Arguments for RandomForestClassifier/Regressor
        """
        super().__init__(task_type=task_type)
        
        if task_type == 'classification':
            self.model = RandomForestClassifier(**kwargs)
        else:
            self.model = RandomForestRegressor(**kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Random Forest."""
        self.model.fit(X, y)
        return self


class NGramFeatureExtractor:
    """N-gram feature extractor for SMILES."""
    
    def __init__(self, max_n: int = 3, min_freq: int = 1):
        """
        Initialize n-gram feature extractor.
        
        Args:
            max_n: Maximum n-gram size
            min_freq: Minimum frequency for n-gram to be included
        """
        self.max_n = max_n
        self.min_freq = min_freq
        self.vocab = None
    
    def fit(self, smiles_list: List[str]):
        """
        Build n-gram vocabulary from SMILES list.
        
        Args:
            smiles_list: List of SMILES strings
        """
        self.vocab = build_ngram_vocabulary(
            smiles_list, 
            max_n=self.max_n, 
            min_freq=self.min_freq
        )
        return self
    
    def transform(self, smiles_list: List[str]) -> np.ndarray:
        """
        Transform SMILES to n-gram feature vectors.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Feature matrix
        """
        if self.vocab is None:
            raise ValueError("Must call fit() before transform()")
        
        X = np.zeros((len(smiles_list), len(self.vocab)))
        for i, smiles in enumerate(smiles_list):
            X[i] = smiles_to_ngram_vector(smiles, self.vocab, max_n=self.max_n)
        
        return X
    
    def fit_transform(self, smiles_list: List[str]) -> np.ndarray:
        """
        Fit and transform SMILES to feature vectors.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Feature matrix
        """
        self.fit(smiles_list)
        return self.transform(smiles_list)


class NGramMLBaseline:
    """N-gram based ML baseline combining feature extraction and model."""
    
    def __init__(self, model_type: str = 'svm', task_type: str = 'classification',
                 max_n: int = 3, min_freq: int = 1, **model_kwargs):
        """
        Initialize N-gram ML baseline.
        
        Args:
            model_type: Type of model ('svm', 'svr', 'logistic', 'random_forest')
            task_type: 'classification' or 'regression'
            max_n: Maximum n-gram size
            min_freq: Minimum frequency for n-gram
            **model_kwargs: Additional arguments for the model
        """
        self.model_type = model_type
        self.task_type = task_type
        self.feature_extractor = NGramFeatureExtractor(max_n=max_n, min_freq=min_freq)
        
        # Initialize model
        if model_type == 'svm':
            self.model = SVMClassifierBaseline(**model_kwargs)
        elif model_type == 'svr':
            self.model = SVRBaseline(**model_kwargs)
        elif model_type == 'logistic':
            self.model = LogisticRegressionBaseline(**model_kwargs)
        elif model_type == 'random_forest':
            self.model = RandomForestBaseline(task_type=task_type, **model_kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, smiles_list: List[str], y: np.ndarray):
        """
        Fit the model on SMILES data.
        
        Args:
            smiles_list: List of SMILES strings
            y: Target labels or values
        """
        X = self.feature_extractor.fit_transform(smiles_list)
        self.model.fit(X, y)
        return self
    
    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """
        Make predictions on SMILES data.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Predictions
        """
        X = self.feature_extractor.transform(smiles_list)
        return self.model.predict(X)
    
    def evaluate(self, smiles_list: List[str], y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on SMILES data.
        
        Args:
            smiles_list: List of SMILES strings
            y: True labels or values
            
        Returns:
            Dictionary of metrics
        """
        X = self.feature_extractor.transform(smiles_list)
        return self.model.evaluate(X, y)


def get_baseline_model(model_type: str, task_type: str = 'classification', **kwargs):
    """
    Factory function to get baseline model.
    
    Args:
        model_type: Type of model
        task_type: 'classification' or 'regression'
        **kwargs: Additional arguments for the model
        
    Returns:
        Baseline model instance
    """
    if model_type == 'logistic':
        return LogisticRegressionBaseline(**kwargs)
    elif model_type == 'svm':
        return SVMClassifierBaseline(**kwargs)
    elif model_type == 'svr':
        return SVRBaseline(**kwargs)
    elif model_type == 'random_forest':
        return RandomForestBaseline(task_type=task_type, **kwargs)
    elif model_type in ['ngram-svm', 'ngram-svr']:
        ml_type = 'svm' if 'svm' in model_type else 'svr'
        return NGramMLBaseline(model_type=ml_type, task_type=task_type, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

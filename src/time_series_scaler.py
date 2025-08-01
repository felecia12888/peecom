__all__ = ["TimeSeriesScaler"]

import numpy as np

class TimeSeriesScaler:
    def fit_transform(self, X):
        self.mean = np.mean(X, axis=(0, 1), keepdims=True)
        self.std = np.std(X, axis=(0, 1), keepdims=True)
        return (X - self.mean) / self.std
        
    def transform(self, X):
        return (X - self.mean) / self.std

# ...existing code if any...

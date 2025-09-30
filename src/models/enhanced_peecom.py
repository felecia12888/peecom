#!/usr/bin/env python3
"""
Enhanced PEECOM with Advanced Feature Engineering

This version includes more sophisticated feature engineering techniques
to improve the poor cross-validation performance.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from .simple_peecom import SimplePEECOM
import warnings
warnings.filterwarnings('ignore')


class EnhancedPEECOM(BaseEstimator, ClassifierMixin):
    """
    Enhanced PEECOM with advanced feature engineering for improved performance
    """

    def __init__(self, 
                 n_estimators=200,  # More trees
                 max_depth=15,      # Deeper trees
                 min_samples_split=10,  # More conservative splitting
                 feature_selection_k=50,  # Select top features
                 use_polynomial=True,
                 polynomial_degree=2,
                 use_pca=False,
                 pca_components=0.95,
                 random_state=42):
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_selection_k = feature_selection_k
        self.use_polynomial = use_polynomial
        self.polynomial_degree = polynomial_degree
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.random_state = random_state
        
        # Components
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.poly_features = None
        self.original_features = None
        self.interaction_pairs = []  # Store interaction pairs for consistency

    def _create_advanced_physics_features(self, X):
        """Create advanced physics-inspired features"""
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        features = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove constant/near-constant features
        valid_cols = [col for col in numeric_cols if X[col].std() > 1e-6]
        
        if len(valid_cols) < 2:
            return features

        try:
            # 1. Enhanced Physics Features
            # Energy and power relationships (more systematic)
            for i, col1 in enumerate(valid_cols[:8]):  # Use more features
                for j, col2 in enumerate(valid_cols[i+1:9], i+1):
                    # Power features (P = F * v, Energy = P * t, etc.)
                    features[f'power_{col1}_{col2}'] = X[col1] * X[col2]
                    
                    # Efficiency ratios (output/input)
                    features[f'efficiency_{col1}_{col2}'] = X[col1] / (X[col2] + 1e-8)
                    
                    # Differential features (gradient-like)
                    features[f'diff_{col1}_{col2}'] = X[col1] - X[col2]
                    
                    # Harmonic mean (for rates)
                    features[f'harmonic_{col1}_{col2}'] = 2 * X[col1] * X[col2] / (X[col1] + X[col2] + 1e-8)

            # 2. Advanced Statistical Features
            # Moving statistics
            features['q25_all'] = X[valid_cols].quantile(0.25, axis=1)
            features['q75_all'] = X[valid_cols].quantile(0.75, axis=1)
            features['iqr_all'] = features['q75_all'] - features['q25_all']
            features['range_all'] = X[valid_cols].max(axis=1) - X[valid_cols].min(axis=1)
            features['cv_all'] = X[valid_cols].std(axis=1) / (X[valid_cols].mean(axis=1) + 1e-8)
            
            # 3. Physics-Inspired Ratios
            # Energy conservation indicators
            total_energy = X[valid_cols].sum(axis=1)
            for col in valid_cols[:5]:
                features[f'energy_fraction_{col}'] = X[col] / (total_energy + 1e-8)
            
            # 4. Stability Indicators
            # Variance-based stability
            features['stability_index'] = 1 / (1 + X[valid_cols].var(axis=1))
            
            # 5. Interaction Features (selective, deterministic)
            # Store high-correlation pairs during training
            if not self.interaction_pairs:
                corr_matrix = X[valid_cols].corr().abs()
                np.fill_diagonal(corr_matrix.values, 0)
                
                for i in range(len(valid_cols)):
                    for j in range(i+1, len(valid_cols)):
                        if corr_matrix.iloc[i, j] > 0.7:  # High correlation
                            self.interaction_pairs.append((valid_cols[i], valid_cols[j]))
                
                print(f"   Created {len(self.interaction_pairs)} high-correlation interaction features")
            
            # Apply stored interaction pairs safely
            if self.interaction_pairs:
                for col1, col2 in self.interaction_pairs:
                    if col1 in X.columns and col2 in X.columns:
                        features[f'interaction_{col1}_{col2}'] = X[col1] * X[col2]

        except Exception as e:
            print(f"Warning: Advanced feature creation failed: {e}")

        # Clean up
        features = features.replace([np.inf, -np.inf], 0)
        features = features.fillna(0)
        
        return features

    def fit(self, X, y):
        """Fit the Enhanced PEECOM model"""
        
        print("Training Enhanced PEECOM with advanced features...")
        
        # 1. Create advanced physics features
        X_enhanced = self._create_advanced_physics_features(X)
        self.feature_names_ = list(X_enhanced.columns)
        self.original_features = X.shape[1]
        print(f"   Features: {self.original_features} → {X_enhanced.shape[1]} (enhanced)")
        
        # 2. Polynomial features (if enabled)
        if self.use_polynomial and X_enhanced.shape[1] < 30:  # Only for smaller feature sets
            self.poly_features = PolynomialFeatures(
                degree=self.polynomial_degree, 
                interaction_only=True, 
                include_bias=False
            )
            X_poly = self.poly_features.fit_transform(X_enhanced)
            X_enhanced = pd.DataFrame(X_poly)
            print(f"   Polynomial features: {X_enhanced.shape[1]} total")
        
        # 3. Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_enhanced)
        
        # 4. Feature selection
        if self.feature_selection_k and X_scaled.shape[1] > self.feature_selection_k:
            self.feature_selector = SelectKBest(f_classif, k=self.feature_selection_k)
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
            print(f"   Feature selection: {X_scaled.shape[1]} → {X_selected.shape[1]}")
        else:
            X_selected = X_scaled
        
        # 5. PCA (if enabled)
        if self.use_pca:
            self.pca = PCA(n_components=self.pca_components)
            X_final = self.pca.fit_transform(X_selected)
            print(f"   PCA: {X_selected.shape[1]} → {X_final.shape[1]} components")
        else:
            X_final = X_selected
        
        # 6. Train enhanced Random Forest
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_final, y)
        
        print(f"✅ Enhanced PEECOM trained successfully!")
        print(f"   Final feature count: {X_final.shape[1]}")
        
        return self

    def _transform_features(self, X):
        """Apply the same feature transformation pipeline"""
        # Create enhanced features
        X_enhanced = self._create_advanced_physics_features(X)
        
        # Polynomial features
        if self.poly_features:
            X_poly = self.poly_features.transform(X_enhanced)
            X_enhanced = pd.DataFrame(X_poly)
        
        # Scale
        X_scaled = self.scaler.transform(X_enhanced)
        
        # Feature selection
        if self.feature_selector:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        # PCA
        if self.pca:
            X_final = self.pca.transform(X_selected)
        else:
            X_final = X_selected
            
        return X_final

    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not fitted yet!")
        
        X_transformed = self._transform_features(X)
        return self.model.predict(X_transformed)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not fitted yet!")
        
        X_transformed = self._transform_features(X)
        return self.model.predict_proba(X_transformed)

    def get_feature_importance(self):
        """Get feature importance from the underlying model"""
        if self.model is None:
            raise ValueError("Model not fitted yet!")
        return self.model.feature_importances_

    def get_feature_names(self):
        return getattr(self, 'feature_names_', None)

    def get_feature_names(self):
        return getattr(self, 'feature_names_', None)


# Export the class
__all__ = ['EnhancedPEECOM']
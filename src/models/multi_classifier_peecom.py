#!/usr/bin/env python3
"""
Enhanced Multi-Classifier PEECOM - Physics-Enhanced Ensemble Framework

This implements the multi-classifier approach where physics features
are tested across multiple algorithms to find the best performance.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from .simple_peecom import SimplePEECOM
import warnings
warnings.filterwarnings('ignore')


class MultiClassifierPEECOM(BaseEstimator, ClassifierMixin):
    """
    Multi-Classifier PEECOM - Tests physics features across multiple algorithms
    and selects the best performing combination.
    """

    def __init__(self, selection_strategy='best_physics_improvement', random_state=42):
        """
        Initialize Multi-Classifier PEECOM
        
        Parameters:
        -----------
        selection_strategy : str
            'best_physics_improvement': Select classifier with highest physics benefit
            'best_absolute': Select classifier with highest absolute performance
            'ensemble_voting': Use weighted ensemble of top performers
        """
        self.selection_strategy = selection_strategy
        self.random_state = random_state
        
        # Initialize classifiers to test
        self.classifiers = {
            'AdaBoost': AdaBoostClassifier(random_state=random_state),
            'Decision Tree': DecisionTreeClassifier(random_state=random_state, max_depth=10),
            'Neural Network': MLPClassifier(random_state=random_state, max_iter=300),
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
            'Random Forest': RandomForestClassifier(random_state=random_state),
            'Naive Bayes': GaussianNB(),
        }
        
        # Results storage
        self.selection_results = {}
        self.selected_classifier = None
        self.selected_name = None
        self.physics_enhancer = SimplePEECOM()
        self.scaler_raw = StandardScaler()
        self.scaler_physics = StandardScaler()
        
    def _test_classifiers(self, X, y):
        """Test all classifiers with and without physics features"""
        
        print("🔬 Testing classifiers with physics features...")
        
        # Create physics features
        X_physics = self.physics_enhancer._create_physics_features(X)
        
        # Scale both feature sets
        X_raw_scaled = self.scaler_raw.fit_transform(X)
        X_physics_scaled = self.scaler_physics.fit_transform(X_physics)
        
        results = {}
        
        for name, clf in self.classifiers.items():
            try:
                # Test with raw features
                scores_raw = cross_val_score(clf, X_raw_scaled, y, cv=5, scoring='accuracy')
                mean_raw = scores_raw.mean()
                
                # Test with physics features
                scores_physics = cross_val_score(clf, X_physics_scaled, y, cv=5, scoring='accuracy')
                mean_physics = scores_physics.mean()
                
                improvement = mean_physics - mean_raw
                
                results[name] = {
                    'raw_performance': mean_raw,
                    'physics_performance': mean_physics,
                    'improvement': improvement,
                    'classifier': clf
                }
                
                print(f"   {name:<18}: Raw={mean_raw:.4f}, Physics={mean_physics:.4f}, Δ={improvement:+.4f}")
                
            except Exception as e:
                print(f"   {name:<18}: Failed - {str(e)}")
                
        return results
    
    def _select_best_classifier(self, results):
        """Select the best classifier based on strategy"""
        
        if self.selection_strategy == 'best_physics_improvement':
            # Select classifier with highest physics improvement
            best_name = max(results.keys(), key=lambda k: results[k]['improvement'])
            criterion = 'physics improvement'
            
        elif self.selection_strategy == 'best_absolute':
            # Select classifier with highest absolute physics performance
            best_name = max(results.keys(), key=lambda k: results[k]['physics_performance'])
            criterion = 'absolute performance'
            
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")
        
        best_result = results[best_name]
        print(f"\n🏆 Selected: {best_name} (best {criterion})")
        print(f"   Raw: {best_result['raw_performance']:.4f}")
        print(f"   Physics: {best_result['physics_performance']:.4f}")
        print(f"   Improvement: {best_result['improvement']:+.4f}")
        
        return best_name, best_result['classifier']
    
    def fit(self, X, y):
        """Fit the Multi-Classifier PEECOM"""
        
        print("🚀 Training Multi-Classifier PEECOM...")
        
        # Test all classifiers
        self.selection_results = self._test_classifiers(X, y)
        
        # Select best classifier
        self.selected_name, self.selected_classifier = self._select_best_classifier(self.selection_results)
        
        # Train the selected classifier with physics features
        X_physics = self.physics_enhancer._create_physics_features(X)
        X_physics_scaled = self.scaler_physics.transform(X_physics)
        
        # Clone and train the selected classifier
        from sklearn.base import clone
        self.selected_classifier = clone(self.selected_classifier)
        self.selected_classifier.fit(X_physics_scaled, y)
        
        print(f"✅ Multi-Classifier PEECOM trained with {self.selected_name}")
        
        return self
    
    @property
    def scaler(self):
        """Return the physics scaler for compatibility"""
        return self.scaler_physics
    
    def predict(self, X):
        """Make predictions using the selected classifier"""
        if self.selected_classifier is None:
            raise ValueError("Model not fitted yet!")
        
        # Create physics features
        X_physics = self.physics_enhancer._create_physics_features(X)
        X_physics_scaled = self.scaler_physics.transform(X_physics)
        
        return self.selected_classifier.predict(X_physics_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.selected_classifier is None:
            raise ValueError("Model not fitted yet!")
        
        # Create physics features
        X_physics = self.physics_enhancer._create_physics_features(X)
        X_physics_scaled = self.scaler_physics.transform(X_physics)
        
        if hasattr(self.selected_classifier, 'predict_proba'):
            return self.selected_classifier.predict_proba(X_physics_scaled)
        else:
            # For classifiers without predict_proba (like SVC with default kernel)
            decision = self.selected_classifier.decision_function(X_physics_scaled)
            # Convert to probabilities using sigmoid for binary, softmax for multiclass
            if decision.ndim == 1:
                # Binary classification
                prob_pos = 1 / (1 + np.exp(-decision))
                return np.column_stack([1 - prob_pos, prob_pos])
            else:
                # Multiclass - softmax
                exp_scores = np.exp(decision - np.max(decision, axis=1, keepdims=True))
                return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def get_selection_summary(self):
        """Get summary of classifier selection process"""
        if not self.selection_results:
            return "No selection performed yet"
        
        summary = f"Multi-Classifier PEECOM Selection Summary\n"
        summary += f"="*50 + "\n"
        summary += f"Selected: {self.selected_name}\n"
        summary += f"Strategy: {self.selection_strategy}\n\n"
        summary += f"All Results:\n"
        
        for name, result in self.selection_results.items():
            marker = "🏆" if name == self.selected_name else "  "
            summary += f"{marker} {name:<18}: {result['physics_performance']:.4f} (Δ{result['improvement']:+.4f})\n"
        
        return summary
    
    def get_feature_importance(self):
        """Get feature importance if available from selected classifier"""
        if self.selected_classifier is None:
            raise ValueError("Model not fitted yet!")
        
        if hasattr(self.selected_classifier, 'feature_importances_'):
            return self.selected_classifier.feature_importances_
        elif hasattr(self.selected_classifier, 'coef_'):
            # For linear models, use absolute coefficient values
            return np.abs(self.selected_classifier.coef_).flatten()
        else:
            return None


# Export the class
__all__ = ['MultiClassifierPEECOM']
"""
Explainability Module for AI Resume Screening System
======================================================
Implements SHAP and LIME for model interpretability.

Author: AI Project Team
Course: CS-351 Artificial Intelligence
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Callable
import matplotlib.pyplot as plt


class ResumeExplainer:
    """
    Explainability wrapper for resume classification models.
    
    Provides SHAP and LIME explanations for model predictions,
    enabling transparent and interpretable hiring decisions.
    
    Attributes:
        model: Trained classification model
        feature_names: List of feature names
        class_names: List of class/category names
    """
    
    def __init__(self, model=None, feature_names: List[str] = None, 
                 class_names: List[str] = None):
        """
        Initialize the explainer.
        
        Args:
            model: Trained sklearn-compatible model
            feature_names: Names of features
            class_names: Names of prediction classes
        """
        self.model = model
        self.feature_names = feature_names or []
        self.class_names = class_names or ['Non-Match', 'Match']
        self.shap_explainer = None
        self.lime_explainer = None
    
    def setup_shap(self, background_data: np.ndarray = None):
        """
        Setup SHAP explainer.
        
        Args:
            background_data: Background data for SHAP (required for KernelExplainer)
        """
        try:
            import shap
            
            if hasattr(self.model, 'predict_proba'):
                if background_data is not None:
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba, 
                        background_data
                    )
                else:
                    self.shap_explainer = shap.Explainer(self.model)
            
            print("✅ SHAP explainer initialized")
        except Exception as e:
            print(f"⚠️ SHAP setup warning: {e}")
    
    def setup_lime(self):
        """Setup LIME explainer for text data."""
        try:
            from lime.lime_text import LimeTextExplainer
            
            self.lime_explainer = LimeTextExplainer(
                class_names=self.class_names,
                split_expression=r'\W+',
                bow=True
            )
            print("✅ LIME explainer initialized")
        except Exception as e:
            print(f"⚠️ LIME setup warning: {e}")
    
    def explain_shap(self, X: np.ndarray, num_features: int = 10) -> Dict:
        """
        Generate SHAP explanation for instances.
        
        Args:
            X: Feature matrix to explain
            num_features: Number of top features to show
            
        Returns:
            Dictionary with SHAP values and feature importances
        """
        if self.shap_explainer is None:
            return {'error': 'SHAP explainer not initialized'}
        
        try:
            import shap
            
            shap_values = self.shap_explainer.shap_values(X)
            
            # Get feature importance
            if isinstance(shap_values, list):
                # Multi-class
                importance = np.abs(shap_values[1]).mean(axis=0)
            else:
                importance = np.abs(shap_values).mean(axis=0)
            
            # Sort by importance
            indices = np.argsort(importance)[::-1][:num_features]
            
            top_features = []
            for idx in indices:
                if idx < len(self.feature_names):
                    top_features.append({
                        'feature': self.feature_names[idx],
                        'importance': float(importance[idx])
                    })
                else:
                    top_features.append({
                        'feature': f'feature_{idx}',
                        'importance': float(importance[idx])
                    })
            
            return {
                'shap_values': shap_values,
                'top_features': top_features,
                'mean_importance': importance
            }
        except Exception as e:
            return {'error': str(e)}
    
    def explain_lime_text(self, text: str, predict_fn: Callable, 
                          num_features: int = 10) -> Dict:
        """
        Generate LIME explanation for text.
        
        Args:
            text: Text to explain
            predict_fn: Prediction function
            num_features: Number of features in explanation
            
        Returns:
            Dictionary with LIME explanation
        """
        if self.lime_explainer is None:
            return {'error': 'LIME explainer not initialized'}
        
        try:
            explanation = self.lime_explainer.explain_instance(
                text,
                predict_fn,
                num_features=num_features
            )
            
            return {
                'prediction': explanation.predict_proba,
                'top_features': explanation.as_list(),
                'local_exp': dict(explanation.local_exp),
                'score': explanation.score
            }
        except Exception as e:
            return {'error': str(e)}
    
    def plot_feature_importance(self, importance_dict: Dict, 
                                 title: str = "Feature Importance",
                                 figsize: tuple = (10, 6)):
        """Plot feature importance bar chart."""
        if 'top_features' not in importance_dict:
            print("No features to plot")
            return
        
        features = [f['feature'] for f in importance_dict['top_features']]
        importances = [f['importance'] for f in importance_dict['top_features']]
        
        # Color based on importance
        colors = ['#2ecc71' if imp > np.mean(importances) else '#3498db' 
                  for imp in importances]
        
        plt.figure(figsize=figsize)
        plt.barh(features[::-1], importances[::-1], color=colors[::-1])
        plt.xlabel('Importance (|SHAP value|)', fontweight='bold')
        plt.ylabel('Feature', fontweight='bold')
        plt.title(title, fontweight='bold', fontsize=14)
        plt.tight_layout()
        
        return plt.gcf()


class TextExplainer:
    """
    Simplified text explainer for resume classification.
    
    Provides word-level importance scores for predictions.
    """
    
    # Keywords that indicate technical roles
    TECHNICAL_KEYWORDS = {
        'python': 0.45, 'machine learning': 0.42, 'tensorflow': 0.38,
        'deep learning': 0.40, 'pytorch': 0.38, 'sql': 0.30,
        'data science': 0.35, 'neural network': 0.35, 'nlp': 0.32,
        'aws': 0.28, 'docker': 0.25, 'kubernetes': 0.25,
        'java': 0.30, 'javascript': 0.28, 'react': 0.25,
        'data analysis': 0.30, 'statistics': 0.28, 'modeling': 0.25
    }
    
    NON_TECHNICAL_KEYWORDS = {
        'management': 0.30, 'leadership': 0.28, 'communication': 0.25,
        'hr': 0.35, 'recruitment': 0.32, 'sales': 0.30,
        'marketing': 0.28, 'customer service': 0.25, 'accounting': 0.28
    }
    
    def __init__(self, prediction_fn: Callable = None):
        """
        Initialize the text explainer.
        
        Args:
            prediction_fn: Function that takes text and returns prediction
        """
        self.prediction_fn = prediction_fn
    
    def explain(self, text: str) -> Dict:
        """
        Explain prediction for given text.
        
        Args:
            text: Resume text to explain
            
        Returns:
            Dictionary with explanation details
        """
        text_lower = text.lower()
        
        # Find keywords
        found_technical = {}
        found_non_technical = {}
        
        for keyword, weight in self.TECHNICAL_KEYWORDS.items():
            if keyword in text_lower:
                found_technical[keyword] = weight
        
        for keyword, weight in self.NON_TECHNICAL_KEYWORDS.items():
            if keyword in text_lower:
                found_non_technical[keyword] = weight
        
        # Calculate scores
        tech_score = sum(found_technical.values())
        non_tech_score = sum(found_non_technical.values())
        
        # Prediction
        if tech_score > non_tech_score:
            prediction = 'Technical'
            confidence = min(tech_score / (tech_score + non_tech_score + 0.1), 0.99)
        else:
            prediction = 'Non-Technical'
            confidence = min(non_tech_score / (tech_score + non_tech_score + 0.1), 0.99)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'technical_keywords': found_technical,
            'non_technical_keywords': found_non_technical,
            'technical_score': tech_score,
            'non_technical_score': non_tech_score,
            'top_positive_features': list(found_technical.items())[:5],
            'top_negative_features': list(found_non_technical.items())[:5]
        }
    
    def print_explanation(self, text: str):
        """Print formatted explanation."""
        exp = self.explain(text)
        
        print("\n" + "=" * 60)
        print("🔍 TEXT EXPLANATION")
        print("=" * 60)
        
        print(f"\n📝 Text Preview: {text[:100]}...")
        print(f"\n🎯 Prediction: {exp['prediction']}")
        print(f"📊 Confidence: {exp['confidence']:.2%}")
        
        print("\n✅ Positive Features (Supporting Prediction):")
        for kw, weight in exp['technical_keywords'].items():
            print(f"   • '{kw}' → +{weight:.2f}")
        
        print("\n❌ Negative Features (Against Prediction):")
        for kw, weight in exp['non_technical_keywords'].items():
            print(f"   • '{kw}' → -{weight:.2f}")
        
        print("\n" + "=" * 60)
        
        return exp


def create_feature_importance_summary(model, feature_names: List[str], 
                                        X_train: np.ndarray) -> pd.DataFrame:
    """
    Create comprehensive feature importance summary.
    
    Args:
        model: Trained model
        feature_names: Feature names
        X_train: Training data
        
    Returns:
        DataFrame with feature importances
    """
    importances = []
    
    # Get model feature importance if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).mean(axis=0) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
    else:
        # Use variance as proxy
        importances = np.var(X_train, axis=0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Feature': feature_names[:len(importances)],
        'Importance': importances[:len(feature_names)]
    })
    
    return df.sort_values('Importance', ascending=False).reset_index(drop=True)


def demo_explainability():
    """Demonstrate explainability features."""
    print("\n" + "=" * 60)
    print("🔍 EXPLAINABILITY DEMONSTRATION")
    print("=" * 60)
    
    # Create text explainer
    explainer = TextExplainer()
    
    # Test cases
    test_texts = [
        "Senior Python developer with 5 years of machine learning experience. "
        "Expert in TensorFlow and PyTorch for deep learning applications.",
        
        "HR Manager with 8 years in recruitment and talent acquisition. "
        "Strong leadership and communication skills.",
        
        "Data Scientist skilled in Python, SQL, and statistical modeling. "
        "Experience with NLP and computer vision projects."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*60}")
        print(f"EXAMPLE {i}")
        explainer.print_explanation(text)
    
    return explainer


if __name__ == "__main__":
    demo_explainability()

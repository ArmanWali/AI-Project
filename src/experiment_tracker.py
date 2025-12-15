"""
MLflow Experiment Tracking Module for AI Resume Screening System
=================================================================
Tracks experiments, metrics, and model artifacts using MLflow.

Author: AI Project Team
Course: CS-351 Artificial Intelligence

BONUS: +3% for ML Experiment Tracking (MLflow/W&B)
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import pickle


@dataclass
class ExperimentRun:
    """Represents a single experiment run."""
    run_id: str
    experiment_name: str
    model_name: str
    timestamp: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    tags: Dict[str, str]
    artifacts: List[str]


class MLflowTracker:
    """
    MLflow-compatible experiment tracking for ML models.
    
    Tracks:
    - Hyperparameters
    - Metrics (accuracy, precision, recall, F1)
    - Model artifacts
    - Training history
    
    Note: This is a lightweight implementation that works without MLflow server.
    For production, connect to actual MLflow server.
    """
    
    def __init__(self, experiment_name: str = "resume_screening",
                 tracking_uri: str = None,
                 use_mlflow: bool = False):
        """
        Initialize the tracker.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking server URI
            use_mlflow: Whether to use actual MLflow (requires installation)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.use_mlflow = use_mlflow
        self.runs: List[ExperimentRun] = []
        self.current_run: Optional[ExperimentRun] = None
        self.artifacts_dir = "mlruns/artifacts"
        
        # Create artifacts directory
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
        # Try to initialize MLflow if requested
        if use_mlflow:
            self._init_mlflow()
    
    def _init_mlflow(self):
        """Initialize actual MLflow tracking."""
        try:
            import mlflow
            
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            
            mlflow.set_experiment(self.experiment_name)
            self.mlflow = mlflow
            print(f"✅ MLflow initialized: {self.experiment_name}")
        except ImportError:
            print("⚠️ MLflow not installed. Using local tracking.")
            self.use_mlflow = False
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> str:
        """
        Start a new experiment run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run
            
        Returns:
            Run ID
        """
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if run_name:
            run_id = f"{run_name}_{run_id}"
        
        self.current_run = ExperimentRun(
            run_id=run_id,
            experiment_name=self.experiment_name,
            model_name="",
            timestamp=datetime.now().isoformat(),
            parameters={},
            metrics={},
            tags=tags or {},
            artifacts=[]
        )
        
        if self.use_mlflow and hasattr(self, 'mlflow'):
            self.mlflow.start_run(run_name=run_name)
            if tags:
                for k, v in tags.items():
                    self.mlflow.set_tag(k, v)
        
        print(f"🚀 Started run: {run_id}")
        return run_id
    
    def log_param(self, key: str, value: Any):
        """Log a parameter."""
        if self.current_run:
            self.current_run.parameters[key] = value
            
            if self.use_mlflow and hasattr(self, 'mlflow'):
                self.mlflow.log_param(key, value)
    
    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters."""
        for key, value in params.items():
            self.log_param(key, value)
    
    def log_metric(self, key: str, value: float, step: int = None):
        """Log a metric."""
        if self.current_run:
            self.current_run.metrics[key] = value
            
            if self.use_mlflow and hasattr(self, 'mlflow'):
                self.mlflow.log_metric(key, value, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)
    
    def log_model(self, model, model_name: str, signature: Dict = None):
        """
        Log a trained model.
        
        Args:
            model: Trained model object
            model_name: Name for the model
            signature: Model signature (input/output schema)
        """
        if self.current_run:
            self.current_run.model_name = model_name
            
            # Save model locally
            model_path = os.path.join(self.artifacts_dir, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            self.current_run.artifacts.append(model_path)
            
            if self.use_mlflow and hasattr(self, 'mlflow'):
                self.mlflow.sklearn.log_model(model, model_name)
            
            print(f"📦 Model logged: {model_name}")
    
    def log_artifact(self, filepath: str, artifact_path: str = None):
        """Log an artifact file."""
        if self.current_run:
            self.current_run.artifacts.append(filepath)
            
            if self.use_mlflow and hasattr(self, 'mlflow'):
                self.mlflow.log_artifact(filepath, artifact_path)
    
    def log_figure(self, fig, filename: str):
        """Log a matplotlib figure."""
        filepath = os.path.join(self.artifacts_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        
        if self.current_run:
            self.current_run.artifacts.append(filepath)
            
            if self.use_mlflow and hasattr(self, 'mlflow'):
                self.mlflow.log_figure(fig, filename)
        
        print(f"📊 Figure logged: {filename}")
    
    def end_run(self):
        """End the current run."""
        if self.current_run:
            self.runs.append(self.current_run)
            
            # Save run info
            run_path = os.path.join(self.artifacts_dir, f"{self.current_run.run_id}.json")
            with open(run_path, 'w') as f:
                json.dump(asdict(self.current_run), f, indent=2)
            
            if self.use_mlflow and hasattr(self, 'mlflow'):
                self.mlflow.end_run()
            
            print(f"✅ Run completed: {self.current_run.run_id}")
            self.current_run = None
    
    def get_run_summary(self, run_id: str = None) -> Dict:
        """Get summary of a specific run or current run."""
        run = None
        
        if run_id:
            for r in self.runs:
                if r.run_id == run_id:
                    run = r
                    break
        else:
            run = self.current_run or (self.runs[-1] if self.runs else None)
        
        if run:
            return asdict(run)
        return {}
    
    def compare_runs(self, run_ids: List[str] = None) -> Dict:
        """Compare metrics across runs."""
        runs_to_compare = self.runs if not run_ids else [r for r in self.runs if r.run_id in run_ids]
        
        if not runs_to_compare:
            return {}
        
        comparison = {
            'runs': [],
            'best_run': None,
            'best_metric': 0
        }
        
        for run in runs_to_compare:
            run_summary = {
                'run_id': run.run_id,
                'model': run.model_name,
                'timestamp': run.timestamp,
                **run.metrics
            }
            comparison['runs'].append(run_summary)
            
            # Track best run (by accuracy or F1)
            score = run.metrics.get('accuracy', run.metrics.get('f1_score', 0))
            if score > comparison['best_metric']:
                comparison['best_metric'] = score
                comparison['best_run'] = run.run_id
        
        return comparison
    
    def print_run_summary(self, run: ExperimentRun = None):
        """Pretty print run summary."""
        run = run or self.current_run or (self.runs[-1] if self.runs else None)
        
        if not run:
            print("No runs available")
            return
        
        print("\n" + "=" * 60)
        print("📊 EXPERIMENT RUN SUMMARY")
        print("=" * 60)
        
        print(f"\n🔹 Run ID: {run.run_id}")
        print(f"🔹 Experiment: {run.experiment_name}")
        print(f"🔹 Model: {run.model_name}")
        print(f"🔹 Timestamp: {run.timestamp}")
        
        print("\n📋 Parameters:")
        for k, v in run.parameters.items():
            print(f"   {k}: {v}")
        
        print("\n📈 Metrics:")
        for k, v in run.metrics.items():
            if isinstance(v, float):
                print(f"   {k}: {v:.4f}")
            else:
                print(f"   {k}: {v}")
        
        print(f"\n📦 Artifacts: {len(run.artifacts)} files")
        for artifact in run.artifacts[:5]:
            print(f"   • {artifact}")
        
        print("\n" + "=" * 60)


def track_experiment(model, X_train, y_train, X_test, y_test,
                     model_name: str, params: Dict,
                     tracker: MLflowTracker = None):
    """
    Convenience function to track a complete experiment.
    
    Args:
        model: Model to train and evaluate
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_name: Name of the model
        params: Hyperparameters
        tracker: MLflowTracker instance
        
    Returns:
        Trained model and metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    tracker = tracker or MLflowTracker()
    
    # Start run
    tracker.start_run(run_name=model_name, tags={'type': 'classification'})
    
    # Log parameters
    tracker.log_params(params)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }
    
    # Log metrics
    tracker.log_metrics(metrics)
    
    # Log model
    tracker.log_model(model, model_name)
    
    # End run
    tracker.end_run()
    
    return model, metrics


def demo_tracking():
    """Demonstrate experiment tracking."""
    print("\n" + "=" * 60)
    print("📊 MLFLOW EXPERIMENT TRACKING DEMO")
    print("=" * 60)
    
    tracker = MLflowTracker(experiment_name="resume_classification_demo")
    
    # Simulate multiple experiment runs
    experiments = [
        {
            'name': 'random_forest_baseline',
            'params': {'n_estimators': 100, 'max_depth': 10},
            'metrics': {'accuracy': 0.9859, 'f1_score': 0.9842, 'precision': 0.9846, 'recall': 0.9859}
        },
        {
            'name': 'logistic_regression',
            'params': {'C': 1.0, 'max_iter': 1000},
            'metrics': {'accuracy': 0.9779, 'f1_score': 0.9756, 'precision': 0.9781, 'recall': 0.9779}
        },
        {
            'name': 'bert_classifier',
            'params': {'learning_rate': 2e-5, 'epochs': 3, 'batch_size': 8},
            'metrics': {'accuracy': 0.9920, 'f1_score': 0.9915, 'precision': 0.9918, 'recall': 0.9920}
        }
    ]
    
    for exp in experiments:
        tracker.start_run(run_name=exp['name'], tags={'model_type': exp['name'].split('_')[0]})
        tracker.log_params(exp['params'])
        tracker.log_metrics(exp['metrics'])
        tracker.current_run.model_name = exp['name']
        tracker.end_run()
    
    # Print comparison
    print("\n📊 EXPERIMENT COMPARISON")
    print("-" * 60)
    comparison = tracker.compare_runs()
    
    print(f"\n{'Model':<25} {'Accuracy':<12} {'F1-Score':<12} {'Precision':<12}")
    print("-" * 60)
    
    for run in comparison['runs']:
        print(f"{run['model']:<25} {run.get('accuracy', 0):<12.4f} {run.get('f1_score', 0):<12.4f} {run.get('precision', 0):<12.4f}")
    
    print(f"\n🏆 Best Run: {comparison['best_run']} (Score: {comparison['best_metric']:.4f})")
    
    return tracker


if __name__ == "__main__":
    demo_tracking()

#!/usr/bin/env python3
"""
Wine Quality Classification - Training Script

This script trains machine learning models for wine quality classification
using the processed dataset and saves the best performing model.

Usage:
    python train_model.py [--model MODEL_TYPE] [--tune] [--output PATH]

Example:
    python train_model.py --model xgboost --tune --output ../models/
"""

import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Project imports
from ttac_test_ds_classification.data import (
    load_wine_quality_dataset,
    clean_wine_data,
    prepare_wine_features_target,
    split_wine_data
)
from ttac_test_ds_classification.models import WineQualityClassifier, compare_models


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def load_and_prepare_data(data_path: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and prepare data for training.
    
    Args:
        data_path: Path to dataset file (optional)
    
    Returns:
        Prepared features and target
    """
    logger = logging.getLogger(__name__)
    
    if data_path and Path(data_path).exists():
        logger.info(f"Loading data from: {data_path}")
        wine_data = pd.read_csv(data_path)
    else:
        logger.info("Loading wine quality dataset from source")
        wine_data = load_wine_quality_dataset(include_both=True)
    
    logger.info(f"Dataset loaded - Shape: {wine_data.shape}")
    
    # Clean data
    logger.info("Cleaning data...")
    wine_data = clean_wine_data(wine_data)
    logger.info(f"Data cleaned - Shape: {wine_data.shape}")
    
    # Prepare features and target
    logger.info("Preparing features and target...")
    X, y = prepare_wine_features_target(wine_data)
    logger.info(f"Features: {X.shape}, Target: {y.shape}")
    
    return X, y


def train_single_model(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tune_hyperparameters: bool = False,
    random_state: int = 42
) -> Tuple[WineQualityClassifier, Dict[str, float]]:
    """
    Train a single model.
    
    Args:
        model_type: Type of model to train
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        tune_hyperparameters: Whether to tune hyperparameters
        random_state: Random state for reproducibility
    
    Returns:
        Trained classifier and metrics
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Training {model_type} model...")
    
    # Initialize classifier
    classifier = WineQualityClassifier(
        model_type=model_type,
        random_state=random_state
    )
    
    # Tune hyperparameters if requested
    if tune_hyperparameters:
        logger.info("Tuning hyperparameters...")
        tuning_results = classifier.tune_hyperparameters(
            X_train, y_train, cv=5, scoring='f1_weighted'
        )
        logger.info(f"Best CV score: {tuning_results['best_score']:.4f}")
        logger.info(f"Best parameters: {tuning_results['best_params']}")
    
    # Train model
    classifier.fit(X_train, y_train)
    
    # Evaluate
    metrics = classifier.evaluate(X_test, y_test)
    
    logger.info(f"{model_type} - Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"{model_type} - F1-Score: {metrics['f1_weighted']:.4f}")
    
    return classifier, metrics


def save_model_artifacts(
    classifier: WineQualityClassifier,
    scaler: StandardScaler,
    metrics: Dict[str, float],
    feature_names: list,
    target_classes: list,
    output_path: Path,
    model_name: str = "best_model"
) -> None:
    """
    Save model and related artifacts.
    
    Args:
        classifier: Trained classifier
        scaler: Feature scaler
        metrics: Model metrics
        feature_names: List of feature names
        target_classes: List of target classes
        output_path: Output directory
        model_name: Name for the model files
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_file = output_path / f"{model_name}.pkl"
    classifier.save_model(str(model_file))
    logger.info(f"Model saved to: {model_file}")
    
    # Save scaler
    scaler_file = output_path / f"{model_name}_scaler.pkl"
    joblib.dump(scaler, scaler_file)
    logger.info(f"Scaler saved to: {scaler_file}")
    
    # Save feature importance if available
    if hasattr(classifier.model, 'feature_importances_'):
        importance_df = classifier.get_feature_importance()
        importance_file = output_path / f"{model_name}_feature_importance.csv"
        importance_df.to_csv(importance_file, index=False)
        logger.info(f"Feature importance saved to: {importance_file}")
    
    # Save metadata
    metadata = {
        'model_type': classifier.model_type,
        'training_date': datetime.now().isoformat(),
        'performance_metrics': {k: float(v) for k, v in metrics.items()},
        'feature_names': feature_names,
        'target_classes': target_classes,
        'model_file': str(model_file.name),
        'scaler_file': str(scaler_file.name)
    }
    
    metadata_file = output_path / f"{model_name}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to: {metadata_file}")


def train_model():
    """Train a machine learning model for wine quality classification."""
    # Legacy function - use main() for full functionality
    logger = logging.getLogger(__name__)
    logger.info("Training wine quality classification model...")
    
    # Simple training with default parameters
    X, y = load_and_prepare_data()
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Train Random Forest (default)
    classifier, metrics = train_single_model(
        "random_forest", X_train_scaled, y_train,
        X_test_scaled, y_test
    )
    
    # Save model
    output_path = Path("models")
    save_model_artifacts(
        classifier, scaler, metrics,
        list(X.columns), sorted(y.unique().tolist()),
        output_path, "wine_quality_model"
    )
    
    logger.info("✅ Training completed successfully!")


def main():
    """Main training function with command line arguments."""
    parser = argparse.ArgumentParser(description="Train wine quality classification model")
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "svm", "xgboost", "all"],
        help="Model type to train (default: random_forest)"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform hyperparameter tuning"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help="Output directory for trained models (default: models)"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to dataset file (optional)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (default: 0.2)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting Wine Quality Classification Training")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load and prepare data
        X, y = load_and_prepare_data(args.data)
        
        # Split data
        logger.info("Splitting data...")
        X_train, X_test, y_train, y_test = split_wine_data(
            X, y,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        # Scale features
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        output_path = Path(args.output)
        
        if args.model == "all":
            # Train all models and compare
            logger.info("Training all models for comparison...")
            
            results = compare_models(
                X_train_scaled, y_train,
                X_test_scaled, y_test,
                tune_hyperparameters=args.tune
            )
            
            # Find best model
            best_model_type = max(results.keys(), key=lambda k: results[k]['f1_weighted'])
            best_metrics = results[best_model_type]
            
            logger.info(f"Best model: {best_model_type}")
            logger.info(f"Best F1-score: {best_metrics['f1_weighted']:.4f}")
            
            # Train best model again to get classifier object
            best_classifier, _ = train_single_model(
                best_model_type, X_train_scaled, y_train,
                X_test_scaled, y_test, args.tune, args.random_state
            )
            
            # Save comparison results
            results_df = pd.DataFrame(results).T
            results_file = output_path / "model_comparison_results.csv"
            output_path.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(results_file)
            logger.info(f"Comparison results saved to: {results_file}")
            
            # Save best model
            save_model_artifacts(
                best_classifier, scaler, best_metrics,
                list(X.columns), sorted(y.unique().tolist()),
                output_path, f"best_model_{best_model_type}"
            )
            
        else:
            # Train single model
            classifier, metrics = train_single_model(
                args.model, X_train_scaled, y_train,
                X_test_scaled, y_test, args.tune, args.random_state
            )
            
            # Save model
            model_name = f"{args.model}_model"
            if args.tune:
                model_name += "_tuned"
            
            save_model_artifacts(
                classifier, scaler, metrics,
                list(X.columns), sorted(y.unique().tolist()),
                output_path, model_name
            )
        
        logger.info("✅ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

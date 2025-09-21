#!/usr/bin/env python3
"""
Wine Quality Classification - Inference Script

This script performs inference using a trained wine quality classification model.
It can process single samples or batch predictions from CSV files.

Usage:
    python inference.py --model MODEL_PATH --input INPUT_DATA [--output OUTPUT_PATH]

Examples:
    # Single prediction with feature values
    python inference.py --model models/best_model.pkl --features 7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4
    
    # Batch prediction from CSV
    python inference.py --model models/best_model.pkl --input data/new_wines.csv --output predictions.csv
"""

import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Project imports
from ttac_test_ds_classification.models import WineQualityClassifier


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


class WineQualityPredictor:
    """
    Wine Quality Prediction class for inference.
    
    Handles loading trained models and making predictions on new data.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to trained model file
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.target_classes = None
        self.metadata = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load trained model and associated artifacts."""
        self.logger.info(f"Loading model from: {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load main model
        self.model = WineQualityClassifier.load_model(str(self.model_path))
        self.feature_names = self.model.feature_names
        
        # Load scaler
        scaler_path = self.model_path.parent / f"{self.model_path.stem}_scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            self.logger.info(f"Scaler loaded from: {scaler_path}")
        else:
            self.logger.warning("No scaler found - predictions may be inaccurate")
        
        # Load metadata
        metadata_path = self.model_path.parent / f"{self.model_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.target_classes = self.metadata.get('target_classes', [])
            self.logger.info(f"Metadata loaded from: {metadata_path}")
        
        self.logger.info("✅ Model loaded successfully")
        self.logger.info(f"Model type: {self.model.model_type}")
        self.logger.info(f"Features expected: {len(self.feature_names)}")
    
    def _validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and prepare features for prediction.
        
        Args:
            features: Input features DataFrame
        
        Returns:
            Validated and prepared features
        """
        # Check if all required features are present
        missing_features = set(self.feature_names) - set(features.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Reorder columns to match training data
        features = features[self.feature_names]
        
        # Check for missing values
        if features.isnull().any().any():
            self.logger.warning("Missing values detected in input features")
            # Simple imputation with median (you might want more sophisticated methods)
            features = features.fillna(features.median())
        
        return features
    
    def _preprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features using the trained scaler.
        
        Args:
            features: Raw features DataFrame
        
        Returns:
            Preprocessed features
        """
        if self.scaler is not None:
            scaled_features = self.scaler.transform(features)
            return pd.DataFrame(
                scaled_features,
                columns=features.columns,
                index=features.index
            )
        else:
            self.logger.warning("No scaler available - using raw features")
            return features
    
    def predict_single(
        self,
        features: Union[List[float], Dict[str, float], pd.Series],
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Make prediction for a single wine sample.
        
        Args:
            features: Feature values (list, dict, or pandas Series)
            return_probabilities: Whether to return class probabilities
        
        Returns:
            Prediction results dictionary
        """
        # Convert input to DataFrame
        if isinstance(features, list):
            if len(features) != len(self.feature_names):
                raise ValueError(f"Expected {len(self.feature_names)} features, got {len(features)}")
            features_df = pd.DataFrame([features], columns=self.feature_names)
        elif isinstance(features, dict):
            features_df = pd.DataFrame([features])
        elif isinstance(features, pd.Series):
            features_df = pd.DataFrame([features])
        else:
            raise ValueError("Features must be list, dict, or pandas Series")
        
        # Validate and preprocess
        features_df = self._validate_features(features_df)
        features_df = self._preprocess_features(features_df)
        
        # Make prediction
        prediction = self.model.predict(features_df)[0]
        
        result = {
            'predicted_quality': int(prediction),
            'model_type': self.model.model_type
        }
        
        if return_probabilities:
            probabilities = self.model.predict_proba(features_df)[0]
            if self.target_classes:
                prob_dict = {
                    f'quality_{cls}': float(prob)
                    for cls, prob in zip(self.target_classes, probabilities)
                }
                result['probabilities'] = prob_dict
                result['confidence'] = float(max(probabilities))
        
        return result
    
    def predict_batch(
        self,
        features: pd.DataFrame,
        return_probabilities: bool = False
    ) -> pd.DataFrame:
        """
        Make predictions for multiple wine samples.
        
        Args:
            features: Features DataFrame
            return_probabilities: Whether to return class probabilities
        
        Returns:
            Predictions DataFrame
        """
        self.logger.info(f"Making batch predictions for {len(features)} samples")
        
        # Validate and preprocess
        features = self._validate_features(features)
        features_processed = self._preprocess_features(features)
        
        # Make predictions
        predictions = self.model.predict(features_processed)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'predicted_quality': predictions.astype(int)
        }, index=features.index)
        
        if return_probabilities:
            probabilities = self.model.predict_proba(features_processed)
            
            if self.target_classes:
                # Add probability columns
                for i, cls in enumerate(self.target_classes):
                    results[f'prob_quality_{cls}'] = probabilities[:, i]
                
                # Add confidence (max probability)
                results['confidence'] = probabilities.max(axis=1)
        
        self.logger.info("✅ Batch predictions completed")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            'model_type': self.model.model_type,
            'feature_names': self.feature_names,
            'target_classes': self.target_classes,
            'model_path': str(self.model_path)
        }
        
        if self.metadata:
            info.update({
                'training_date': self.metadata.get('training_date'),
                'performance_metrics': self.metadata.get('performance_metrics')
            })
        
        return info


def parse_feature_string(feature_string: str, feature_names: List[str]) -> List[float]:
    """
    Parse comma-separated feature string into list of floats.
    
    Args:
        feature_string: Comma-separated feature values
        feature_names: Expected feature names
    
    Returns:
        List of feature values
    """
    try:
        features = [float(x.strip()) for x in feature_string.split(',')]
        if len(features) != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} features, got {len(features)}")
        return features
    except ValueError as e:
        raise ValueError(f"Error parsing features: {e}")


def load_input_data(input_path: str) -> pd.DataFrame:
    """
    Load input data from CSV file.
    
    Args:
        input_path: Path to input CSV file
    
    Returns:
        Input data DataFrame
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if input_file.suffix.lower() != '.csv':
        raise ValueError("Input file must be a CSV file")
    
    return pd.read_csv(input_path)


def save_predictions(
    predictions: pd.DataFrame,
    output_path: str,
    include_input: bool = True,
    input_data: pd.DataFrame = None
) -> None:
    """
    Save predictions to CSV file.
    
    Args:
        predictions: Predictions DataFrame
        output_path: Output file path
        include_input: Whether to include input features
        input_data: Original input data (if including input)
    """
    if include_input and input_data is not None:
        # Combine input and predictions
        output_df = pd.concat([input_data, predictions], axis=1)
    else:
        output_df = predictions
    
    output_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to: {output_path}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Wine quality classification inference")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model file"
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--features",
        type=str,
        help="Comma-separated feature values for single prediction"
    )
    input_group.add_argument(
        "--input",
        type=str,
        help="Path to CSV file with features for batch prediction"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for predictions (CSV format)"
    )
    parser.add_argument(
        "--probabilities",
        action="store_true",
        help="Include prediction probabilities"
    )
    parser.add_argument(
        "--model-info",
        action="store_true",
        help="Display model information"
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
    
    try:
        # Initialize predictor
        predictor = WineQualityPredictor(args.model)
        
        # Display model info if requested
        if args.model_info:
            model_info = predictor.get_model_info()
            print("\n📊 MODEL INFORMATION")
            print("=" * 50)
            for key, value in model_info.items():
                print(f"{key}: {value}")
            print()
        
        # Single prediction
        if args.features:
            logger.info("Performing single prediction")
            
            # Parse features
            features_list = parse_feature_string(args.features, predictor.feature_names)
            
            # Make prediction
            result = predictor.predict_single(
                features_list,
                return_probabilities=args.probabilities
            )
            
            # Display result
            print("\n🎯 PREDICTION RESULT")
            print("=" * 30)
            print(f"Predicted Quality: {result['predicted_quality']}")
            print(f"Model Type: {result['model_type']}")
            
            if 'confidence' in result:
                print(f"Confidence: {result['confidence']:.3f}")
            
            if 'probabilities' in result:
                print("\nClass Probabilities:")
                for quality, prob in result['probabilities'].items():
                    print(f"  {quality}: {prob:.3f}")
        
        # Batch prediction
        elif args.input:
            logger.info(f"Performing batch prediction from: {args.input}")
            
            # Load input data
            input_data = load_input_data(args.input)
            logger.info(f"Loaded {len(input_data)} samples")
            
            # Make predictions
            predictions = predictor.predict_batch(
                input_data,
                return_probabilities=args.probabilities
            )
            
            # Save or display results
            if args.output:
                save_predictions(
                    predictions, args.output,
                    include_input=True, input_data=input_data
                )
            else:
                print("\n📈 BATCH PREDICTION RESULTS")
                print("=" * 50)
                
                # Show summary
                print(f"Total samples: {len(predictions)}")
                print(f"Average predicted quality: {predictions['predicted_quality'].mean():.2f}")
                print(f"Quality distribution:")
                quality_dist = predictions['predicted_quality'].value_counts().sort_index()
                for quality, count in quality_dist.items():
                    print(f"  Quality {quality}: {count} samples")
                
                if 'confidence' in predictions.columns:
                    print(f"Average confidence: {predictions['confidence'].mean():.3f}")
                
                # Show first few predictions
                print(f"\nFirst 10 predictions:")
                display_cols = ['predicted_quality']
                if 'confidence' in predictions.columns:
                    display_cols.append('confidence')
                print(predictions[display_cols].head(10).to_string())
        
        logger.info("✅ Inference completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Inference failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

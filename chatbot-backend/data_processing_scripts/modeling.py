import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, batch_size: int = 1000):
        """Initialize model trainer"""
        self.data_dir = Path(__file__).parent.parent / "data"
        self.model_dir = self.data_dir / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        
        # Set style for visualizations with error handling
        try:
            # Try different style options
            style_options = ['seaborn', 'seaborn-v0_8', 'seaborn-darkgrid', 'default']
            for style in style_options:
                try:
                    plt.style.use(style)
                    logger.info(f"Using matplotlib style: {style}")
                    break
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Could not set matplotlib style: {str(e)}")
            plt.style.use('default')
        
        try:
            sns.set_theme(style="whitegrid")
            logger.info("Using seaborn whitegrid style")
        except Exception as e:
            logger.warning(f"Could not set seaborn style: {str(e)}")
        
        try:
            sns.set_palette("husl")
            logger.info("Using seaborn husl palette")
        except Exception as e:
            logger.warning(f"Could not set seaborn palette: {str(e)}")
    
    def load_data(self) -> pd.DataFrame:
        """Load processed data"""
        try:
            logger.info("Loading processed data...")
            with open(self.data_dir / "processed_data.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
                df = pd.DataFrame(data)
            logger.info(f"Loaded {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features and target for modeling"""
        try:
            # Select numeric features
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Select categorical features
            categorical_features = ['category', 'intent']  # Add other categorical features as needed
            
            # Prepare feature matrix
            X = df[numeric_features + categorical_features].copy()
            
            # Encode categorical variables
            label_encoders = {}
            for col in categorical_features:
                label_encoders[col] = LabelEncoder()
                X[col] = label_encoders[col].fit_transform(X[col])
            
            # Save label encoders
            with open(self.model_dir / 'label_encoders.json', 'w') as f:
                json.dump({k: v.classes_.tolist() for k, v in label_encoders.items()}, f)
            
            # Prepare target variable (example: using 'category' as target)
            y = X['category']
            X = X.drop('category', axis=1)
            
            # Scale features
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            
            # Save scaler
            with open(self.model_dir / 'feature_scaler.json', 'w') as f:
                json.dump({
                    'mean': scaler.mean_.tolist(),
                    'scale': scaler.scale_.tolist(),
                    'feature_names': X.columns.tolist()
                }, f)
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def train_model(self, X: pd.DataFrame, y: pd.Series):
        """Train the model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Save model performance metrics
            with open(self.model_dir / 'model_metrics.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            # Save model
            import joblib
            joblib.dump(model, self.model_dir / 'model.joblib')
            
            # Analyze feature importance
            self.analyze_feature_importance(model, X)
            
            return model, report
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def analyze_feature_importance(self, model: RandomForestClassifier, X: pd.DataFrame):
        """Analyze feature importance"""
        try:
            # Get feature importance
            importance = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Save feature importance
            feature_importance.to_csv(self.model_dir / 'feature_importance.csv', index=False)
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            sns.barplot(data=feature_importance, x='importance', y='feature')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig(self.model_dir / 'feature_importance.png')
            plt.close()
            
            # Create interactive visualization
            fig = px.bar(feature_importance, x='importance', y='feature',
                        title='Feature Importance')
            fig.write_html(self.model_dir / 'feature_importance.html')
            
            logger.info("Feature importance analysis completed")
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {str(e)}")
            raise
    
    def run_modeling_pipeline(self):
        """Run the complete modeling pipeline"""
        try:
            logger.info("Starting modeling pipeline")
            
            # Load data
            df = self.load_data()
            
            # Prepare features
            logger.info("Preparing features...")
            X, y = self.prepare_features(df)
            
            # Train model
            logger.info("Training model...")
            model, report = self.train_model(X, y)
            
            logger.info("Modeling pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in modeling pipeline: {str(e)}")
            raise

def main():
    """Main function to run modeling pipeline"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run modeling pipeline
    trainer = ModelTrainer(batch_size=1000)
    trainer.run_modeling_pipeline()

if __name__ == "__main__":
    main() 
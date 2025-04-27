import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        """Initialize data preprocessor"""
        self.data_dir = Path("../data")
        self.analysis_dir = self.data_dir / "analysis"
        self.viz_dir = self.data_dir / "visualizations"
        
        # Create necessary directories
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> pd.DataFrame:
        """Load raw data"""
        try:
            with open(self.data_dir / "processed_data.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            logger.info(f"Loaded {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform exploratory data analysis"""
        try:
             # Basic statistics
            stats = {
                'total_samples': int(len(df)),
                'unique_categories': int(df['category'].nunique()),
                'unique_intents': int(df['intent'].nunique()),
                'avg_text_length': float(df['instruction'].str.len().mean()),  # float để tránh np.float64
                'max_text_length': int(df['instruction'].str.len().max())
            }

            category_dist = {k: int(v) for k, v in df['category'].value_counts().to_dict().items()}
            intent_dist = {k: int(v) for k, v in df['intent'].value_counts().to_dict().items()}

            results = {
                'basic_stats': stats,
                'category_distribution': category_dist,
                'intent_distribution': intent_dist
            }

            # Lưu file JSON mà không cần `default=str`
            with open(self.analysis_dir / "eda_results.json", 'w') as f:
                json.dump(results, f, indent=2)

            
            logger.info("EDA completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in EDA: {str(e)}")
            raise
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create data visualizations"""
        try:
            # 1. Category Distribution
            plt.figure(figsize=(10, 6))
            df['category'].value_counts().plot(kind='bar')
            plt.title('Category Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'category_distribution.png')
            plt.close()
            
            # 2. Intent Distribution
            plt.figure(figsize=(10, 6))
            df['intent'].value_counts().plot(kind='bar')
            plt.title('Intent Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'intent_distribution.png')
            plt.close()
            
            # 3. Text Length Distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=df['instruction'].str.len(), bins=50)
            plt.title('Text Length Distribution')
            plt.xlabel('Text Length')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'text_length_distribution.png')
            plt.close()
            
            logger.info("Visualizations created successfully")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for training"""
        try:
            # Convert text to lowercase
            df['instruction'] = df['instruction'].str.lower()
            
            # Remove extra whitespace
            df['instruction'] = df['instruction'].str.strip()
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['instruction'])
            
            # Remove rows with missing values
            df = df.dropna()
            
            logger.info(f"Preprocessed {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def run_pipeline(self):
        """Run the complete data preprocessing pipeline"""
        try:
            # Load data
            df = self.load_data()
            
            # Preprocess data
            df = self.preprocess_data(df)
            
            # Analyze data
            analysis_results = self.analyze_data(df)
            
            # Create visualizations
            self.create_visualizations(df)
            
            # Save preprocessed data
            df.to_json(self.data_dir / "preprocessed_data.json", orient='records')
            
            logger.info("Data preprocessing pipeline completed successfully")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise

def main():
    """Main function to run data preprocessing"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run preprocessing pipeline
    preprocessor = DataPreprocessor()
    preprocessor.run_pipeline()

if __name__ == "__main__":
    main() 
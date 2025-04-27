import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

# Try to import GPU libraries
try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("CuPy GPU acceleration enabled")
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("CuPy not available, using CPU only")

try:
    import cudf
    RAPIDS_AVAILABLE = True
    logger.info("RAPIDS GPU acceleration enabled")
except ImportError:
    RAPIDS_AVAILABLE = False
    logger.warning("RAPIDS not available, using CPU only")

class DataAnalyzer:
    def __init__(self, batch_size: int = 1000, skip_visualizations: bool = False, use_gpu: bool = True):
        """Initialize data analyzer"""
        self.data_dir = Path("../data")
        self.analysis_dir = self.data_dir / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.skip_visualizations = skip_visualizations
        self.use_gpu = use_gpu and (GPU_AVAILABLE or RAPIDS_AVAILABLE)
        
        if not self.use_gpu:
            logger.warning("GPU acceleration disabled, using CPU only")
        
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
        """Load processed data in batches"""
        try:
            logger.info("Loading data in batches...")
            all_data = []
            
            with open(self.data_dir / "processed_data.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
                total_records = len(data)
                
                for i in tqdm(range(0, total_records, self.batch_size), desc="Loading data"):
                    batch = data[i:i + self.batch_size]
                    all_data.extend(batch)
            
            # Convert to GPU DataFrame if available
            if self.use_gpu and RAPIDS_AVAILABLE:
                df = cudf.DataFrame(all_data)
                logger.info("Data loaded into GPU memory")
            else:
                df = pd.DataFrame(all_data)
            
            logger.info(f"Loaded {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def analyze_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze and handle missing values"""
        try:
            # Calculate missing value statistics
            missing_stats = df.isnull().sum()
            missing_percent = (missing_stats / len(df)) * 100
            
            # Log only columns with significant missing values
            significant_missing = missing_percent[missing_percent > 0]
            if not significant_missing.empty:
                logger.info("Columns with missing values:")
                for col, percent in significant_missing.items():
                    logger.info(f"{col}: {percent:.2f}% missing")
            
            # Remove columns with too many missing values (>50%)
            columns_to_drop = missing_percent[missing_percent > 50].index
            if len(columns_to_drop) > 0:
                logger.info(f"Removed {len(columns_to_drop)} columns with >50% missing values")
                df = df.drop(columns=columns_to_drop)
            
            # Handle remaining missing values based on data type
            for col in df.columns:
                if df[col].isnull().any():
                    if df[col].dtype in ['int64', 'float64']:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0])
            
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise
    
    def handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate records"""
        try:
            initial_count = len(df)
            df = df.drop_duplicates()
            removed_count = initial_count - len(df)
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} duplicate records ({removed_count/initial_count*100:.2f}%)")
            
            return df
            
        except Exception as e:
            logger.error(f"Error handling duplicates: {str(e)}")
            raise
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in numeric columns"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            total_outliers = 0
            
            for col in numeric_columns:
                # Calculate IQR using GPU if available
                if self.use_gpu and RAPIDS_AVAILABLE:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Count outliers
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                    if len(outliers) > 0:
                        total_outliers += len(outliers)
                    
                    # Clip values
                    df[col] = df[col].clip(lower_bound, upper_bound)
                else:
                    # CPU version
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                    if len(outliers) > 0:
                        total_outliers += len(outliers)
                    
                    df[col] = df[col].clip(lower_bound, upper_bound)
            
            if total_outliers > 0:
                logger.info(f"Handled {total_outliers} outliers across {len(numeric_columns)} numeric columns")
            
            return df
            
        except Exception as e:
            logger.error(f"Error handling outliers: {str(e)}")
            raise
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numeric data"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) > 0:
                if self.use_gpu and RAPIDS_AVAILABLE:
                    # GPU version using RAPIDS
                    for col in numeric_columns:
                        df[col] = (df[col] - df[col].mean()) / df[col].std()
                        df[f'{col}_minmax'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                else:
                    # CPU version
                    for col in numeric_columns:
                        standard_scaler = StandardScaler()
                        df[col] = standard_scaler.fit_transform(df[[col]])
                        
                        minmax_scaler = MinMaxScaler()
                        df[f'{col}_minmax'] = minmax_scaler.fit_transform(df[[col]])
                
                logger.info(f"Normalized numeric columns: {numeric_columns.tolist()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error normalizing data: {str(e)}")
            raise
    
    def analyze_class_balance(self, df: pd.DataFrame):
        """Analyze class balance in categorical columns"""
        try:
            categorical_columns = df.select_dtypes(include=['object']).columns
            logger.info(f"Analyzing class balance for {len(categorical_columns)} categorical columns")
            
            for col in tqdm(categorical_columns, desc="Analyzing class balance"):
                logger.info(f"Processing column: {col}")
                
                # Calculate class distribution
                class_dist = df[col].value_counts()
                class_percent = (class_dist / len(df)) * 100
                
                # Log only significant class imbalance
                max_percent = class_percent.max()
                min_percent = class_percent.min()
                if max_percent > 80 or min_percent < 20:
                    logger.info(f"Class imbalance detected in {col}:")
                    logger.info(f"Majority class: {max_percent:.2f}%")
                    logger.info(f"Minority class: {min_percent:.2f}%")
                
                if not self.skip_visualizations:
                    # Skip visualization for columns with too many unique values
                    if len(class_dist) > 50:
                        logger.warning(f"Skipping visualization for {col} - too many unique values ({len(class_dist)})")
                        continue
                    
                    try:
                        # Create visualizations with progress logging
                        logger.info(f"Creating visualization for {col}")
                        plt.figure(figsize=(12, 6))
                        
                        # Use top 20 categories if there are many
                        plot_data = class_dist.head(20)
                        sns.barplot(x=plot_data.index, y=plot_data.values)
                        plt.title(f'Class Distribution - {col} (Top 20)')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        
                        # Save plot
                        plt.savefig(self.analysis_dir / f'class_distribution_{col}.png')
                        logger.info(f"Saved visualization for {col}")
                        plt.close()
                        
                    except Exception as e:
                        logger.error(f"Error creating visualization for {col}: {str(e)}")
                        continue
            
        except Exception as e:
            logger.error(f"Error analyzing class balance: {str(e)}")
            raise
    
    def create_pairplot(self, df: pd.DataFrame):
        """Create pairplot for numeric features"""
        if self.skip_visualizations:
            logger.info("Skipping pairplot creation")
            return
            
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) < 2:
                logger.warning("Not enough numeric columns for pairplot")
                return
                
            logger.info(f"Creating pairplot for {len(numeric_columns)} numeric columns")
            
            # Create pairplot with progress logging
            plt.figure(figsize=(15, 15))
            sns.pairplot(df[numeric_columns], diag_kind='kde')
            
            # Save plot with error handling
            try:
                plt.savefig(self.analysis_dir / 'pairplot.png')
                logger.info("Saved pairplot")
            except Exception as e:
                logger.error(f"Error saving pairplot: {str(e)}")
            finally:
                plt.close()
                
        except Exception as e:
            logger.error(f"Error creating pairplot: {str(e)}")
            raise
    
    def analyze_feature_importance(self, df: pd.DataFrame):
        """Analyze feature importance using correlation analysis"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) < 2:
                logger.warning("Not enough numeric columns for feature importance analysis")
                return
                
            logger.info(f"Analyzing feature importance for {len(numeric_columns)} numeric columns")
            
            # Calculate correlation matrix
            correlation_matrix = df[numeric_columns].corr()
            
            if not self.skip_visualizations:
                # Create heatmap with progress logging
                plt.figure(figsize=(12, 8))
                
                # Convert to pandas if it's a cudf DataFrame
                if hasattr(correlation_matrix, 'to_pandas'):
                    correlation_matrix = correlation_matrix.to_pandas()
                
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Feature Correlation Heatmap')
                
                # Save heatmap with error handling
                try:
                    plt.savefig(self.analysis_dir / 'feature_correlation.png')
                    logger.info("Saved feature correlation heatmap")
                except Exception as e:
                    logger.error(f"Error saving heatmap: {str(e)}")
                finally:
                    plt.close()
            
            # Save correlation matrix
            # Convert to pandas if it's a cudf DataFrame
            if hasattr(correlation_matrix, 'to_pandas'):
                correlation_matrix = correlation_matrix.to_pandas()
            
            correlation_matrix.to_csv(self.analysis_dir / 'feature_correlation.csv')
            logger.info("Saved feature correlation matrix")
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {str(e)}")
            raise
    
    def generate_analysis_report(self, df: pd.DataFrame):
        """Generate a report of the analysis process"""
        try:
            report = {
                'timestamp': datetime.now().timestamp(),
                'total_records': len(df),
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_records': len(df) - len(df.drop_duplicates()),
                'numeric_features': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_features': df.select_dtypes(include=['object']).columns.tolist(),
                'feature_statistics': df.describe().to_dict()
            }
            
            # Save report
            with open(self.analysis_dir / 'analysis_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error generating analysis report: {str(e)}")
            raise
    
    def run_analysis_pipeline(self):
        """Run the complete data analysis pipeline"""
        try:
            logger.info("Starting data analysis pipeline")
            
            # Load data
            df = self.load_data()
            
            # Analyze missing values
            logger.info("Analyzing missing values...")
            df = self.analyze_missing_values(df)
            
            # Handle duplicates
            logger.info("Handling duplicates...")
            df = self.handle_duplicates(df)
            
            # Handle outliers
            logger.info("Handling outliers...")
            df = self.handle_outliers(df)
            
            # Normalize data
            logger.info("Normalizing data...")
            df = self.normalize_data(df)
            
            # Analyze class balance
            logger.info("Analyzing class balance...")
            self.analyze_class_balance(df)
            
            # Create pairplot
            logger.info("Creating pairplot...")
            self.create_pairplot(df)
            
            # Analyze feature importance
            logger.info("Analyzing feature importance...")
            self.analyze_feature_importance(df)
            
            # Generate analysis report
            logger.info("Generating analysis report...")
            self.generate_analysis_report(df)
            
            logger.info("Data analysis pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in analysis pipeline: {str(e)}")
            raise

def main():
    """Main function to run data analysis"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run analysis pipeline with GPU acceleration if available
    analyzer = DataAnalyzer(batch_size=1000, skip_visualizations=False, use_gpu=True)
    analyzer.run_analysis_pipeline()

if __name__ == "__main__":
    main() 
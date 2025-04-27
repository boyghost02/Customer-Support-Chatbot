import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

class DataVisualizer:
    def __init__(self, batch_size: int = 1000, skip_visualizations: bool = False):
        """Initialize data visualizer"""
        self.data_dir = Path("../data")
        self.visualization_dir = self.data_dir / "visualizations"
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.skip_visualizations = skip_visualizations
        
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
            with open(self.data_dir / "processed_data.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            logger.info(f"Loaded {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_distribution_plots(self, df: pd.DataFrame):
        """Create distribution plots for numeric features"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            logger.info(f"Creating distribution plots for {len(numeric_columns)} numeric columns")
            
            for col in tqdm(numeric_columns, desc="Creating distribution plots"):
                # Create histogram with KDE
                plt.figure(figsize=(10, 6))
                sns.histplot(data=df, x=col, kde=True)
                plt.title(f'Distribution of {col}')
                plt.tight_layout()
                plt.savefig(self.visualization_dir / f'distribution_{col}.png')
                plt.close()
                
                # Create interactive histogram
                fig = px.histogram(df, x=col, title=f'Distribution of {col}')
                fig.write_html(self.visualization_dir / f'distribution_{col}.html')
                
        except Exception as e:
            logger.error(f"Error creating distribution plots: {str(e)}")
            raise
    
    def create_box_plots(self, df: pd.DataFrame):
        """Create box plots for numeric features"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            logger.info(f"Creating box plots for {len(numeric_columns)} numeric columns")
            
            # Create static box plot
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df[numeric_columns])
            plt.title('Box Plots of Numeric Features')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.visualization_dir / 'box_plots.png')
            plt.close()
            
            # Create interactive box plot
            fig = px.box(df, y=numeric_columns, title='Box Plots of Numeric Features')
            fig.write_html(self.visualization_dir / 'box_plots.html')
            
        except Exception as e:
            logger.error(f"Error creating box plots: {str(e)}")
            raise
    
    def create_correlation_heatmap(self, df: pd.DataFrame):
        """Create correlation heatmap for numeric features"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) > 0:
                # Calculate correlation matrix
                corr_matrix = df[numeric_columns].corr()
                
                # Create static heatmap
                plt.figure(figsize=(12, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Feature Correlation Heatmap')
                plt.tight_layout()
                plt.savefig(self.visualization_dir / 'correlation_heatmap.png')
                plt.close()
                
                # Create interactive heatmap
                fig = px.imshow(corr_matrix,
                              title='Feature Correlation Heatmap',
                              color_continuous_scale='RdBu')
                fig.write_html(self.visualization_dir / 'correlation_heatmap.html')
                
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {str(e)}")
            raise
    
    def create_pairplot(self, df: pd.DataFrame):
        """Create pair plot for numeric features"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) > 0:
                # Create static pair plot
                sns.pairplot(df[numeric_columns])
                plt.savefig(self.visualization_dir / 'pairplot.png')
                plt.close()
                
                # Create interactive pair plot
                fig = px.scatter_matrix(df, dimensions=numeric_columns)
                fig.write_html(self.visualization_dir / 'pairplot.html')
                
        except Exception as e:
            logger.error(f"Error creating pair plot: {str(e)}")
            raise
    
    def create_categorical_plots(self, df: pd.DataFrame):
        """Create plots for categorical features"""
        try:
            categorical_columns = df.select_dtypes(include=['object']).columns
            logger.info(f"Creating categorical plots for {len(categorical_columns)} categorical columns")
            
            for col in tqdm(categorical_columns, desc="Creating categorical plots"):
                # Skip columns with too many unique values
                if len(df[col].unique()) > 50:
                    logger.warning(f"Skipping visualization for {col} - too many unique values ({len(df[col].unique())})")
                    continue
                
                # Calculate value counts
                value_counts = df[col].value_counts()
                
                # Create bar plot
                plt.figure(figsize=(12, 6))
                sns.barplot(x=value_counts.values, y=value_counts.index)
                plt.title(f'Distribution of {col}')
                plt.tight_layout()
                plt.savefig(self.visualization_dir / f'categorical_{col}_bar.png')
                plt.close()
                
                # Create pie chart
                plt.figure(figsize=(8, 8))
                plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
                plt.title(f'Distribution of {col}')
                plt.savefig(self.visualization_dir / f'categorical_{col}_pie.png')
                plt.close()
                
                # Create interactive plots
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f'Distribution of {col}')
                fig.write_html(self.visualization_dir / f'categorical_{col}_bar.html')
                
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f'Distribution of {col}')
                fig.write_html(self.visualization_dir / f'categorical_{col}_pie.html')
                
        except Exception as e:
            logger.error(f"Error creating categorical plots: {str(e)}")
            raise
    
    def create_time_series_plots(self, df: pd.DataFrame):
        """Create time series plots if timestamp data is available"""
        try:
            # Check for timestamp columns
            timestamp_columns = df.select_dtypes(include=['datetime64']).columns
            
            if len(timestamp_columns) > 0:
                for col in timestamp_columns:
                    # Create time series plot
                    plt.figure(figsize=(12, 6))
                    df.set_index(col).resample('D').count().plot()
                    plt.title(f'Time Series of {col}')
                    plt.tight_layout()
                    plt.savefig(self.visualization_dir / f'time_series_{col}.png')
                    plt.close()
                    
                    # Create interactive time series plot
                    fig = px.line(df.set_index(col).resample('D').count(),
                                title=f'Time Series of {col}')
                    fig.write_html(self.visualization_dir / f'time_series_{col}.html')
                    
        except Exception as e:
            logger.error(f"Error creating time series plots: {str(e)}")
            raise
    
    def create_scatter_plots(self, df: pd.DataFrame):
        """Create scatter plots for numeric features"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) >= 2:
                # Create scatter plot matrix
                fig = px.scatter_matrix(df, dimensions=numeric_columns[:5])  # Limit to 5 features
                fig.write_html(self.visualization_dir / 'scatter_matrix.html')
                
                # Create individual scatter plots
                for i in range(len(numeric_columns)-1):
                    for j in range(i+1, len(numeric_columns)):
                        fig = px.scatter(df, x=numeric_columns[i], y=numeric_columns[j],
                                       title=f'Scatter Plot: {numeric_columns[i]} vs {numeric_columns[j]}')
                        fig.write_html(self.visualization_dir / f'scatter_{numeric_columns[i]}_{numeric_columns[j]}.html')
                        
        except Exception as e:
            logger.error(f"Error creating scatter plots: {str(e)}")
            raise
    
    def run_visualization_pipeline(self):
        """Run the complete visualization pipeline"""
        try:
            logger.info("Starting visualization pipeline")
            
            # Load data
            df = self.load_data()
            
            # Create visualizations with progress tracking
            logger.info("Creating distribution plots...")
            self.create_distribution_plots(df)
            
            logger.info("Creating box plots...")
            self.create_box_plots(df)
            
            logger.info("Creating correlation heatmap...")
            self.create_correlation_heatmap(df)
            
            logger.info("Creating pairplot...")
            self.create_pairplot(df)
            
            logger.info("Creating categorical plots...")
            self.create_categorical_plots(df)
            
            logger.info("Creating time series plots...")
            self.create_time_series_plots(df)
            
            logger.info("Creating scatter plots...")
            self.create_scatter_plots(df)
            
            logger.info("Data visualization pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in visualization pipeline: {str(e)}")
            raise

def main():
    """Main function to run data visualization"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run visualization pipeline
    visualizer = DataVisualizer()
    visualizer.run_visualization_pipeline()

if __name__ == "__main__":
    main() 
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import re
import string
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    """
    Analyzes customer support data for the chatbot.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the DataAnalyzer.
        
        Args:
            data_dir: Directory containing the data
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.analyzed_dir = os.path.join(data_dir, "analyzed")
        self.visualizations_dir = os.path.join(data_dir, "visualizations")
        
        # Create directories if they don't exist
        for directory in [self.processed_dir, self.analyzed_dir, self.visualizations_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Define predefined categories
        self.categories = [
            "order_tracking", 
            "product_info", 
            "returns", 
            "shipping", 
            "payment", 
            "account", 
            "product_recommendation",
            "general"
        ]
        
        # Define stopwords
        self.stopwords = set([
            "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
            "when", "where", "how", "who", "which", "this", "that", "these", "those",
            "then", "just", "so", "than", "such", "both", "through", "about", "for",
            "is", "of", "while", "during", "to", "from", "in", "out", "on", "off",
            "again", "further", "then", "once", "here", "there", "all", "any", "both",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "can", "will", "just",
            "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn",
            "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
            "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn", "i", "me", "my",
            "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
            "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
            "it", "its", "itself", "they", "them", "their", "theirs", "themselves"
        ])
        
        # Vietnamese stopwords
        self.vietnamese_stopwords = set([
            "và", "của", "cho", "là", "để", "trong", "đến", "với", "các", "những",
            "được", "tại", "có", "không", "này", "khi", "từ", "bạn", "tôi", "chúng",
            "mình", "đã", "sẽ", "đang", "về", "như", "nhưng", "nếu", "vì", "một",
            "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười"
        ])
        
        logger.info(f"DataAnalyzer initialized with data directory: {data_dir}")
    
    def load_data(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            filename: Name of the file to load. If None, loads the most recent combined data file.
            
        Returns:
            DataFrame containing the loaded data
        """
        if filename is None:
            # Find the most recent combined data file
            files = [f for f in os.listdir(self.raw_dir) if f.startswith("combined_data_")]
            if not files:
                raise FileNotFoundError("No combined data files found")
            
            files.sort(reverse=True)  # Sort by date (newest first)
            filename = files[0]
        
        file_path = os.path.join(self.raw_dir, filename)
        logger.info(f"Loading data from {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} records")
        
        return df
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        
        # Remove numbers
        text = re.sub(r"\d+", "", text)
        
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        # Detect language (simple approach)
        is_vietnamese = any(char in "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ" for char in text)
        
        # Remove stopwords
        if is_vietnamese:
            words = [word for word in text.split() if word not in self.vietnamese_stopwords]
        else:
            words = [word for word in text.split() if word not in self.stopwords]
        
        return " ".join(words)
    
    def analyze_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze the data.
        
        Args:
            df: DataFrame containing the data to analyze
            
        Returns:
            DataFrame with analysis results
        """
        logger.info("Starting data analysis")
        
        # Make a copy to avoid modifying the original
        df_analyzed = df.copy()
        
        # Clean text
        logger.info("Cleaning text")
        df_analyzed["question_clean"] = df_analyzed["question"].apply(self.clean_text)
        df_analyzed["answer_clean"] = df_analyzed["answer"].apply(self.clean_text)
        
        # Calculate text length
        logger.info("Calculating text length")
        df_analyzed["question_length"] = df_analyzed["question"].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
        df_analyzed["answer_length"] = df_analyzed["answer"].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
        
        # Extract entities (simplified approach)
        logger.info("Extracting entities")
        df_analyzed["entities"] = df_analyzed["question"].apply(self.extract_entities)
        
        # Cluster uncategorized data
        if "category" not in df_analyzed.columns:
            logger.info("Clustering uncategorized data")
            df_analyzed["category"] = self.cluster_data(df_analyzed)
        
        # Save the analyzed data
        output_path = os.path.join(self.analyzed_dir, f"analyzed_data_{datetime.now().strftime('%Y%m%d')}.csv")
        df_analyzed.to_csv(output_path, index=False)
        logger.info(f"Saved analyzed data to {output_path}")
        
        return df_analyzed
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from text (simplified approach).
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entities
        """
        if not isinstance(text, str):
            return []
        
        entities = []
        
        # Extract order numbers
        order_matches = re.findall(r"order\s+#?(\d+)", text.lower())
        if order_matches:
            entities.extend([f"ORDER_ID:{match}" for match in order_matches])
        
        # Extract product names (simplified)
        product_patterns = [
            r"laptop", r"computer", r"phone", r"tablet", r"camera", r"headphone",
            r"speaker", r"monitor", r"keyboard", r"mouse", r"printer", r"scanner"
        ]
        
        for pattern in product_patterns:
            if re.search(pattern, text.lower()):
                entities.append(f"PRODUCT_TYPE:{pattern}")
        
        return entities
    
    def cluster_data(self, df: pd.DataFrame) -> List[str]:
        """
        Cluster data into categories.
        
        Args:
            df: DataFrame containing the data to cluster
            
        Returns:
            List of assigned categories
        """
        # Combine question and answer for better clustering
        texts = df["question_clean"] + " " + df["answer_clean"]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Cluster the data
        num_clusters = len(self.categories)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        # Map cluster IDs to categories
        cluster_to_category = {}
        for i in range(num_clusters):
            # Get the most common words in this cluster
            cluster_docs = [j for j, cluster in enumerate(clusters) if cluster == i]
            if not cluster_docs:
                continue
                
            cluster_tfidf = tfidf_matrix[cluster_docs]
            
            # Get the top terms for this cluster
            top_terms_idx = cluster_tfidf.sum(axis=0).argsort()[0, -10:].tolist()[0]
            top_terms = [term for term, idx in vectorizer.vocabulary_.items() if idx in top_terms_idx]
            
            # Assign category based on top terms
            assigned_category = "general"
            for category in self.categories:
                category_terms = category.split("_")
                if any(term in top_terms for term in category_terms):
                    assigned_category = category
                    break
            
            cluster_to_category[i] = assigned_category
        
        # Assign categories to data points
        categories = [cluster_to_category.get(cluster, "general") for cluster in clusters]
        
        return categories
    
    def visualize_data(self, df: pd.DataFrame) -> None:
        """
        Create visualizations of the data.
        
        Args:
            df: DataFrame containing the data to visualize
        """
        logger.info("Creating visualizations")
        
        # Ensure the visualizations directory exists
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
        # 1. Category distribution
        plt.figure(figsize=(12, 6))
        category_counts = df["category"].value_counts()
        sns.barplot(x=category_counts.values, y=category_counts.index)
        plt.title("Distribution of Categories")
        plt.xlabel("Count")
        plt.ylabel("Category")
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, "category_distribution.png"))
        plt.close()
        
        # 2. Source distribution (if available)
        if "source" in df.columns:
            plt.figure(figsize=(10, 6))
            source_counts = df["source"].value_counts()
            plt.pie(source_counts.values, labels=source_counts.index, autopct="%1.1f%%", startangle=90)
            plt.axis("equal")
            plt.title("Distribution of Data Sources")
            plt.savefig(os.path.join(self.visualizations_dir, "source_distribution.png"))
            plt.close()
        
        # 3. Text length distribution
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(df["question_length"], kde=True)
        plt.title("Question Length Distribution")
        plt.xlabel("Number of Words")
        plt.ylabel("Frequency")
        
        plt.subplot(1, 2, 2)
        sns.histplot(df["answer_length"], kde=True)
        plt.title("Answer Length Distribution")
        plt.xlabel("Number of Words")
        plt.ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, "text_length_distribution.png"))
        plt.close()
        
        # 4. Topic clusters visualization
        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(df["question_clean"] + " " + df["answer_clean"])
            
            # Apply dimensionality reduction
            # First reduce with PCA to 50 dimensions for efficiency
            pca = PCA(n_components=min(50, tfidf_matrix.shape[1] - 1))
            reduced_features = pca.fit_transform(tfidf_matrix.toarray())
            
            # Then apply t-SNE for visualization
            tsne = TSNE(n_components=2, random_state=42)
            tsne_features = tsne.fit_transform(reduced_features)
            
            # Create a DataFrame for plotting
            plot_df = pd.DataFrame({
                "x": tsne_features[:, 0],
                "y": tsne_features[:, 1],
                "category": df["category"]
            })
            
            # Plot the clusters
            plt.figure(figsize=(14, 10))
            
            # Get unique categories and assign colors
            categories = plot_df["category"].unique()
            
            # Create a scatter plot for each category
            for i, category in enumerate(categories):
                category_data = plot_df[plot_df["category"] == category]
                plt.scatter(category_data["x"], category_data["y"], label=category, alpha=0.7)
            
            plt.title("Topic Clusters Visualization")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(os.path.join(self.visualizations_dir, "topic_clusters.png"))
            plt.close()
        
        except Exception as e:
            logger.error(f"Error creating topic clusters visualization: {e}")
        
        logger.info(f"Saved visualizations to {self.visualizations_dir}")
    
    def prepare_for_langchain(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """
        Prepare the data for use in LangChain.
        
        Args:
            df: DataFrame containing the analyzed data
            
        Returns:
            Dictionary mapping categories to lists of Q&A pairs
        """
        logger.info("Preparing data for LangChain")
        
        # Group data by category
        category_data = {}
        
        for category in df["category"].unique():
            category_df = df[df["category"] == category]
            
            qa_pairs = []
            for _, row in category_df.iterrows():
                qa_pair = {
                    "question": row["question"],
                    "answer": row["answer"],
                    "metadata": {
                        "source": row.get("source", "unknown"),
                        "category": category
                    }
                }
                
                # Add entities if available
                if "entities" in row and row["entities"]:
                    qa_pair["metadata"]["entities"] = row["entities"]
                
                qa_pairs.append(qa_pair)
            
            category_data[category] = qa_pairs
        
        # Save the prepared data
        output_path = os.path.join(self.processed_dir, "langchain_data.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(category_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved LangChain data to {output_path}")
        
        return category_data

# Example usage
if __name__ == "__main__":
    analyzer = DataAnalyzer()
    
    try:
        df = analyzer.load_data()
        df_analyzed = analyzer.analyze_data(df)
        analyzer.visualize_data(df_analyzed)
        analyzer.prepare_for_langchain(df_analyzed)
    except FileNotFoundError:
        logger.error("No data files found. Please run the data collector first.")

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
import os
from dotenv import load_dotenv
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Try to import LangChain components with error handling
try:
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores import Chroma
    from langchain_community.document_loaders import JSONLoader
    LANGCHAIN_AVAILABLE = True
    logger.info("LangChain components imported successfully")
except ImportError as e:
    logger.error(f"Error importing LangChain components: {str(e)}")
    LANGCHAIN_AVAILABLE = False

class LangChainDataPreparator:
    def __init__(self, batch_size: int = 1000):
        """Initialize data preparator"""
        self.data_dir = Path(__file__).parent.parent / "data"
        self.langchain_dir = self.data_dir / "langchain"
        self.langchain_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        
        # Initialize embeddings
        self.embeddings = self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embeddings with error handling"""
        try:
            # Try OpenAI embeddings first
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key and openai_key != "your-openai-api-key":
                try:
                    embeddings = OpenAIEmbeddings()
                    logger.info("Initialized OpenAI embeddings")
                    return embeddings
                except Exception as e:
                    logger.warning(f"Could not initialize OpenAI embeddings: {str(e)}")
                    logger.info("Falling back to HuggingFace embeddings")
            
            # Fallback to HuggingFace embeddings
            try:
                logger.info("Initializing HuggingFace embeddings...")
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},  # Force CPU to avoid CUDA issues
                    encode_kwargs={'normalize_embeddings': True}  # Normalize embeddings for better similarity search
                )
                logger.info("Initialized HuggingFace embeddings successfully")
                return embeddings
            except Exception as e:
                logger.error(f"Could not initialize HuggingFace embeddings: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise
    
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
    
    def prepare_documents(self, df: pd.DataFrame) -> List[Document]:
        """Prepare documents for vector store"""
        try:
            documents = []
            for _, row in df.iterrows():
                # Create text content
                text_content = f"""
                Instruction: {row['instruction']}
                Response: {row['response']}
                Category: {row['category']}
                Intent: {row['intent']}
                """
                
                # Create metadata
                metadata = {
                    'category': row['category'],
                    'intent': row['intent'],
                    'flags': row.get('flags', [])
                }
                
                # Create Document object
                doc = Document(
                    page_content=text_content.strip(),
                    metadata=metadata
                )
                documents.append(doc)
            
            logger.info(f"Created {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error preparing documents: {str(e)}")
            raise
    
    def create_vector_store(self, documents: List[Document], store_type: str = 'faiss'):
        """Create vector store from documents"""
        try:
            if store_type == 'faiss':
                vector_store = FAISS.from_documents(documents, self.embeddings)
                vector_store.save_local(self.langchain_dir / 'faiss_index')
                logger.info("Created and saved FAISS vector store")
            elif store_type == 'chroma':
                vector_store = Chroma.from_documents(
                    documents,
                    self.embeddings,
                    persist_directory=str(self.langchain_dir / 'chroma_db')
                )
                vector_store.persist()
                logger.info("Created and saved Chroma vector store")
            else:
                raise ValueError(f"Unsupported vector store type: {store_type}")
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def run_preparation_pipeline(self):
        """Run the complete data preparation pipeline"""
        try:
            if not LANGCHAIN_AVAILABLE:
                raise ImportError("LangChain components are not available")
            
            logger.info("Starting LangChain data preparation pipeline")
            
            # Load data
            df = self.load_data()
            
            # Prepare documents
            logger.info("Preparing documents...")
            documents = self.prepare_documents(df)
            
            # Create vector stores
            logger.info("Creating vector stores...")
            self.create_vector_store(documents, 'faiss')
            self.create_vector_store(documents, 'chroma')
            
            logger.info("LangChain data preparation pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in preparation pipeline: {str(e)}")
            raise

def main():
    """Main function to run data preparation pipeline"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run preparation pipeline
    preparator = LangChainDataPreparator(batch_size=1000)
    preparator.run_preparation_pipeline()

if __name__ == "__main__":
    main() 
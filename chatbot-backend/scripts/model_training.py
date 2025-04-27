import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from datasets import Dataset
from typing import Dict, Any, Tuple
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        """Initialize model trainer"""
        self.data_dir = Path(__file__).parent.parent / "data"
        self.model_dir = self.data_dir / "models"
        self.results_dir = self.data_dir / "model_results"
        
        # Create necessary directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configuration
        self.model_name = "t5-base"
        self.max_length = 80
        self.batch_size = 16  # Increased batch size for GPU
        self.epochs = 5
        self.learning_rate = 3e-5
        
        # Check GPU availability and display detailed information
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            # Get GPU information
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            
            logger.info("GPU Information:")
            logger.info(f"GPU Name: {gpu_name}")
            logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
            logger.info(f"Number of GPUs: {gpu_count}")
            logger.info(f"Current GPU Device: {current_device}")
            
            # Test GPU with a small tensor
            try:
                test_tensor = torch.randn(1000, 1000).cuda()
                logger.info("GPU test successful - CUDA is working properly")
            except Exception as e:
                logger.error(f"GPU test failed: {str(e)}")
        else:
            logger.warning("No GPU available. Training will be slower on CPU.")
    
    def load_preprocessed_data(self) -> pd.DataFrame:
        """Load preprocessed data"""
        try:
            with open(self.data_dir / "preprocessed_data.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            logger.info(f"Loaded {len(df)} preprocessed records")
            return df
        except Exception as e:
            logger.error(f"Error loading preprocessed data: {str(e)}")
            raise
    
    def prepare_dataset(self, df: pd.DataFrame) -> Tuple[Dataset, Dataset]:
        """Prepare datasets for training"""
        try:
            # Split data
            train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
            
            # Convert to HuggingFace datasets
            train_dataset = Dataset.from_pandas(train_df)
            val_dataset = Dataset.from_pandas(val_df)
            
            logger.info(f"Prepared datasets: {len(train_dataset)} training, {len(val_dataset)} validation")
            return train_dataset, val_dataset
            
        except Exception as e:
            logger.error(f"Error preparing datasets: {str(e)}")
            raise
    
    def tokenize_function(self, examples):
        """Tokenize text data"""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Get unique intents and create label mapping
        unique_intents = sorted(set(examples["intent"]))
        label2id = {label: idx for idx, label in enumerate(unique_intents)}
        
        # Convert labels to integers
        labels = [label2id[label] for label in examples["intent"]]
        
        # Tokenize text
        tokenized = tokenizer(
            examples["instruction"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=None  # Return lists instead of tensors
        )
        
        # Add labels to tokenized output
        tokenized["labels"] = labels
        return tokenized
    
    def train_intent_classifier(self, train_dataset: Dataset, val_dataset: Dataset):
        """Train intent classification model"""
        try:
            # Initialize tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Get unique intents and create label mapping
            unique_intents = sorted(train_dataset.unique("intent"))
            label2id = {label: idx for idx, label in enumerate(unique_intents)}
            id2label = {idx: label for label, idx in label2id.items()}
            
            # Initialize model with label mapping
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(unique_intents),
                id2label=id2label,
                label2id=label2id
            )
            
            # Move model to GPU if available and verify
            model = model.to(self.device)
            logger.info(f"Model moved to {self.device}")
            if torch.cuda.is_available():
                logger.info(f"Model is on GPU: {next(model.parameters()).is_cuda}")
            
            # Prepare datasets
            train_dataset = train_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=train_dataset.column_names
            )
            val_dataset = val_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=val_dataset.column_names
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(self.model_dir / "intent_classifier"),
                num_train_epochs=self.epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                weight_decay=0.01,
                logging_dir=str(self.results_dir / "logs"),
                logging_steps=100,
                eval_steps=100,
                save_steps=100,
                save_total_limit=2,
                # GPU specific settings
                no_cuda=False,  # Enable CUDA
                fp16=True,  # Enable mixed precision training
                gradient_accumulation_steps=4,  # Gradient accumulation for larger effective batch size
                dataloader_num_workers=4,  # Number of data loading workers
                dataloader_pin_memory=True,  # Pin memory for faster data transfer to GPU
                # Performance optimization
                optim="adamw_torch",  # Use PyTorch's AdamW optimizer
                warmup_ratio=0.1,  # Warmup for better training stability
                lr_scheduler_type="cosine",  # Cosine learning rate schedule
                report_to="tensorboard"  # Enable tensorboard logging
            )
            
            # Log training configuration
            logger.info("Training Configuration:")
            logger.info(f"Using device: {training_args.device}")
            logger.info(f"Mixed precision training: {training_args.fp16}")
            logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
            logger.info(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
            )
            
            # Train model
            trainer.train()
            
            # Save model and tokenizer
            trainer.save_model()
            tokenizer.save_pretrained(self.model_dir / "intent_classifier")
            
            # Save label mapping
            with open(self.model_dir / "intent_classifier" / "label_mapping.json", 'w') as f:
                json.dump({
                    "label2id": label2id,
                    "id2label": id2label
                }, f, indent=2)
            
            logger.info("Intent classifier training completed")
            return trainer  # Return the trainer object
            
        except Exception as e:
            logger.error(f"Error training intent classifier: {str(e)}")
            raise
    
    def prepare_langchain_data(self, df: pd.DataFrame):
        """Prepare data for Langchain"""
        try:
            # Group by intent and create examples
            langchain_data = {}
            for intent in df['intent'].unique():
                examples = df[df['intent'] == intent]['instruction'].tolist()
                langchain_data[intent] = examples
            
            # Save Langchain data
            with open(self.data_dir / "langchain" / "training_data.json", 'w') as f:
                json.dump(langchain_data, f, indent=2)
            
            logger.info("Langchain data preparation completed")
            
        except Exception as e:
            logger.error(f"Error preparing Langchain data: {str(e)}")
            raise
    
    def analyze_results(self, trainer, train_dataset, val_dataset):
        """Analyze training results and generate performance metrics"""
        try:
            # Get training metrics from log history
            log_history = trainer.state.log_history
            
            # Extract metrics safely with default values
            training_metrics = {
                'epoch': log_history[-1].get('epoch', 0),
                'loss': log_history[-1].get('loss', 0),
                'learning_rate': log_history[-1].get('learning_rate', 0),
                'training_time': log_history[-1].get('train_runtime', 0),
                'samples_per_second': log_history[-1].get('train_samples_per_second', 0),
                'steps_per_second': log_history[-1].get('train_steps_per_second', 0)
            }
            
            # Get evaluation metrics
            eval_metrics = {
                'eval_loss': log_history[-1].get('eval_loss', 0),
                'eval_runtime': log_history[-1].get('eval_runtime', 0),
                'eval_samples_per_second': log_history[-1].get('eval_samples_per_second', 0),
                'eval_steps_per_second': log_history[-1].get('eval_steps_per_second', 0)
            }
            
            # Prepare validation dataset for predictions
            val_dataset = val_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=val_dataset.column_names
            )
            
            # Get predictions
            predictions = trainer.predict(val_dataset)
            # Handle T5 model predictions
            if isinstance(predictions.predictions, tuple):
                # T5 returns a tuple of (logits, past_key_values)
                logits = predictions.predictions[0]
            else:
                logits = predictions.predictions
                
            # Convert to numpy and get predictions
            preds = np.argmax(logits, axis=-1)
            labels = predictions.label_ids
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(labels, preds),
                'precision': precision_score(labels, preds, average='weighted'),
                'recall': recall_score(labels, preds, average='weighted'),
                'f1': f1_score(labels, preds, average='weighted'),
                'confusion_matrix': confusion_matrix(labels, preds).tolist()
            }
            
            # Combine all metrics
            results = {
                'training_metrics': training_metrics,
                'evaluation_metrics': eval_metrics,
                'performance_metrics': metrics,
                'timestamp': datetime.now().timestamp()
            }
            
            # Save results
            results_dir = Path("../data/training_results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            with open(results_dir / 'training_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info("Training results analysis completed")
            
        except Exception as e:
            logger.error(f"Error analyzing results: {str(e)}")
            raise
    
    def run_pipeline(self):
        """Run the complete model training pipeline"""
        try:
            # Load preprocessed data
            df = self.load_preprocessed_data()
            
            # Prepare datasets
            train_dataset, val_dataset = self.prepare_dataset(df)
            
            # Train intent classifier and get trainer
            trainer = self.train_intent_classifier(train_dataset, val_dataset)
            
            # Prepare Langchain data
            self.prepare_langchain_data(df)
            
            # Analyze results
            self.analyze_results(trainer, train_dataset, val_dataset)
            
            logger.info("Model training pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            raise

def main():
    """Main function to run model training"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run training pipeline
    trainer = ModelTrainer()
    trainer.run_pipeline()

if __name__ == "__main__":
    main() 
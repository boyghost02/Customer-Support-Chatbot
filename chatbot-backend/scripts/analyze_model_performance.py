import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from datetime import datetime
import numpy as np

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

def load_training_results():
    """Load training results from JSON file"""
    results_path = Path(__file__).parent.parent / "data" / "model_results" / "training_results.json"
    if not results_path.exists():
        print("Error: Training results file not found!")
        print(f"Expected path: {results_path}")
        return None
    
    with open(results_path, 'r') as f:
        return json.load(f)

def plot_loss_curves(results):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(results['training_loss_history']) + 1)
    
    plt.plot(epochs, results['training_loss_history'], 'b-', label='Training Loss')
    plt.plot(epochs, results['validation_loss_history'], 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig('loss_curves.png')
    plt.close()

def plot_accuracy_curves(results):
    """Plot training and validation accuracy curves"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(results['training_accuracy_history']) + 1)
    
    plt.plot(epochs, results['training_accuracy_history'], 'b-', label='Training Accuracy')
    plt.plot(epochs, results['validation_accuracy_history'], 'r-', label='Validation Accuracy')
    
    plt.title('Training and Validation Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig('accuracy_curves.png')
    plt.close()

def plot_confusion_matrix(results):
    """Plot confusion matrix"""
    plt.figure(figsize=(12, 8))
    cm = np.array(results['confusion_matrix'])
    
    # Create DataFrame for better visualization
    cm_df = pd.DataFrame(cm, 
                        index=results['class_names'],
                        columns=results['class_names'])
    
    # Plot heatmap
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save plot
    plt.savefig('confusion_matrix.png')
    plt.close()

def analyze_performance():
    """Analyze and display model performance metrics"""
    results = load_training_results()
    if not results:
        return
    
    print("\n=== Model Performance Analysis ===")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Basic Metrics
    print("\n1. Basic Metrics:")
    print(f"Final Training Loss: {results['training_loss']:.4f}")
    print(f"Final Validation Loss: {results['validation_loss']:.4f}")
    print(f"Final Training Accuracy: {results['training_accuracy']:.2f}%")
    print(f"Final Validation Accuracy: {results['validation_accuracy']:.2f}%")
    
    # Learning Progress
    print("\n2. Learning Progress:")
    print(f"Best Validation Accuracy: {max(results['validation_accuracy_history']):.2f}%")
    print(f"Best Validation Loss: {min(results['validation_loss_history']):.4f}")
    print(f"Training Time: {results['training_time']:.2f} seconds")
    
    # Class-wise Performance
    print("\n3. Class-wise Performance:")
    for i, class_name in enumerate(results['class_names']):
        print(f"\n{class_name}:")
        print(f"  Precision: {results['class_metrics'][i]['precision']:.4f}")
        print(f"  Recall: {results['class_metrics'][i]['recall']:.4f}")
        print(f"  F1-score: {results['class_metrics'][i]['f1']:.4f}")
    
    # Dataset Statistics
    print("\n4. Dataset Statistics:")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Training Samples: {results['train_samples']}")
    print(f"Validation Samples: {results['val_samples']}")
    print(f"Number of Classes: {results['num_classes']}")
    
    # Training Configuration
    print("\n5. Training Configuration:")
    print(f"Model: {results['model_name']}")
    print(f"Batch Size: {results['batch_size']}")
    print(f"Learning Rate: {results['learning_rate']}")
    print(f"Epochs: {results['epochs']}")
    print(f"Max Sequence Length: {results['max_length']}")
    
    # Generate plots
    print("\nGenerating performance plots...")
    plot_loss_curves(results)
    plot_accuracy_curves(results)
    plot_confusion_matrix(results)
    print("Plots have been saved as:")
    print("- loss_curves.png")
    print("- accuracy_curves.png")
    print("- confusion_matrix.png")

if __name__ == "__main__":
    analyze_performance() 
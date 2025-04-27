import json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_results():
    """Load training results from JSON file"""
    results_path = Path("../data/training_results/training_results.json")
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_path}")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in results file")
        return None

def print_training_metrics(metrics):
    """Print training metrics"""
    print("\nTraining Metrics:")
    print(f"Epoch: {metrics['epoch']:.2f}")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Learning Rate: {metrics['learning_rate']:.2e}")
    print(f"Training Time: {metrics['training_time']:.2f} seconds")
    print(f"Samples per Second: {metrics['samples_per_second']:.2f}")
    print(f"Steps per Second: {metrics['steps_per_second']:.2f}")

def print_evaluation_metrics(metrics):
    """Print evaluation metrics"""
    print("\nEvaluation Metrics:")
    print(f"Validation Loss: {metrics['eval_loss']:.4f}")
    print(f"Evaluation Time: {metrics['eval_runtime']:.2f} seconds")
    print(f"Evaluation Samples per Second: {metrics['eval_samples_per_second']:.2f}")
    print(f"Evaluation Steps per Second: {metrics['eval_steps_per_second']:.2f}")

def print_performance_metrics(metrics):
    """Print performance metrics"""
    print("\nPerformance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

def plot_confusion_matrix(confusion_matrix, labels=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if labels:
        plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45)
        plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    """Main function to display training results"""
    print("\n=== Model Training Results ===\n")
    
    # Load results
    results = load_results()
    if not results:
        return
    
    # Print timestamp
    print(f"Training Date: {datetime.fromtimestamp(results['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print metrics
    print_training_metrics(results['training_metrics'])
    print_evaluation_metrics(results['evaluation_metrics'])
    print_performance_metrics(results['performance_metrics'])
    
    # Plot confusion matrix
    if 'confusion_matrix' in results['performance_metrics']:
        # Load label mapping
        label_mapping_path = Path("../data/models/intent_classifier/label_mapping.json")
        if label_mapping_path.exists():
            with open(label_mapping_path, 'r') as f:
                label_mapping = json.load(f)
                id2label = label_mapping['id2label']
                labels = [id2label[str(i)] for i in range(len(id2label))]
        else:
            labels = None
        
        plot_confusion_matrix(
            np.array(results['performance_metrics']['confusion_matrix']),
            labels=labels
        )
        print("\nConfusion matrix plot saved as 'confusion_matrix.png'")
    else:
        print("\nNo confusion matrix data available")

if __name__ == "__main__":
    main() 
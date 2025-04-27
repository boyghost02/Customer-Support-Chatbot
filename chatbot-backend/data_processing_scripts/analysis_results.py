import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from IPython.display import Image

def setup_directories():
    """Define directories for analysis"""
    analysis_dir = '../data/analysis'
    viz_dir = '../data/visualizations'
    model_dir = '../data/model_results'
    langchain_dir = '../data/langchain'
    return analysis_dir, viz_dir, model_dir, langchain_dir

def load_analysis_results(analysis_dir: str) -> dict:
    """Load analysis results from JSON file"""
    with open(os.path.join(analysis_dir, 'analysis_results.json'), 'r', encoding='utf-8') as f:
        return json.load(f)

def load_visualization_metadata(viz_dir: str) -> dict:
    """Load visualization metadata from JSON file"""
    with open(os.path.join(viz_dir, 'visualization_metadata.json'), 'r', encoding='utf-8') as f:
        return json.load(f)

def load_model_results(model_dir: str) -> dict:
    """Load model results from JSON file"""
    with open(os.path.join(model_dir, 'model_results.json'), 'r', encoding='utf-8') as f:
        return json.load(f)

def load_preparation_metadata(langchain_dir: str) -> dict:
    """Load preparation metadata from JSON file"""
    with open(os.path.join(langchain_dir, 'preparation_metadata.json'), 'r', encoding='utf-8') as f:
        return json.load(f)

def display_analysis_results(analysis_results: dict):
    """Display key analysis results"""
    print("Dataset Overview:")
    print(f"Shape: {analysis_results['dataset_overview']['shape']}")
    print(f"Columns: {analysis_results['dataset_overview']['columns']}")
    
    print("\nBasic Statistics:")
    print(pd.DataFrame(analysis_results['basic_statistics']))
    
    print("\nCategory Distribution:")
    print(pd.Series(analysis_results['category_distribution']))
    
    print("\nIntent Distribution (Top 10):")
    print(pd.Series(analysis_results['intent_distribution']).head(10))

def display_visualization_metadata(viz_metadata: dict):
    """Display visualization metadata"""
    print("\nLength Statistics:")
    print(pd.DataFrame(viz_metadata['length_statistics']))

def display_model_results(model_results: dict):
    """Display model performance results"""
    print("\nCategory Classification Results:")
    print(pd.DataFrame(model_results['category_classification']))
    
    print("\nIntent Classification Results:")
    print(pd.DataFrame(model_results['intent_classification']))

def display_preparation_metadata(prep_metadata: dict):
    """Display data preparation metadata"""
    print("\nData Preparation Overview:")
    print(f"Dataset Size: {prep_metadata['dataset_size']}")
    print(f"Number of Categories: {len(prep_metadata['categories'])}")
    print(f"Number of Intents: {len(prep_metadata['intents'])}")
    print(f"Few-shot Examples: {prep_metadata['few_shot_examples_count']}")
    print(f"Training Examples: {prep_metadata['training_data_count']}")
    
    print("\nAverage Lengths:")
    print(f"Instruction: {prep_metadata['average_instruction_length']:.2f} characters")
    print(f"Response: {prep_metadata['average_response_length']:.2f} characters")

def generate_summary(analysis_results: dict, viz_metadata: dict,
                    model_results: dict, prep_metadata: dict) -> dict:
    """Generate a comprehensive summary of the analysis"""
    summary = {
        'dataset_summary': {
            'size': analysis_results['dataset_overview']['shape'][0],
            'categories': len(analysis_results['category_distribution']),
            'intents': len(analysis_results['intent_distribution'])
        },
        'model_performance': {
            'category_accuracy': model_results['category_classification']['accuracy'],
            'intent_accuracy': model_results['intent_classification']['accuracy']
        },
        'data_characteristics': {
            'avg_instruction_length': viz_metadata['length_statistics']['instruction']['mean'],
            'avg_response_length': viz_metadata['length_statistics']['response']['mean']
        },
        'recommendations': [
            'Consider data augmentation for underrepresented categories',
            'Implement ensemble methods to improve model performance',
            'Add more few-shot examples for complex intents',
            'Regularly update the training data with new examples'
        ]
    }
    return summary

def save_summary(summary: dict, analysis_dir: str):
    """Save analysis summary to JSON file"""
    with open(os.path.join(analysis_dir, 'analysis_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

def main():
    """Main function to analyze results"""
    # Setup
    analysis_dir, viz_dir, model_dir, langchain_dir = setup_directories()
    
    # Load results
    analysis_results = load_analysis_results(analysis_dir)
    viz_metadata = load_visualization_metadata(viz_dir)
    model_results = load_model_results(model_dir)
    prep_metadata = load_preparation_metadata(langchain_dir)
    
    # Display results
    display_analysis_results(analysis_results)
    display_visualization_metadata(viz_metadata)
    display_model_results(model_results)
    display_preparation_metadata(prep_metadata)
    
    # Generate and save summary
    summary = generate_summary(analysis_results, viz_metadata, model_results, prep_metadata)
    print("\nAnalysis Summary:")
    print(json.dumps(summary, indent=2))
    
    save_summary(summary, analysis_dir)
    print(f"\nSummary has been saved to: {analysis_dir}")

if __name__ == "__main__":
    main() 
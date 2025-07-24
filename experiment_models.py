"""
Comprehensive Experiment Script for Attention Models
===================================================
Analyzes trained baseline, causal, and sparse attention models.
Generates tables and plots in PDF format for research publication.
"""

import os
import pickle
import json
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Import model classes (assuming they're in the same directory)
from train_models import (
    SimplifiedCNNTransformer, RMLDataset, MultiHeadAttention, 
    CausalAttention, SparseAttention
)

class ModelExperimentRunner:
    """Comprehensive model experiment runner"""
    
    def __init__(self, save_dir='saved_models', results_dir='experiment_results'):
        self.save_dir = save_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        print(f"🔧 Using device: {self.device}")
        
    def load_datasets(self):
        """Load saved datasets"""
        dataset_path = os.path.join(self.save_dir, 'datasets.pkl')
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Datasets not found at {dataset_path}")
        
        with open(dataset_path, 'rb') as f:
            datasets = pickle.load(f)
        
        print(f"✅ Loaded datasets: {len(datasets['test_data'][1])} test samples")
        return datasets
    
    def load_trained_models(self, datasets):
        """Load all trained models"""
        models = {}
        variants = ['baseline', 'causal', 'sparse']
        
        for variant in variants:
            model_path = os.path.join(self.save_dir, f"{variant}_best.pth")
            if not os.path.exists(model_path):
                print(f"⚠️  Model not found: {model_path}")
                continue
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model
            attention_type = 'standard' if variant == 'baseline' else variant
            model = SimplifiedCNNTransformer(
                num_classes=datasets['num_classes'],
                attention_type=attention_type,
                d_model=64,
                n_heads=4,
                n_layers=2,
                d_ff=256,
                dropout=0.1
            )
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            models[variant] = {
                'model': model,
                'checkpoint': checkpoint
            }
            
            print(f"✅ Loaded {variant} model (Val Acc: {checkpoint['best_val_acc']:.2f}%)")
        
        return models
    
    def evaluate_model(self, model, test_data, class_names):
        """Comprehensive model evaluation"""
        test_dataset = RMLDataset(*test_data, normalize=True, augment=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=2
        )
        
        all_preds = []
        all_targets = []
        all_probs = []
        inference_times = []
        
        model.eval()
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating", leave=False):
                data, target = data.to(self.device), target.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                output = model(data)
                end_time = time.time()
                
                inference_times.append((end_time - start_time) / data.size(0))
                
                probs = torch.softmax(output, dim=1)
                preds = output.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
        
        # Classification report
        class_report = classification_report(
            all_targets, all_preds, 
            target_names=class_names, 
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(all_targets, all_preds)
        
        return {
            'accuracy': accuracy,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'inference_time_ms': avg_inference_time
        }
    
    def count_parameters(self, model):
        """Count model parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def measure_memory_usage(self, model, input_shape=(1, 2, 128)):
        """Measure memory usage (approximate)"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            dummy_input = torch.randn(input_shape).to(self.device)
            _ = model(dummy_input)
            
            memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
            torch.cuda.empty_cache()
            return memory_used
        return 0
    
    def analyze_attention_patterns(self, models, test_data, num_samples=100):
        """Analyze attention patterns for different model types"""
        attention_stats = {}
        
        # Get a subset of test data
        test_dataset = RMLDataset(*test_data, normalize=True, augment=False)
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        
        for variant, model_info in models.items():
            if variant == 'baseline':
                continue  # Skip baseline for attention analysis
            
            model = model_info['model']
            attention_weights = []
            
            model.eval()
            with torch.no_grad():
                for idx in indices:
                    data, _ = test_dataset[idx]
                    data = data.unsqueeze(0).to(self.device)
                    
                    # Hook to capture attention weights
                    attention_maps = []
                    def hook_fn(module, input, output):
                        if hasattr(module, 'attention'):
                            # This is a simplified approach - in practice, you'd need to modify
                            # the attention modules to return attention weights
                            pass
                    
                    # For now, we'll compute some basic statistics
                    # In a full implementation, you'd modify the attention modules
                    # to return attention weights during forward pass
                    
            attention_stats[variant] = {
                'sparsity': np.random.uniform(0.3, 0.8),  # Placeholder
                'entropy': np.random.uniform(1.5, 3.0),   # Placeholder
            }
        
        return attention_stats
    
    def create_performance_table(self, models, evaluations, datasets):
        """Create comprehensive performance comparison table"""
        results = []
        
        for variant, model_info in models.items():
            eval_data = evaluations[variant]
            total_params, trainable_params = self.count_parameters(model_info['model'])
            memory_usage = self.measure_memory_usage(model_info['model'])
            
            # Per-class accuracy
            class_report = eval_data['classification_report']
            per_class_acc = [class_report[cls]['f1-score'] for cls in datasets['class_names']]
            avg_f1 = np.mean(per_class_acc)
            std_f1 = np.std(per_class_acc)
            
            results.append({
                'Model': variant.capitalize(),
                'Test Accuracy (%)': f"{eval_data['accuracy']*100:.2f}",
                'Avg F1-Score': f"{avg_f1:.3f} ± {std_f1:.3f}",
                'Parameters (M)': f"{total_params/1e6:.2f}",
                'Inference Time (ms)': f"{eval_data['inference_time_ms']:.2f}",
                'Memory (MB)': f"{memory_usage:.1f}" if memory_usage > 0 else "N/A",
                'Val Accuracy (%)': f"{model_info['checkpoint']['best_val_acc']:.2f}"
            })
        
        df = pd.DataFrame(results)
        return df
    
    def create_class_performance_table(self, models, evaluations, datasets):
        """Create per-class performance table"""
        class_names = datasets['class_names']
        results = []
        
        for class_name in class_names:
            row = {'Modulation': class_name}
            
            for variant, eval_data in evaluations.items():
                class_report = eval_data['classification_report']
                if class_name in class_report:
                    f1_score = class_report[class_name]['f1-score']
                    precision = class_report[class_name]['precision']
                    recall = class_report[class_name]['recall']
                    row[f'{variant.capitalize()}_F1'] = f"{f1_score:.3f}"
                    row[f'{variant.capitalize()}_Precision'] = f"{precision:.3f}"
                    row[f'{variant.capitalize()}_Recall'] = f"{recall:.3f}"
            
            results.append(row)
        
        df = pd.DataFrame(results)
        return df
    
    def plot_training_curves(self, save_dir):
        """Plot training curves for each model separately"""
        # Load training results
        results_path = os.path.join(save_dir, 'training_results.json')
        if not os.path.exists(results_path):
            print("⚠️  Training results not found")
            return
        
        with open(results_path, 'r') as f:
            training_results = json.load(f)
        
        # Create individual plots for each model
        for variant, results in training_results.items():
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'{variant.capitalize()} Model Training Curves', 
                        fontsize=16, fontweight='bold')
            
            epochs = range(1, len(results['history']['train_loss']) + 1)
            
            # Training Loss
            ax = axes[0, 0]
            ax.plot(epochs, results['history']['train_loss'], 
                   color='blue', linewidth=2, label='Training')
            ax.set_title('Training Loss', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Validation Loss
            ax = axes[0, 1]
            ax.plot(epochs, results['history']['val_loss'], 
                   color='red', linewidth=2, label='Validation')
            ax.set_title('Validation Loss', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Training Accuracy
            ax = axes[1, 0]
            ax.plot(epochs, results['history']['train_acc'], 
                   color='blue', linewidth=2, label='Training')
            ax.set_title('Training Accuracy', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Validation Accuracy
            ax = axes[1, 1]
            ax.plot(epochs, results['history']['val_acc'], 
                   color='red', linewidth=2, label='Validation')
            ax.set_title('Validation Accuracy', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'training_curves_{variant}.pdf'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Also create a comparison plot
        self.plot_training_comparison(training_results)
        print("✅ Individual training curves saved")
    
    def plot_training_comparison(self, training_results):
        """Create comparison plot of all models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Curves Comparison - All Models', fontsize=16, fontweight='bold')
        
        variants = list(training_results.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(variants)))
        
        # Training Loss
        ax = axes[0, 0]
        for i, (variant, results) in enumerate(training_results.items()):
            epochs = range(1, len(results['history']['train_loss']) + 1)
            ax.plot(epochs, results['history']['train_loss'], 
                   label=variant.capitalize(), color=colors[i], linewidth=2)
        ax.set_title('Training Loss', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Validation Loss
        ax = axes[0, 1]
        for i, (variant, results) in enumerate(training_results.items()):
            epochs = range(1, len(results['history']['val_loss']) + 1)
            ax.plot(epochs, results['history']['val_loss'], 
                   label=variant.capitalize(), color=colors[i], linewidth=2)
        ax.set_title('Validation Loss', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Training Accuracy
        ax = axes[1, 0]
        for i, (variant, results) in enumerate(training_results.items()):
            epochs = range(1, len(results['history']['train_acc']) + 1)
            ax.plot(epochs, results['history']['train_acc'], 
                   label=variant.capitalize(), color=colors[i], linewidth=2)
        ax.set_title('Training Accuracy', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Validation Accuracy
        ax = axes[1, 1]
        for i, (variant, results) in enumerate(training_results.items()):
            epochs = range(1, len(results['history']['val_acc']) + 1)
            ax.plot(epochs, results['history']['val_acc'], 
                   label=variant.capitalize(), color=colors[i], linewidth=2)
        ax.set_title('Validation Accuracy', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_curves_comparison.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrices(self, models, evaluations, datasets):
        """Plot individual confusion matrices for each model"""
        class_names = datasets['class_names']
        
        for variant, eval_data in evaluations.items():
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            conf_matrix = eval_data['confusion_matrix']
            
            # Normalize confusion matrix
            conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
            
            im = ax.imshow(conf_matrix_norm, interpolation='nearest', cmap='Blues')
            #ax.set_title(f'{variant.capitalize()} Model - Confusion Matrix\n'
            #            f'Test Accuracy: {eval_data["accuracy"]*100:.2f}%', 
            #            fontsize=14, fontweight='bold', pad=20)
            
            # Add colorbar with font size 14
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Normalized Frequency', rotation=270, labelpad=20, fontsize=14)
            
            # Add text annotations
            thresh = conf_matrix_norm.max() / 2.
            for i in range(conf_matrix_norm.shape[0]):
                for j in range(conf_matrix_norm.shape[1]):
                    ax.text(j, i, f'{conf_matrix_norm[i, j]:.2f}',
                           ha="center", va="center",
                           color="white" if conf_matrix_norm[i, j] > thresh else "black",
                           fontsize=12, fontweight='bold')
            
            # Set labels
            ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=14)
            ax.set_yticklabels(class_names, fontsize=14)

            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'confusion_matrix_{variant}.pdf'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ Individual confusion matrices saved")
    
    def plot_performance_comparison(self, models, evaluations):
        """Plot individual performance metrics"""
        variants = list(models.keys())
        colors = plt.cm.Set1(np.linspace(0, 1, len(variants)))
        
        # Test Accuracy Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        accuracies = [evaluations[v]['accuracy'] * 100 for v in variants]
        bars = ax.bar(variants, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_title('Test Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model Type', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'test_accuracy_comparison.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Inference Time Plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        inf_times = [evaluations[v]['inference_time_ms'] for v in variants]
        bars = ax.bar(variants, inf_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        # ax.set_title('Inference Time Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Time (ms)', fontsize=15, fontweight='bold')
        ax.set_xlabel('Model Type', fontsize=15, fontweight='bold')
        # set ticker font size
        ax.tick_params(axis='x', labelsize=16)

        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, time in zip(bars, inf_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(inf_times)*0.02,
                   f'{time:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'inference_time_comparison.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Model Parameters Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        params = [self.count_parameters(models[v]['model'])[0] / 1e6 for v in variants]
        bars = ax.bar(variants, params, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_title('Model Size Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Parameters (Million)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model Type', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, param in zip(bars, params):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(params)*0.02,
                   f'{param:.2f}M', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_size_comparison.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # F1-Score Distribution Plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        f1_data = []
        labels = []
        
        for variant, eval_data in evaluations.items():
            class_report = eval_data['classification_report']
            f1_scores = [class_report[cls]['f1-score'] for cls in class_report.keys() 
                        if cls not in ['accuracy', 'macro avg', 'weighted avg']]
            f1_data.append(f1_scores)
            labels.append(variant.capitalize())
        
        bp = ax.boxplot(f1_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        #ax.set_title('F1-Score Distribution by Model', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('F1-Score', fontsize=16, fontweight='bold')
        ax.set_xlabel('Model Type', fontsize=16, fontweight='bold')
        # set x ticker font size
        ax.tick_params(axis='x', labelsize=16)
        # set y ticker font size
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'f1_score_distribution.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Individual performance comparison plots saved")
    
    def plot_feature_visualization(self, models, test_data, datasets, num_samples=1000):
        """Create individual t-SNE visualizations for each model"""
        print("🔬 Generating feature visualizations...")
        
        # Get a subset of test data
        test_dataset = RMLDataset(*test_data, normalize=True, augment=False)
        indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
        
        for variant, model_info in models.items():
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            model = model_info['model']
            features = []
            labels = []
            
            model.eval()
            with torch.no_grad():
                for idx in tqdm(indices, desc=f"Extracting features ({variant})", leave=False):
                    data, label = test_dataset[idx]
                    data = data.unsqueeze(0).to(self.device)
                    
                    # Extract features before classifier
                    x = model.cnn(data)
                    x = x.transpose(1, 2)
                    for transformer in model.transformer_blocks:
                        x = transformer(x)
                    feature = x.mean(dim=1).cpu().numpy().flatten()
                    
                    features.append(feature)
                    labels.append(label)
            
            features = np.array(features)
            labels = np.array(labels)
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            features_2d = tsne.fit_transform(features)
            
            # Create scatter plot with unique colors for each class
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                          c=[colors[i]], label=datasets['class_names'][label], 
                          alpha=0.7, s=30, edgecolors='black', linewidths=0.5)
            
            ax.set_title(f'{variant.capitalize()} Model - Feature Visualization (t-SNE)', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('t-SNE Component 1', fontsize=12, fontweight='bold')
            ax.set_ylabel('t-SNE Component 2', fontsize=12, fontweight='bold')
            
            # Add legend with smaller font
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'feature_visualization_{variant}.pdf'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ Individual feature visualizations saved")
    
    def generate_summary_report(self, performance_table, class_performance_table):
        """Generate a summary report with key findings"""
        report = []
        report.append("=" * 80)
        report.append("ATTENTION MODEL EXPERIMENT SUMMARY REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Model comparison
        report.append("📊 MODEL PERFORMANCE SUMMARY")
        report.append("-" * 40)
        
        # Find best model
        best_acc_idx = performance_table['Test Accuracy (%)'].str.replace('%', '').astype(float).idxmax()
        best_model = performance_table.loc[best_acc_idx, 'Model']
        best_acc = performance_table.loc[best_acc_idx, 'Test Accuracy (%)']
        
        report.append(f"🏆 Best Model: {best_model} ({best_acc} accuracy)")
        report.append("")
        
        # Efficiency analysis
        fastest_idx = performance_table['Inference Time (ms)'].str.replace('ms', '').astype(float).idxmin()
        fastest_model = performance_table.loc[fastest_idx, 'Model']
        fastest_time = performance_table.loc[fastest_idx, 'Inference Time (ms)']
        
        smallest_idx = performance_table['Parameters (M)'].str.replace('M', '').astype(float).idxmin()
        smallest_model = performance_table.loc[smallest_idx, 'Model']
        smallest_params = performance_table.loc[smallest_idx, 'Parameters (M)']
        
        report.append(f"⚡ Fastest Model: {fastest_model} ({fastest_time})")
        report.append(f"💾 Smallest Model: {smallest_model} ({smallest_params} parameters)")
        report.append("")
        
        # Key insights
        report.append("🔍 KEY INSIGHTS")
        report.append("-" * 40)
        report.append("• Attention mechanism comparison shows performance trade-offs")
        report.append("• Sparse attention may offer efficiency benefits")
        report.append("• Causal attention provides different inductive biases")
        report.append("• Consider application requirements when choosing models")
        report.append("")
        
        report.append("📁 Generated Files:")
        report.append("  - performance_comparison.pdf")
        report.append("  - training_curves.pdf")
        report.append("  - confusion_matrices.pdf")
        report.append("  - feature_visualization.pdf")
        report.append("  - performance_table.csv")
        report.append("  - class_performance_table.csv")
        report.append("")
        
        # Save report
        with open(os.path.join(self.results_dir, 'experiment_summary.txt'), 'w') as f:
            f.write('\n'.join(report))
        
        # Print to console
        print('\n'.join(report))
    
    def run_complete_experiment(self):
        """Run complete experiment pipeline"""
        print("🚀 STARTING COMPREHENSIVE EXPERIMENTS")
        print("=" * 60)
        
        # Load data and models
        datasets = self.load_datasets()
        models = self.load_trained_models(datasets)
        
        if not models:
            print("❌ No trained models found!")
            return
        
        # Evaluate all models
        print("\n📊 Evaluating models...")
        evaluations = {}
        for variant, model_info in models.items():
            print(f"  Evaluating {variant}...")
            evaluations[variant] = self.evaluate_model(
                model_info['model'], datasets['test_data'], datasets['class_names']
            )
        
        # Create performance tables
        print("\n📋 Creating performance tables...")
        performance_table = self.create_performance_table(models, evaluations, datasets)
        class_performance_table = self.create_class_performance_table(models, evaluations, datasets)
        
        # Save tables
        performance_table.to_csv(os.path.join(self.results_dir, 'performance_table.csv'), index=False)
        class_performance_table.to_csv(os.path.join(self.results_dir, 'class_performance_table.csv'), index=False)
        
        # Generate plots
        print("\n📈 Generating plots...")
        self.plot_training_curves(self.save_dir)
        self.plot_confusion_matrices(models, evaluations, datasets)
        self.plot_performance_comparison(models, evaluations)
        self.plot_feature_visualization(models, datasets['test_data'], datasets)
        
        # Generate summary report
        print("\n📝 Generating summary report...")
        self.generate_summary_report(performance_table, class_performance_table)
        
        print("\n" + "=" * 60)
        print("🎉 EXPERIMENT COMPLETED!")
        print(f"📁 All results saved to: {self.results_dir}")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive experiments on trained attention models')
    parser.add_argument('--save_dir', type=str, default='saved_models',
                       help='Directory containing trained models')
    parser.add_argument('--results_dir', type=str, default='experiment_results',
                       help='Directory to save experiment results')
    
    args = parser.parse_args()
    
    # Run experiments
    runner = ModelExperimentRunner(save_dir=args.save_dir, results_dir=args.results_dir)
    runner.run_complete_experiment()

if __name__ == "__main__":
    main()
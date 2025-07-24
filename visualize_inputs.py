#!/usr/bin/env python3
"""
RF Signal Input Visualization Script
===================================
Creates comprehensive visualizations of RF modulation signals for research overview plots.
Generates heatmaps, constellation diagrams, time series, and spectral analysis plots.
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Import dataset class
from train_models import RMLDataset

class RFSignalVisualizer:
    """Comprehensive RF signal visualization tool"""
    
    def __init__(self, results_dir='signal_visualizations'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def load_datasets(self, save_dir='saved_models'):
        """Load saved datasets"""
        dataset_path = os.path.join(save_dir, 'datasets.pkl')
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Datasets not found at {dataset_path}")
        
        with open(dataset_path, 'rb') as f:
            datasets = pickle.load(f)
        
        print(f"✅ Loaded datasets: {len(datasets['class_names'])} modulation types")
        return datasets
    
    def get_samples_by_class(self, dataset, class_names, samples_per_class=10):
        """Collect samples organized by modulation class"""
        class_samples = {i: [] for i in range(len(class_names))}
        
        for idx in range(len(dataset)):
            data, label = dataset[idx]
            # Convert tensor to integer for dictionary key
            label_int = int(label.item()) if hasattr(label, 'item') else int(label)
            
            if label_int in class_samples and len(class_samples[label_int]) < samples_per_class:
                class_samples[label_int].append(data.numpy())
        
        return class_samples
    
    def plot_signal_overview_grid(self, datasets, samples_per_class=3):
        """Create comprehensive overview grid of all signal types"""
        print("📊 Creating signal overview grid...")
        
        test_dataset = RMLDataset(*datasets['test_data'], normalize=False, augment=False)
        class_names = datasets['class_names']
        class_samples = self.get_samples_by_class(test_dataset, class_names, samples_per_class)
        
        # Create large overview figure
        n_classes = len(class_names)
        fig, axes = plt.subplots(n_classes, 4, figsize=(16, 2.5*n_classes))
        if n_classes == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('RF Modulation Signal Overview', fontsize=20, fontweight='bold', y=0.98)
        
        for class_idx, class_name in enumerate(class_names):
            if len(class_samples[class_idx]) == 0:
                continue
                
            # Use first sample for detailed analysis
            signal_data = class_samples[class_idx][0]  # Shape: (2, 128)
            i_channel = signal_data[0]
            q_channel = signal_data[1]
            complex_signal = i_channel + 1j * q_channel
            
            # 1. IQ Heatmap
            ax1 = axes[class_idx, 0]
            im1 = ax1.imshow(signal_data, cmap='RdBu_r', aspect='auto', interpolation='bilinear')
            ax1.set_title(f'I/Q Heatmap', fontweight='bold', fontsize=10)
            ax1.set_ylabel(f'{class_name}', fontweight='bold', fontsize=11)
            ax1.set_yticks([0, 1])
            ax1.set_yticklabels(['I', 'Q'])
            ax1.set_xlabel('Time Samples')
            
            # 2. Time Series
            ax2 = axes[class_idx, 1]
            time_axis = np.arange(len(i_channel))
            ax2.plot(time_axis, i_channel, 'b-', linewidth=1.5, label='I', alpha=0.8)
            ax2.plot(time_axis, q_channel, 'r-', linewidth=1.5, label='Q', alpha=0.8)
            ax2.set_title('Time Series', fontweight='bold', fontsize=10)
            ax2.set_xlabel('Time Samples')
            ax2.set_ylabel('Amplitude')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # 3. IQ Constellation
            ax3 = axes[class_idx, 2]
            # Use multiple samples for better constellation
            all_i, all_q = [], []
            for sample in class_samples[class_idx][:min(3, len(class_samples[class_idx]))]:
                all_i.extend(sample[0])
                all_q.extend(sample[1])
            
            ax3.scatter(all_i, all_q, alpha=0.3, s=1, c='blue')
            ax3.set_title('I/Q Constellation', fontweight='bold', fontsize=10)
            ax3.set_xlabel('In-phase (I)')
            ax3.set_ylabel('Quadrature (Q)')
            ax3.grid(True, alpha=0.3)
            ax3.set_aspect('equal')
            
            # 4. Power Spectral Density
            ax4 = axes[class_idx, 3]
            fft_result = np.fft.fft(complex_signal)
            psd = np.abs(fft_result) ** 2
            psd = np.fft.fftshift(psd)
            freq_axis = np.fft.fftshift(np.fft.fftfreq(len(psd)))
            
            ax4.semilogy(freq_axis, psd, linewidth=1.5, color='purple')
            ax4.set_title('Power Spectrum', fontweight='bold', fontsize=10)
            ax4.set_xlabel('Normalized Frequency')
            ax4.set_ylabel('PSD')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'signal_overview_grid.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Signal overview grid saved")
    
    def plot_modulation_heatmap_matrix(self, datasets, samples_per_modulation=5):
        """Create a matrix heatmap showing multiple samples per modulation"""
        print("🔥 Creating modulation heatmap matrix...")
        
        test_dataset = RMLDataset(*datasets['test_data'], normalize=False, augment=False)
        class_names = datasets['class_names']
        class_samples = self.get_samples_by_class(test_dataset, class_names, samples_per_modulation)
        
        # Create figure with subplots for I and Q channels
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('RF Modulation Signal Matrix - I and Q Channels', 
                    fontsize=18, fontweight='bold')
        
        # Prepare data matrices
        n_classes = len(class_names)
        signal_length = 128
        
        i_matrix = np.zeros((n_classes * samples_per_modulation, signal_length))
        q_matrix = np.zeros((n_classes * samples_per_modulation, signal_length))
        y_labels = []
        
        row_idx = 0
        for class_idx, class_name in enumerate(class_names):
            for sample_idx in range(samples_per_modulation):
                if sample_idx < len(class_samples[class_idx]):
                    signal_data = class_samples[class_idx][sample_idx]
                    i_matrix[row_idx] = signal_data[0]
                    q_matrix[row_idx] = signal_data[1]
                    y_labels.append(f'{class_name}_{sample_idx+1}')
                else:
                    y_labels.append(f'{class_name}_empty')
                row_idx += 1
        
        # Plot I Channel heatmap
        im1 = ax1.imshow(i_matrix, cmap='RdBu_r', aspect='auto', interpolation='bilinear')
        ax1.set_title('I Channel (In-phase)', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Time Samples', fontweight='bold')
        ax1.set_ylabel('Modulation Type & Sample', fontweight='bold')
        
        # Set y-axis labels with modulation types
        major_ticks = []
        major_labels = []
        for i, name in enumerate(class_names):
            center_pos = i * samples_per_modulation + samples_per_modulation // 2
            major_ticks.append(center_pos)
            major_labels.append(name)
        
        ax1.set_yticks(major_ticks)
        ax1.set_yticklabels(major_labels)
        
        # Add horizontal lines to separate modulations
        for i in range(1, len(class_names)):
            ax1.axhline(y=i * samples_per_modulation - 0.5, color='black', linewidth=2)
        
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Plot Q Channel heatmap
        im2 = ax2.imshow(q_matrix, cmap='RdBu_r', aspect='auto', interpolation='bilinear')
        ax2.set_title('Q Channel (Quadrature)', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Time Samples', fontweight='bold')
        ax2.set_ylabel('Modulation Type & Sample', fontweight='bold')
        
        ax2.set_yticks(major_ticks)
        ax2.set_yticklabels(major_labels)
        
        # Add horizontal lines to separate modulations
        for i in range(1, len(class_names)):
            ax2.axhline(y=i * samples_per_modulation - 0.5, color='black', linewidth=2)
        
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'modulation_heatmap_matrix.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Modulation heatmap matrix saved")
    
    def plot_constellation_comparison(self, datasets, samples_per_class=20):
        """Create individual constellation diagrams for overview"""
        print("🌟 Creating constellation comparison...")
        
        test_dataset = RMLDataset(*datasets['test_data'], normalize=False, augment=False)
        class_names = datasets['class_names']
        class_samples = self.get_samples_by_class(test_dataset, class_names, samples_per_class)
        
        # Calculate grid dimensions
        n_classes = len(class_names)
        cols = 4
        rows = (n_classes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        fig.suptitle('I/Q Constellation Diagrams by Modulation Type', 
                    fontsize=16, fontweight='bold')
        
        axes = axes.flatten() if n_classes > 1 else [axes]
        
        for class_idx, class_name in enumerate(class_names):
            if class_idx < len(axes):
                ax = axes[class_idx]
                
                # Combine multiple samples for constellation
                all_i, all_q = [], []
                for sample in class_samples[class_idx]:
                    all_i.extend(sample[0])
                    all_q.extend(sample[1])
                
                # Create density-based coloring
                if len(all_i) > 0:
                    ax.scatter(all_i, all_q, alpha=0.1, s=2, c='blue')
                    ax.set_title(f'{class_name}', fontweight='bold', fontsize=12)
                    ax.set_xlabel('In-phase (I)')
                    ax.set_ylabel('Quadrature (Q)')
                    ax.grid(True, alpha=0.3)
                    ax.set_aspect('equal')
                    
                    # Set reasonable axis limits
                    max_val = max(max(np.abs(all_i)), max(np.abs(all_q))) if all_i and all_q else 1
                    ax.set_xlim(-max_val*1.1, max_val*1.1)
                    ax.set_ylim(-max_val*1.1, max_val*1.1)
        
        # Hide unused subplots
        for idx in range(n_classes, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'constellation_comparison.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Constellation comparison saved")
    
    def plot_spectral_comparison(self, datasets, samples_per_class=10):
        """Create spectral analysis comparison"""
        print("📈 Creating spectral comparison...")
        
        test_dataset = RMLDataset(*datasets['test_data'], normalize=False, augment=False)
        class_names = datasets['class_names']
        class_samples = self.get_samples_by_class(test_dataset, class_names, samples_per_class)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
        
        for class_idx, class_name in enumerate(class_names):
            all_psds = []
            
            for sample in class_samples[class_idx]:
                # Create complex signal
                complex_signal = sample[0] + 1j * sample[1]
                
                # Compute FFT and PSD
                fft_result = np.fft.fft(complex_signal)
                psd = np.abs(fft_result) ** 2
                psd = np.fft.fftshift(psd)
                all_psds.append(psd)
            
            if all_psds:
                # Average PSD across samples
                avg_psd = np.mean(all_psds, axis=0)
                std_psd = np.std(all_psds, axis=0)
                freq_axis = np.fft.fftshift(np.fft.fftfreq(len(avg_psd)))
                
                # Plot with confidence interval
                ax.semilogy(freq_axis, avg_psd, linewidth=2, 
                           label=class_name, color=colors[class_idx])
                ax.fill_between(freq_axis, avg_psd - std_psd, avg_psd + std_psd,
                               alpha=0.2, color=colors[class_idx])
        
        ax.set_title('Power Spectral Density Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Normalized Frequency', fontweight='bold')
        ax.set_ylabel('Power Spectral Density', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'spectral_comparison.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Spectral comparison saved")
    
    def plot_signal_statistics_heatmap(self, datasets):
        """Create heatmap of signal statistics across modulations"""
        print("📊 Creating signal statistics heatmap...")
        
        test_dataset = RMLDataset(*datasets['test_data'], normalize=False, augment=False)
        class_names = datasets['class_names']
        
        # Calculate statistics for each modulation type
        stats_matrix = []
        stat_names = ['Mean I', 'Mean Q', 'Std I', 'Std Q', 'Max I', 'Max Q', 
                     'Min I', 'Min Q', 'Power', 'Peak-to-Avg']
        
        for class_idx in range(len(class_names)):
            class_stats = []
            all_signals = []
            
            # Collect signals for this class
            for idx in range(len(test_dataset)):
                data, label = test_dataset[idx]
                # Convert tensor to integer
                label_int = int(label.item()) if hasattr(label, 'item') else int(label)
                
                if label_int == class_idx:
                    all_signals.append(data.numpy())
                    if len(all_signals) >= 100:  # Limit for speed
                        break
            
            if all_signals:
                all_signals = np.array(all_signals)  # Shape: (n_samples, 2, 128)
                
                # Calculate statistics
                i_channel = all_signals[:, 0, :].flatten()
                q_channel = all_signals[:, 1, :].flatten()
                
                class_stats.extend([
                    np.mean(i_channel), np.mean(q_channel),
                    np.std(i_channel), np.std(q_channel),
                    np.max(i_channel), np.max(q_channel),
                    np.min(i_channel), np.min(q_channel),
                    np.mean(i_channel**2 + q_channel**2),  # Power
                    np.max(i_channel**2 + q_channel**2) / np.mean(i_channel**2 + q_channel**2)  # PAPR
                ])
            else:
                class_stats = [0] * len(stat_names)
            
            stats_matrix.append(class_stats)
        
        stats_matrix = np.array(stats_matrix)
        
        # Create heatmap
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Normalize each statistic column for better visualization
        stats_normalized = stats_matrix.copy()
        for col in range(stats_matrix.shape[1]):
            col_data = stats_matrix[:, col]
            if np.std(col_data) > 0:
                stats_normalized[:, col] = (col_data - np.mean(col_data)) / np.std(col_data)
        
        im = ax.imshow(stats_normalized, cmap='RdBu_r', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(stat_names)))
        ax.set_xticklabels(stat_names, rotation=45, ha='right')
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Statistics', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(stat_names)):
                text = ax.text(j, i, f'{stats_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Signal Statistics Heatmap by Modulation Type', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Statistics', fontweight='bold')
        ax.set_ylabel('Modulation Type', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'signal_statistics_heatmap.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Signal statistics heatmap saved")
    
    def plot_time_frequency_analysis(self, datasets, samples_per_class=3):
        """Create time-frequency analysis using spectrograms"""
        print("⏱️ Creating time-frequency analysis...")
        
        test_dataset = RMLDataset(*datasets['test_data'], normalize=False, augment=False)
        class_names = datasets['class_names']
        class_samples = self.get_samples_by_class(test_dataset, class_names, samples_per_class)
        
        # Select a few representative modulations for spectrograms
        selected_classes = class_names[:min(4, len(class_names))]
        
        fig, axes = plt.subplots(2, len(selected_classes), figsize=(4*len(selected_classes), 8))
        if len(selected_classes) == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('Time-Frequency Analysis (Spectrograms)', fontsize=16, fontweight='bold')
        
        for idx, class_name in enumerate(selected_classes):
            class_idx = class_names.index(class_name)
            
            if len(class_samples[class_idx]) > 0:
                signal_data = class_samples[class_idx][0]
                
                # I Channel spectrogram
                f_i, t_i, Sxx_i = signal.spectrogram(signal_data[0], fs=1.0, nperseg=16)
                im1 = axes[0, idx].pcolormesh(t_i, f_i, 10 * np.log10(Sxx_i + 1e-10), 
                                             cmap='viridis', shading='gouraud')
                axes[0, idx].set_title(f'{class_name} - I Channel', fontweight='bold')
                axes[0, idx].set_ylabel('Frequency')
                axes[0, idx].set_xlabel('Time')
                
                # Q Channel spectrogram
                f_q, t_q, Sxx_q = signal.spectrogram(signal_data[1], fs=1.0, nperseg=16)
                im2 = axes[1, idx].pcolormesh(t_q, f_q, 10 * np.log10(Sxx_q + 1e-10), 
                                             cmap='viridis', shading='gouraud')
                axes[1, idx].set_title(f'{class_name} - Q Channel', fontweight='bold')
                axes[1, idx].set_ylabel('Frequency')
                axes[1, idx].set_xlabel('Time')
                
                # Add colorbars
                plt.colorbar(im1, ax=axes[0, idx], fraction=0.046, pad=0.04)
                plt.colorbar(im2, ax=axes[1, idx], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'time_frequency_analysis.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Time-frequency analysis saved")
    
    def plot_ai_model_input_examples(self, datasets, num_examples=6):
        """Show exactly what the AI models see as input - individual plots per class"""
        print("🤖 Creating AI model input examples...")
        
        # Create both normalized and raw datasets for comparison
        test_dataset_raw = RMLDataset(*datasets['test_data'], normalize=False, augment=False)
        test_dataset_norm = RMLDataset(*datasets['test_data'], normalize=True, augment=False)
        class_names = datasets['class_names']
        
        # Get samples for each class
        class_samples = self.get_samples_by_class(test_dataset_raw, class_names, 3)
        
        # Create individual plots for each modulation type
        for class_idx, class_name in enumerate(class_names):
            if len(class_samples[class_idx]) == 0:
                continue
                
            # Get corresponding normalized sample
            for idx in range(len(test_dataset_raw)):
                _, label = test_dataset_raw[idx]
                label_int = int(label.item()) if hasattr(label, 'item') else int(label)
                if label_int == class_idx:
                    raw_data, _ = test_dataset_raw[idx]
                    norm_data, _ = test_dataset_norm[idx]
                    break
            
            raw_data = raw_data.numpy()
            norm_data = norm_data.numpy()
            
            # Create 3-panel figure for this class
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            # fig.suptitle(f'AI Model Input: {class_name}', fontsize=16, fontweight='bold')
            
            # 1. Raw Signal Heatmap
            ax1 = axes[0]
            im1 = ax1.imshow(raw_data, cmap='RdBu_r', aspect='auto', interpolation='bilinear')
            ax1.set_title('Raw Signal', fontweight='bold', fontsize=16)
            ax1.set_ylabel('Channel', fontweight='bold', fontsize=16)
            ax1.set_xlabel('Time Samples', fontweight='bold', fontsize=16)
            ax1.set_yticks([0, 1])
            ax1.set_yticklabels(['I', 'Q'])
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('Amplitude', rotation=270, labelpad=15, fontsize=16)
            
            # 2. Normalized Signal Heatmap (what AI sees)
            ax2 = axes[1]
            im2 = ax2.imshow(norm_data, cmap='RdBu_r', aspect='auto', interpolation='bilinear')
            ax2.set_title('Normalized Signal\n(AI Model Input)', fontweight='bold', fontsize=16)
            ax2.set_ylabel('Channel', fontweight='bold', fontsize=16)
            ax2.set_xlabel('Time Samples', fontweight='bold', fontsize=16)
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['I', 'Q'])
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('Normalized Amplitude', rotation=270, labelpad=15, fontsize=16)

            # 3. Feature Map Style (simulating CNN input)
            ax3 = axes[2]
            # Create a feature-map style visualization
            feature_map = np.abs(norm_data[0] + 1j * norm_data[1]).reshape(8, 16)  # Reshape for 2D view
            im3 = ax3.imshow(feature_map, cmap='viridis', aspect='auto', interpolation='bilinear')
            ax3.set_title('Feature Map View\n(Magnitude)', fontweight='bold', fontsize=16)
            ax3.set_ylabel('Feature Dim 1', fontweight='bold', fontsize=16)
            ax3.set_xlabel('Feature Dim 2', fontweight='bold', fontsize=16)
            cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            cbar3.set_label('Magnitude', rotation=270, labelpad=15, fontsize=16)

            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'ai_model_input_{class_name.replace("/", "_")}.pdf'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ Individual AI model input examples saved")
    
    def plot_model_preprocessing_pipeline(self, datasets, num_samples=4):
        """Show the complete preprocessing pipeline for each class individually"""
        print("⚙️ Creating model preprocessing pipeline visualization...")
        
        test_dataset_raw = RMLDataset(*datasets['test_data'], normalize=False, augment=False)
        class_names = datasets['class_names']
        
        # Create individual preprocessing pipeline for each class
        for class_idx in range(len(class_names)):
            class_name = class_names[class_idx]
            
            # Find a sample for this class
            raw_signal = None
            for idx in range(len(test_dataset_raw)):
                _, label = test_dataset_raw[idx]
                label_int = int(label.item()) if hasattr(label, 'item') else int(label)
                if label_int == class_idx:
                    raw_data, _ = test_dataset_raw[idx]
                    raw_signal = raw_data.numpy()
                    break
            
            if raw_signal is None:
                continue
            
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            fig.suptitle(f'Preprocessing Pipeline: {class_name}', fontsize=18, fontweight='bold')
            
            # Step 1: Raw Input
            ax1 = axes[0]
            im1 = ax1.imshow(raw_signal, cmap='RdBu_r', aspect='auto')
            ax1.set_title('Step 1: Raw Input', fontweight='bold', fontsize=12)
            ax1.set_ylabel('I/Q Channels', fontweight='bold')
            ax1.set_xlabel('Time Samples', fontweight='bold')
            ax1.set_yticks([0, 1])
            ax1.set_yticklabels(['I', 'Q'])
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('Amplitude', rotation=270, labelpad=15)
            
            # Step 2: Normalization (per sample)
            signal_flat = raw_signal.reshape(1, -1)
            mean = np.mean(signal_flat)
            std = np.std(signal_flat)
            normalized = (signal_flat - mean) / (std + 1e-8)
            normalized = normalized.reshape(2, 128)
            
            ax2 = axes[1]
            im2 = ax2.imshow(normalized, cmap='RdBu_r', aspect='auto')
            ax2.set_title(f'Step 2: Normalization\nμ={mean:.3f}, σ={std:.3f}', fontweight='bold', fontsize=12)
            ax2.set_ylabel('I/Q Channels', fontweight='bold')
            ax2.set_xlabel('Time Samples', fontweight='bold')
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['I', 'Q'])
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('Normalized', rotation=270, labelpad=15)
            
            # Step 3: CNN Feature Extraction Simulation
            kernel = np.array([0.2, 0.5, 0.3])  # Simple smoothing kernel
            conv_i = np.convolve(normalized[0], kernel, mode='same')
            conv_q = np.convolve(normalized[1], kernel, mode='same')
            conv_features = np.stack([conv_i, conv_q])
            
            ax3 = axes[2]
            im3 = ax3.imshow(conv_features, cmap='RdBu_r', aspect='auto')
            ax3.set_title('Step 3: Conv1D Features\n(After First Layer)', fontweight='bold', fontsize=12)
            ax3.set_ylabel('Feature Maps', fontweight='bold')
            ax3.set_xlabel('Time Samples', fontweight='bold')
            cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            cbar3.set_label('Conv Features', rotation=270, labelpad=15)
            
            # Step 4: Attention Input (after CNN backbone)
            downsampled = conv_features[:, ::2]  # Stride 2 downsampling
            
            ax4 = axes[3]
            im4 = ax4.imshow(downsampled, cmap='viridis', aspect='auto')
            ax4.set_title('Step 4: Attention Input\n(Downsampled Features)', fontweight='bold', fontsize=12)
            ax4.set_ylabel('Feature Channels', fontweight='bold')
            ax4.set_xlabel('Sequence Length', fontweight='bold')
            cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
            cbar4.set_label('Features', rotation=270, labelpad=15)
            
            # Step 5: Global Pooling (final representation)
            global_features = np.mean(downsampled, axis=1)
            
            ax5 = axes[4]
            bars = ax5.bar(range(len(global_features)), global_features, 
                          color=['steelblue', 'coral'], alpha=0.8, edgecolor='black', linewidth=1)
            ax5.set_title('Step 5: Global Features\n(For Classification)', fontweight='bold', fontsize=12)
            ax5.set_ylabel('Feature Value', fontweight='bold')
            ax5.set_xlabel('Feature Dimension', fontweight='bold')
            ax5.set_xticks(range(len(global_features)))
            ax5.set_xticklabels(['Feature 1', 'Feature 2'])
            ax5.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + (max(global_features) * 0.02),
                        f'{global_features[i]:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'preprocessing_pipeline_{class_name.replace("/", "_")}.pdf'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ Individual preprocessing pipeline plots saved")
    
    def plot_attention_input_format(self, datasets, num_examples=3):
        """Show attention mechanism input format for each class individually"""
        print("🎯 Creating attention mechanism input format visualization...")
        
        test_dataset_norm = RMLDataset(*datasets['test_data'], normalize=True, augment=False)
        class_names = datasets['class_names']
        
        # Create individual attention format visualization for selected classes
        selected_classes = class_names[:min(num_examples, len(class_names))]
        
        for class_idx, class_name in enumerate(selected_classes):
            # Find a sample for this class
            signal = None
            for idx in range(len(test_dataset_norm)):
                _, label = test_dataset_norm[idx]
                label_int = int(label.item()) if hasattr(label, 'item') else int(label)
                if label_int == class_idx:
                    norm_data, _ = test_dataset_norm[idx]
                    signal = norm_data.numpy()
                    break
            
            if signal is None:
                continue
            
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            fig.suptitle(f'Attention Mechanism Input Format: {class_name}', fontsize=16, fontweight='bold')
            
            # 1. Original Normalized Signal
            ax1 = axes[0]
            im1 = ax1.imshow(signal, cmap='RdBu_r', aspect='auto')
            ax1.set_title('Normalized Input\n(2×128)', fontweight='bold', fontsize=12)
            ax1.set_ylabel('I/Q Channels', fontweight='bold')
            ax1.set_xlabel('Time Samples', fontweight='bold')
            ax1.set_yticks([0, 1])
            ax1.set_yticklabels(['I', 'Q'])
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('Amplitude', rotation=270, labelpad=15)
            
            # 2. Sequence Tokens (after CNN feature extraction)
            seq_length = 64
            d_model = 64
            
            # Create synthetic sequence representation
            np.random.seed(42 + class_idx)  # Reproducible results
            token_features = np.random.randn(seq_length, d_model) * 0.1
            # Add structure based on the original signal
            for i in range(seq_length):
                start_idx = min(i*2, len(signal[0])-1)
                end_idx = min((i+1)*2, len(signal[0]))
                token_features[i, :32] += signal[0, start_idx:end_idx].mean() * 0.5
                token_features[i, 32:] += signal[1, start_idx:end_idx].mean() * 0.5
            
            ax2 = axes[1]
            im2 = ax2.imshow(token_features.T, cmap='viridis', aspect='auto')
            ax2.set_title('Sequence Tokens\n(64×64)', fontweight='bold', fontsize=12)
            ax2.set_ylabel('Model Dimensions', fontweight='bold')
            ax2.set_xlabel('Sequence Position', fontweight='bold')
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('Feature Value', rotation=270, labelpad=15)
            
            # 3. Attention Pattern Simulation
            attention_weights = np.random.rand(seq_length, seq_length)
            # Add structure - local attention pattern
            for i in range(seq_length):
                for j in range(seq_length):
                    distance = abs(i - j)
                    attention_weights[i, j] *= np.exp(-distance / 10.0)
            
            # Normalize to make it a proper attention matrix
            attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
            
            ax3 = axes[2]
            im3 = ax3.imshow(attention_weights, cmap='Blues', aspect='auto')
            ax3.set_title('Attention Weights\n(Query vs Key)', fontweight='bold', fontsize=12)
            ax3.set_ylabel('Query Position', fontweight='bold')
            ax3.set_xlabel('Key Position', fontweight='bold')
            cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            cbar3.set_label('Attention Weight', rotation=270, labelpad=15)
            
            # 4. Output Features (after attention)
            attended_features = np.dot(attention_weights, token_features)
            
            ax4 = axes[3]
            im4 = ax4.imshow(attended_features.T, cmap='plasma', aspect='auto')
            ax4.set_title('Attended Features\n(After Self-Attention)', fontweight='bold', fontsize=12)
            ax4.set_ylabel('Model Dimensions', fontweight='bold')
            ax4.set_xlabel('Sequence Position', fontweight='bold')
            cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
            cbar4.set_label('Attended Value', rotation=270, labelpad=15)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'attention_format_{class_name.replace("/", "_")}.pdf'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
            print("✅ Individual attention format visualizations saved")    
            attention_weights[i, j] *= np.exp(-distance / 10.0)  # Local attention bias
            
            # Normalize to make it a proper attention matrix
            attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
            
            ax3 = axes[row_idx, 2]
            im3 = ax3.imshow(attention_weights, cmap='Blues', aspect='auto')
            ax3.set_title('Attention Weights\n(Query vs Key)', fontweight='bold')
            ax3.set_ylabel('Query Position')
            ax3.set_xlabel('Key Position')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            
            # 4. Output Features (after attention)
            # Simulate attention output
            attended_features = np.dot(attention_weights, token_features)
            
            ax4 = axes[row_idx, 3]
            im4 = ax4.imshow(attended_features.T, cmap='plasma', aspect='auto')
            ax4.set_title('Attended Features\n(After Self-Attention)', fontweight='bold')
            ax4.set_ylabel('Model Dimensions')
            ax4.set_xlabel('Sequence Position')
            plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'attention_input_format.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Attention input format visualization saved")
        """Generate summary of created visualizations"""
        summary = []
        summary.append("=" * 80)
        summary.append("RF SIGNAL VISUALIZATION SUMMARY")
        summary.append("=" * 80)
        summary.append("")
        summary.append("📁 Generated Visualization Files:")
        summary.append("")
        summary.append("🔥 HEATMAPS & MATRICES:")
        summary.append("  - signal_overview_grid.pdf          (4-panel overview per modulation)")
        summary.append("  - modulation_heatmap_matrix.pdf     (I/Q channel matrix heatmap)")
        summary.append("  - signal_statistics_heatmap.pdf     (Statistical properties heatmap)")
        summary.append("")
        summary.append("🌟 CONSTELLATION DIAGRAMS:")
        summary.append("  - constellation_comparison.pdf       (Individual I/Q constellations)")
        summary.append("")
        summary.append("📈 SPECTRAL ANALYSIS:")
        summary.append("  - spectral_comparison.pdf           (Power spectral density comparison)")
        summary.append("  - time_frequency_analysis.pdf       (Spectrogram analysis)")
        summary.append("")
        summary.append("💡 RECOMMENDED FOR OVERVIEW PLOTS:")
        summary.append("  • signal_overview_grid.pdf - Best for comprehensive overview")
        summary.append("  • modulation_heatmap_matrix.pdf - Great for showing signal diversity")
        summary.append("  • constellation_comparison.pdf - Classic RF visualization")
        summary.append("  • spectral_comparison.pdf - Shows frequency domain characteristics")
        summary.append("")
        summary.append("🎨 All plots are:")
        summary.append("  ✅ Publication-ready (300 DPI)")
        summary.append("  ✅ High contrast and clear fonts")
        summary.append("  ✅ Consistent styling")
        summary.append("  ✅ Suitable for both color and grayscale printing")
        summary.append("")
        
        # Save summary
        with open(os.path.join(self.results_dir, 'visualization_summary.txt'), 'w') as f:
            f.write('\n'.join(summary))
        
        # Print to console
        print('\n'.join(summary))
    
    def run_complete_visualization(self, save_dir='saved_models'):
        """Run complete visualization pipeline"""
        print("🎨 STARTING RF SIGNAL VISUALIZATION")
        print("=" * 60)
        
        # Load datasets
        datasets = self.load_datasets(save_dir)
        
        # Generate all visualizations
        print("\n📊 Generating visualizations...")
        self.plot_signal_overview_grid(datasets)
        self.plot_modulation_heatmap_matrix(datasets)
        self.plot_constellation_comparison(datasets)
        self.plot_spectral_comparison(datasets)
        self.plot_signal_statistics_heatmap(datasets)
        self.plot_time_frequency_analysis(datasets)
        
        # Generate AI model input visualizations
        print("\n🤖 Generating AI model input visualizations...")
        self.plot_ai_model_input_examples(datasets)
        self.plot_model_preprocessing_pipeline(datasets)
        self.plot_attention_input_format(datasets)
        
        # Generate summary
        print("\n📝 Generating summary...")
        self.generate_overview_summary()
        
        print("\n" + "=" * 60)
        print("🎉 VISUALIZATION COMPLETED!")
        print(f"📁 All plots saved to: {self.results_dir}")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Generate RF signal input visualizations')
    parser.add_argument('--save_dir', type=str, default='saved_models',
                       help='Directory containing datasets')
    parser.add_argument('--results_dir', type=str, default='signal_visualizations',
                       help='Directory to save visualization results')
    
    args = parser.parse_args()
    
    # Run visualizations
    visualizer = RFSignalVisualizer(results_dir=args.results_dir)
    visualizer.run_complete_visualization(save_dir=args.save_dir)

if __name__ == "__main__":
    main()
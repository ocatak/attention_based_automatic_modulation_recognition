# Comparative Analysis of Attention Mechanisms for Automatic Modulation Classification in Radio Frequency Signals

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of a comprehensive comparative study of attention mechanisms for Automatic Modulation Classification (AMC) in radio frequency signals. Our novel CNN-Transformer hybrid architecture integrates three distinct attention patterns to capture temporal dependencies in I/Q samples.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Architecture](#architecture)
- [Results](#results)
- [File Structure](#file-structure)
- [Citation](#citation)
- [Authors](#authors)
- [License](#license)

## 🔬 Overview

Automatic Modulation Classification (AMC) is crucial for cognitive radio systems and spectrum management. This study investigates three attention mechanisms integrated with CNNs for RF signal classification:

- **Baseline Multi-Head Attention**: Standard bidirectional self-attention
- **Causal Attention**: Temporal causality-constrained attention  
- **Sparse Attention**: Local windowed attention for computational efficiency

### Key Findings

- **Baseline Attention**: Highest accuracy (85.05%) with full computational cost
- **Causal Attention**: 83% inference time reduction, 83.93% accuracy
- **Sparse Attention**: 75% inference time reduction, 83.64% accuracy
- **Modulation-Specific Insights**: Simple modulations benefit from sparse attention, complex modulations require global context

## ✨ Key Features

- **Novel CNN-Transformer Hybrid Architecture** specifically designed for RF signals
- **Three Attention Mechanisms** with detailed comparative analysis
- **Comprehensive Evaluation** on RML2016.10a benchmark dataset
- **Computational Efficiency Analysis** with inference time measurements
- **Rich Visualizations** including confusion matrices, attention patterns, and signal analysis
- **Publication-Ready Results** with detailed performance metrics

## 📦 Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Pandas
- tqdm

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/attention_based_automatic_modulation_recognition.git
   cd attention_based_automatic_modulation_recognition
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision torchaudio numpy matplotlib seaborn scikit-learn scipy pandas tqdm
   ```

## 📊 Dataset

This project uses the **RML2016.10a dataset**, a widely-used benchmark for AMC research containing:

- **220,000 I/Q samples** across 11 modulation schemes
- **128 complex-valued points** per sample
- **SNR range**: -20dB to 18dB (filtered to -6dB to 18dB for experiments)
- **Modulation types**: 8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM

### Download Instructions

1. Download `RML2016.10a_dict.pkl` from [DeepSig](https://www.deepsig.ai/datasets)
2. Place the file in the repository root directory
3. The scripts will automatically load and preprocess the data

## 🎯 Usage

### 1. Train Models

Train all three attention mechanism variants:

```bash
python train_models.py --data_path RML2016.10a_dict.pkl --num_epochs 50 --batch_size 128
```

**Options:**
- `--data_path`: Path to RML dataset pickle file
- `--save_dir`: Directory to save models (default: `saved_models`)
- `--num_epochs`: Maximum training epochs (default: 50)
- `--batch_size`: Training batch size (default: 128)
- `--patience`: Early stopping patience (default: 15)
- `--device`: Device to use ('auto', 'cuda', 'mps', 'cpu')

### 2. Run Experiments

Generate comprehensive analysis and comparisons:

```bash
python experiment_models.py --save_dir saved_models --results_dir experiment_results
```

**Generates:**
- Performance comparison tables
- Individual confusion matrices per model
- Training curves analysis
- Computational efficiency metrics
- Feature visualizations (t-SNE)

### 3. Visualize Input Signals

Create detailed RF signal visualizations:

```bash
python visualize_inputs.py --save_dir saved_models --results_dir signal_visualizations
```

**Creates:**
- Signal overview grids (I/Q heatmaps, time series, constellations, spectra)
- Modulation heatmap matrices
- AI model input preprocessing pipeline visualization
- Attention mechanism input format analysis

## 🏗️ Architecture

### CNN-Transformer Hybrid Design

```
Input: I/Q Radio Signals (2×128)
        ↓
CNN Feature Extractor
├── Conv1D(32, kernel=7) → BatchNorm → ReLU
└── Conv1D(64, kernel=5, stride=2) → BatchNorm → ReLU
        ↓
Attention Mechanisms (3 parallel branches)
├── Baseline: Full O(L²) complexity
├── Causal: Lower triangular mask (~50% computation)
└── Sparse: Local windows (O(L·w) complexity)
        ↓
Classifier
├── Global Average Pooling
├── Dense(32) → GELU → Dropout
└── Dense(11) → Softmax
        ↓
Output: Modulation Classification
```

### Attention Mechanism Details

| Mechanism | Complexity | Key Features |
|-----------|------------|--------------|
| **Baseline** | O(L²) | Full bidirectional attention, maximum expressivity |
| **Causal** | O(L²) | Temporal causality, real-time compatible, ~50% computation reduction |
| **Sparse** | O(L·w) | Local windows (w=8), maximum computational efficiency |

## 📈 Results

### Performance Summary

| Model | Test Accuracy | Avg F1-Score | Parameters | Inference Time |
|-------|---------------|--------------|------------|----------------|
| Baseline | **85.05%** | 0.843 ± 0.129 | 0.11M | 0.06ms |
| Causal | 83.93% | 0.832 ± 0.133 | 0.11M | **0.02ms** |
| Sparse | 83.64% | 0.830 ± 0.136 | 0.11M | **0.03ms** |

### Key Insights

- **Computational Efficiency**: Causal and sparse attention provide 83% and 75% inference time reductions
- **Modulation-Specific Performance**: 
  - Simple modulations (PAM4, CPFSK, GFSK) excel with sparse attention
  - Complex modulations (QAM16, QAM64) prefer full attention
  - All models struggle with WBFM (analog modulation)
- **Error Patterns**: Consistent QAM16/QAM64 confusion across all models

## 📁 File Structure

```
attention_based_automatic_modulation_recognition/
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitignore                   # Git ignore rules
├── 📄 LICENSE                      # MIT license
├── 🐍 train_models.py              # Main training script
├── 🐍 experiment_models.py         # Comprehensive analysis script
├── 🐍 visualize_inputs.py          # Signal visualization script
│
├── 📁 saved_models/                # Generated during training
│   ├── baseline_best.pth           # Best baseline model
│   ├── causal_best.pth             # Best causal model
│   ├── sparse_best.pth             # Best sparse model
│   ├── datasets.pkl                # Preprocessed datasets
│   └── training_results.json       # Training history
│
├── 📁 experiment_results/          # Generated during experiments
│   ├── training_curves_*.pdf       # Individual training curves
│   ├── confusion_matrix_*.pdf      # Individual confusion matrices
│   ├── *_comparison.pdf            # Performance comparisons
│   ├── feature_visualization_*.pdf # t-SNE visualizations
│   ├── performance_table.csv       # Results summary table
│   └── experiment_summary.txt      # Detailed analysis report
│
└── 📁 signal_visualizations/       # Generated during visualization
    ├── signal_overview_grid.pdf     # 4-panel signal overview
    ├── modulation_heatmap_matrix.pdf # I/Q channel matrices
    ├── constellation_comparison.pdf  # I/Q constellation diagrams
    ├── ai_model_input_*.pdf         # AI preprocessing examples
    └── preprocessing_pipeline_*.pdf  # Step-by-step preprocessing
```

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{catak2024attention,
  title={Comparative Analysis of Attention Mechanisms for Automatic Modulation Classification in Radio Frequency Signals},
  author={Catak, Ferhat Ozgur and Kuzlu, Murat and Cali, Umit},
  journal={IEEE Transactions on Cognitive Communications and Networking},
  year={2024},
  publisher={IEEE}
}
```

## 👥 Authors

- **Ferhat Ozgur Catak** - University of Stavanger, Norway ([f.ozgur.catak@uis.no](mailto:f.ozgur.catak@uis.no))
- **Murat Kuzlu** - Old Dominion University, Norfolk, VA, USA ([mkuzlu@odu.edu](mailto:mkuzlu@odu.edu))
- **Umit Cali** - University of York, York, United Kingdom ([umit.cali@york.ac.uk](mailto:umit.cali@york.ac.uk))

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Work

- [RML2016.10a Dataset](https://www.deepsig.ai/datasets)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Attention Mechanisms in Deep Learning](https://arxiv.org/abs/1706.03762)

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📞 Contact

For questions about the code or research, please contact:
- Ferhat Ozgur Catak: [f.ozgur.catak@uis.no](mailto:f.ozgur.catak@uis.no)

---

**⭐ If you find this work useful, please consider starring the repository!**
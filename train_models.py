#!/usr/bin/env python3
"""
Simplified Training Script for Attention Models
==============================================
Trains baseline, causal, and sparse attention models with simplified architecture.
Saves models and datasets for later experimentation.
"""

import os
import pickle
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ======================== DATA LOADING ========================
class RMLDataset(torch.utils.data.Dataset):
    """Simple RML dataset with normalization"""
    def __init__(self, X, y, normalize=True, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment
        
        if normalize:
            self.X = self.normalize_samples(self.X)
    
    def normalize_samples(self, X):
        """Normalize each sample"""
        X_flat = X.view(X.size(0), -1)
        mean = X_flat.mean(dim=1, keepdim=True)
        std = X_flat.std(dim=1, keepdim=True)
        std = torch.clamp(std, min=1e-8)
        X_normalized = (X_flat - mean) / std
        return X_normalized.view(X.size(0), 2, 128)
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment and torch.rand(1) < 0.3:
            # Simple noise augmentation
            noise = torch.randn_like(x) * 0.02
            x = x + noise
        return x, self.y[idx]

def load_rml_data(pkl_path, test_size=0.2, val_size=0.25, min_snr=-6, max_snr=18):
    """Load and split RML2016.10a dataset"""
    print(f"Loading RML2016.10a dataset from {pkl_path}...")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Filter by SNR and build arrays
    mods = sorted({m for (m,s) in data.keys()})
    mod_to_idx = {m:i for i,m in enumerate(mods)}

    X, y = [], []
    for (m,snr), samples in data.items():
        if min_snr <= snr <= max_snr:
            idx = mod_to_idx[m]
            for samp in samples:
                X.append(samp)
                y.append(idx)

    X = np.stack(X)
    y = np.array(y, dtype=int)

    print(f"Loaded {len(mods)} classes: {mods}")
    print(f"Total samples: {len(y)}")

    # Split data
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=val_size, stratify=y_tmp, random_state=42)

    print(f"Train: {len(y_tr)}, Val: {len(y_val)}, Test: {len(y_te)}")

    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te), len(mods), mods

# ======================== SIMPLIFIED ATTENTION MODELS ========================
class MultiHeadAttention(nn.Module):
    """Standard multi-head attention"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = (self.d_k) ** -0.5
        
    def forward(self, x, mask=None):
        batch_size, seq_len = x.size(0), x.size(1)
        
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(out)

class CausalAttention(nn.Module):
    """Causal (masked) attention"""
    def __init__(self, d_model, n_heads, dropout=0.1, max_len=64):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = (self.d_k) ** -0.5
        
        # Causal mask
        self.register_buffer('causal_mask', 
                           torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0))
        
    def forward(self, x, mask=None):
        batch_size, seq_len = x.size(0), x.size(1)
        
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(causal_mask == 0, -1e9)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(out)

class SparseAttention(nn.Module):
    """Sparse attention with local window"""
    def __init__(self, d_model, n_heads, dropout=0.1, local_window=8):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.local_window = local_window
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = (self.d_k) ** -0.5
        
    def create_sparse_mask(self, seq_len, device):
        """Create sparse attention mask"""
        mask = torch.zeros(seq_len, seq_len, device=device)
        
        # Local attention window
        for i in range(seq_len):
            start = max(0, i - self.local_window // 2)
            end = min(seq_len, i + self.local_window // 2 + 1)
            mask[i, start:end] = 1
        
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x, mask=None):
        batch_size, seq_len = x.size(0), x.size(1)
        
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply sparse mask
        sparse_mask = self.create_sparse_mask(seq_len, x.device)
        scores = scores.masked_fill(sparse_mask == 0, -1e9)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(out)

class TransformerBlock(nn.Module):
    """Simplified transformer block"""
    def __init__(self, d_model, n_heads, d_ff, dropout, attention_type='standard'):
        super().__init__()
        
        if attention_type == 'causal':
            self.attention = CausalAttention(d_model, n_heads, dropout)
        elif attention_type == 'sparse':
            self.attention = SparseAttention(d_model, n_heads, dropout)
        else:
            self.attention = MultiHeadAttention(d_model, n_heads, dropout)
            
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class SimplifiedCNNTransformer(nn.Module):
    """Simplified CNN-Transformer model"""
    def __init__(self, num_classes, attention_type='standard', d_model=64, n_heads=4, 
                 n_layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.attention_type = attention_type
        
        # Simplified CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout1d(0.1)
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, attention_type)
            for _ in range(n_layers)
        ])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # CNN feature extraction
        x = self.cnn(x)  # (batch_size, d_model, seq_len)
        x = x.transpose(1, 2)  # (batch_size, seq_len, d_model)
        
        x = self.dropout(x)
        
        # Transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        return self.classifier(x)

# ======================== EARLY STOPPING ========================
class EarlyStopping:
    """Early stopping with model saving"""
    def __init__(self, patience=10, min_delta=0.001, save_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        
    def __call__(self, val_score, model, epoch=None):
        if self.best_score is None:
            self.best_score = val_score
            self.best_epoch = epoch if epoch is not None else 0
            self.save_checkpoint(model, val_score, epoch)
        elif val_score > self.best_score + self.min_delta:
            print(f"    ✅ Val accuracy improved: {self.best_score:.3f} → {val_score:.3f}")
            self.best_score = val_score
            self.best_epoch = epoch if epoch is not None else 0
            self.counter = 0
            self.save_checkpoint(model, val_score, epoch)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"    🛑 Early stopping at epoch {epoch}")
                return True
        return False
    
    def save_checkpoint(self, model, val_score, epoch=None):
        """Save model checkpoint"""
        if self.save_path is not None:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'best_val_acc': val_score,
                'best_epoch': self.best_epoch,
                'final_epoch': epoch,
                'model_config': {
                    'attention_type': getattr(model, 'attention_type', 'standard'),
                    'num_classes': model.classifier[-1].out_features,
                    'd_model': model.d_model
                }
            }
            torch.save(checkpoint, self.save_path)

# ======================== TRAINER ========================
class ModelTrainer:
    """Model trainer"""
    def __init__(self, device='auto', save_dir='saved_models'):
        self.device = self._setup_device(device)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def _setup_device(self, device):
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)

    def train_model(self, model, train_data, val_data, model_name, num_epochs=50, batch_size=128, patience=15):
        """Train a single model"""
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            RMLDataset(*train_data, normalize=True, augment=True),
            batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            RMLDataset(*val_data, normalize=True, augment=False),
            batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )
        
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Early stopping
        model_path = os.path.join(self.save_dir, f"{model_name}_best.pth")
        early_stopping = EarlyStopping(patience=patience, save_path=model_path)

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{num_epochs}", leave=False):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                train_correct += pred.eq(target).sum().item()
                train_total += data.size(0)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item() * data.size(0)
                    pred = output.argmax(dim=1)
                    val_correct += pred.eq(target).sum().item()
                    val_total += data.size(0)
            
            # Calculate metrics
            train_loss /= train_total
            train_acc = 100. * train_correct / train_total
            val_loss /= val_total
            val_acc = 100. * val_correct / val_total
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1:3d} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.2e}")
            
            scheduler.step()
            
            # Early stopping
            if early_stopping(val_acc, model, epoch + 1):
                break
        
        print(f"✅ Best validation accuracy: {early_stopping.best_score:.2f}%")
        return early_stopping.best_score, history

def main():
    parser = argparse.ArgumentParser(description='Train simplified attention models')
    parser.add_argument('--data_path', type=str, default='RML2016.10a_dict.pkl',
                       help='Path to RML dataset')
    parser.add_argument('--save_dir', type=str, default='saved_models',
                       help='Directory to save models and data')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Maximum epochs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use')
    
    # patience for early stopping
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    print("🚀 SIMPLIFIED ATTENTION MODEL TRAINING")
    print("=" * 60)
    
    # Check data file
    if not os.path.exists(args.data_path):
        print(f"❌ Data file not found: {args.data_path}")
        return
    
    # Load and split data
    train_data, val_data, test_data, num_classes, class_names = load_rml_data(args.data_path)
    
    # Save datasets
    os.makedirs(args.save_dir, exist_ok=True)
    datasets = {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'num_classes': num_classes,
        'class_names': class_names
    }
    with open(os.path.join(args.save_dir, 'datasets.pkl'), 'wb') as f:
        pickle.dump(datasets, f)
    print(f"✅ Datasets saved to {args.save_dir}/datasets.pkl")
    
    # Initialize trainer
    trainer = ModelTrainer(device=args.device, save_dir=args.save_dir)
    
    # Model variants to train
    variants = ['baseline', 'causal', 'sparse']
    results = {}
    
    # Train each variant
    for variant in variants:
        print(f"\n[{variants.index(variant)+1}/{len(variants)}] Training {variant}...")
        
        model = SimplifiedCNNTransformer(
            num_classes=num_classes,
            attention_type='standard' if variant == 'baseline' else variant,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            dropout=0.1
        )
        
        best_acc, history = trainer.train_model(
            model, train_data, val_data, variant, 
            num_epochs=args.num_epochs, batch_size=args.batch_size,
            patience=args.patience
        )
        
        results[variant] = {
            'best_val_acc': best_acc,
            'history': history
        }
    
    # Save training results
    with open(os.path.join(args.save_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TRAINING COMPLETED!")
    print("=" * 60)
    print("Results Summary:")
    for variant, result in sorted(results.items(), key=lambda x: x[1]['best_val_acc'], reverse=True):
        print(f"  {variant:10s}: {result['best_val_acc']:.2f}%")
    
    print(f"\n📁 All files saved to: {args.save_dir}")
    print("🔬 Ready for experiments! Run experiment_models.py next.")

if __name__ == "__main__":
    main()
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the emulator path
sys.path.insert(0, './projects/lsst_y1/')
from emulator import ResMLP, ResTRF

# ============================================================================
# CONFIGURATION
# ============================================================================
LOW_ACCURACY_MODEL_PATH = './projects/lsst_y1/emulators/xi_low_accuracy'
HIGH_ACCURACY_MODEL_PATH = './projects/lsst_y1/emulators/xi_high_accuracy'  # You may need to adjust this
DATA_PATH = './projects/lsst_y1/data/lsst_y1_train.dataset'

# ============================================================================
# STEP 1: Load the dataset
# ============================================================================
print("Loading dataset...")
data = np.load(DATA_PATH, allow_pickle=True).item()

# Extract inputs (cosmological parameters) and outputs (xi correlation functions)
X_train = data['X_train']  # Cosmological parameters
y_train = data['y_train']  # True xi values
X_test = data.get('X_test', None)
y_test = data.get('y_test', None)

# If no test set exists, create one by splitting
if X_test is None:
    print("No test set found, creating one from training data...")
    # Use last 10% as test set
    split_idx = int(0.9 * len(X_train))
    X_test = X_train[split_idx:]
    y_test = y_train[split_idx:]
    X_train = X_train[:split_idx]
    y_train = y_train[:split_idx]

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Input dimension: {X_train.shape[1]}")
print(f"Output dimension: {y_train.shape[1]}")

# ============================================================================
# STEP 2: Load the trained models
# ============================================================================
def load_model(model_path):
    """Load a trained emulator model"""
    # Load the model metadata
    import h5py
    
    with h5py.File(f"{model_path}.h5", 'r') as f:
        model_type = f.attrs.get('model_type', 'ResMLP')
        input_dim = f.attrs['input_dim']
        output_dim = f.attrs['output_dim']
        hidden_dim = f.attrs.get('hidden_dim', 512)
        num_layers = f.attrs.get('num_layers', 5)
        
        # Load normalization parameters
        X_mean = f['X_mean'][:]
        X_std = f['X_std'][:]
        y_mean = f['y_mean'][:]
        y_std = f['y_std'][:]
    
    # Create model
    if model_type == 'ResMLP':
        model = ResMLP(input_dim, output_dim, hidden_dim, num_layers)
    else:
        model = ResTRF(input_dim, output_dim, hidden_dim, num_layers)
    
    # Load weights
    model.load_state_dict(torch.load(f"{model_path}.pth", map_location='cpu'))
    model.eval()
    
    return model, X_mean, X_std, y_mean, y_std

print("\nLoading low accuracy model...")
low_acc_model, low_X_mean, low_X_std, low_y_mean, low_y_std = load_model(LOW_ACCURACY_MODEL_PATH)

print("Loading high accuracy model...")
# If high accuracy model doesn't exist yet, use low accuracy for demonstration
try:
    high_acc_model, high_X_mean, high_X_std, high_y_mean, high_y_std = load_model(HIGH_ACCURACY_MODEL_PATH)
except:
    print("High accuracy model not found, using low accuracy for both (for demonstration)")
    high_acc_model, high_X_mean, high_X_std, high_y_mean, high_y_std = low_acc_model, low_X_mean, low_X_std, low_y_mean, low_y_std

# ============================================================================
# STEP 3: Make predictions
# ============================================================================
def predict(model, X, X_mean, X_std, y_mean, y_std):
    """Make normalized predictions"""
    # Normalize inputs
    X_norm = (X - X_mean) / X_std
    
    # Convert to torch and predict
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_norm)
        y_pred_norm = model(X_tensor).numpy()
    
    # Denormalize outputs
    y_pred = y_pred_norm * y_std + y_mean
    
    return y_pred

print("\nMaking predictions...")
y_pred_low = predict(low_acc_model, X_test, low_X_mean, low_X_std, low_y_mean, low_y_std)
y_pred_high = predict(high_acc_model, X_test, high_X_mean, high_X_std, high_y_mean, high_y_std)

# ============================================================================
# STEP 4: Calculate fractional differences
# ============================================================================
def fractional_difference(y_true, y_pred):
    """Calculate fractional difference: (pred - true) / true"""
    return (y_pred - y_true) / (y_true + 1e-10)  # Add small value to avoid division by zero

frac_diff_low = fractional_difference(y_test, y_pred_low)
frac_diff_high = fractional_difference(y_test, y_pred_high)

# ============================================================================
# STEP 5: Create plots
# ============================================================================
print("\nCreating plots...")

# Select a few test samples to plot
n_samples = min(5, len(X_test))
sample_indices = np.random.choice(len(X_test), n_samples, replace=False)

# PLOT 1: Compare high vs low accuracy predictions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comparison: High Accuracy vs Low Accuracy Models', fontsize=16)

for i, idx in enumerate(sample_indices[:4]):
    ax = axes[i//2, i%2]
    
    # Plot truth
    ax.plot(y_test[idx], 'k-', linewidth=2, label='Truth', alpha=0.7)
    
    # Plot predictions
    ax.plot(y_pred_low[idx], 'b--', linewidth=1.5, label='Low Accuracy', alpha=0.7)
    ax.plot(y_pred_high[idx], 'r:', linewidth=1.5, label='High Accuracy', alpha=0.7)
    
    ax.set_xlabel('Data Vector Index')
    ax.set_ylabel('ξ (correlation function)')
    ax.set_title(f'Test Sample {idx}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot1_model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: plot1_model_comparison.png")

# PLOT 2: Truth data vectors and fractional differences
fig, axes = plt.subplots(n_samples, 2, figsize=(14, 4*n_samples))
fig.suptitle('Truth Data Vectors and Fractional Differences', fontsize=16)

for i, idx in enumerate(sample_indices):
    # Left: Truth data vector
    axes[i, 0].plot(y_test[idx], 'k-', linewidth=2, label='Truth')
    axes[i, 0].set_xlabel('Data Vector Index')
    axes[i, 0].set_ylabel('ξ (correlation function)')
    axes[i, 0].set_title(f'Truth - Sample {idx}')
    axes[i, 0].grid(True, alpha=0.3)
    axes[i, 0].legend()
    
    # Right: Fractional difference (using low accuracy as example)
    axes[i, 1].plot(frac_diff_low[idx], 'b-', linewidth=1.5, label='Frac. Diff (Low Acc)')
    axes[i, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[i, 1].set_xlabel('Data Vector Index')
    axes[i, 1].set_ylabel('(Pred - Truth) / Truth')
    axes[i, 1].set_title(f'Fractional Difference - Sample {idx}')
    axes[i, 1].grid(True, alpha=0.3)
    axes[i, 1].legend()

plt.tight_layout()
plt.savefig('plot2_truth_and_frac_diff.png', dpi=300, bbox_inches='tight')
print("Saved: plot2_truth_and_frac_diff.png")

# PLOT 3: NN predicted data vectors and their fractional differences
fig, axes = plt.subplots(n_samples, 2, figsize=(14, 4*n_samples))
fig.suptitle('NN Predicted Data Vectors and Fractional Differences', fontsize=16)

for i, idx in enumerate(sample_indices):
    # Left: NN predictions
    axes[i, 0].plot(y_test[idx], 'k-', linewidth=2, label='Truth', alpha=0.5)
    axes[i, 0].plot(y_pred_low[idx], 'b--', linewidth=1.5, label='Low Acc Prediction')
    axes[i, 0].plot(y_pred_high[idx], 'r:', linewidth=1.5, label='High Acc Prediction')
    axes[i, 0].set_xlabel('Data Vector Index')
    axes[i, 0].set_ylabel('ξ (correlation function)')
    axes[i, 0].set_title(f'NN Predictions - Sample {idx}')
    axes[i, 0].legend()
    axes[i, 0].grid(True, alpha=0.3)
    
    # Right: Fractional differences
    axes[i, 1].plot(frac_diff_low[idx], 'b-', linewidth=1.5, label='Low Acc Frac. Diff')
    axes[i, 1].plot(frac_diff_high[idx], 'r-', linewidth=1.5, label='High Acc Frac. Diff')
    axes[i, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[i, 1].set_xlabel('Data Vector Index')
    axes[i, 1].set_ylabel('(Pred - Truth) / Truth')
    axes[i, 1].set_title(f'Fractional Difference - Sample {idx}')
    axes[i, 1].legend()
    axes[i, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot3_predictions_and_frac_diff.png', dpi=300, bbox_inches='tight')
print("Saved: plot3_predictions_and_frac_diff.png")

# PLOT 4: Summary statistics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Summary Statistics Across All Test Samples', fontsize=16)

# Mean absolute fractional difference
axes[0, 0].hist(np.abs(frac_diff_low).mean(axis=1), bins=30, alpha=0.7, label='Low Accuracy', color='blue')
axes[0, 0].hist(np.abs(frac_diff_high).mean(axis=1), bins=30, alpha=0.7, label='High Accuracy', color='red')
axes[0, 0].set_xlabel('Mean |Fractional Difference|')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Distribution of Mean Absolute Fractional Error')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Max absolute fractional difference
axes[0, 1].hist(np.abs(frac_diff_low).max(axis=1), bins=30, alpha=0.7, label='Low Accuracy', color='blue')
axes[0, 1].hist(np.abs(frac_diff_high).max(axis=1), bins=30, alpha=0.7, label='High Accuracy', color='red')
axes[0, 1].set_xlabel('Max |Fractional Difference|')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Distribution of Maximum Absolute Fractional Error')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Mean fractional difference per data vector element
axes[1, 0].plot(frac_diff_low.mean(axis=0), 'b-', linewidth=2, label='Low Accuracy', alpha=0.7)
axes[1, 0].plot(frac_diff_high.mean(axis=0), 'r-', linewidth=2, label='High Accuracy', alpha=0.7)
axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Data Vector Index')
axes[1, 0].set_ylabel('Mean Fractional Difference')
axes[1, 0].set_title('Mean Fractional Difference per Element')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Std of fractional difference per data vector element
axes[1, 1].plot(frac_diff_low.std(axis=0), 'b-', linewidth=2, label='Low Accuracy', alpha=0.7)
axes[1, 1].plot(frac_diff_high.std(axis=0), 'r-', linewidth=2, label='High Accuracy', alpha=0.7)
axes[1, 1].set_xlabel('Data Vector Index')
axes[1, 1].set_ylabel('Std of Fractional Difference')
axes[1, 1].set_title('Fractional Difference Std per Element')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot4_summary_statistics.png', dpi=300, bbox_inches='tight')
print("Saved: plot4_summary_statistics.png")

# ============================================================================
# STEP 6: Print summary metrics
# ============================================================================
print("\n" + "="*60)
print("SUMMARY METRICS")
print("="*60)
print(f"\nLow Accuracy Model:")
print(f"  Mean absolute fractional error: {np.abs(frac_diff_low).mean():.6f}")
print(f"  Median absolute fractional error: {np.median(np.abs(frac_diff_low)):.6f}")
print(f"  Max absolute fractional error: {np.abs(frac_diff_low).max():.6f}")

print(f"\nHigh Accuracy Model:")
print(f"  Mean absolute fractional error: {np.abs(frac_diff_high).mean():.6f}")
print(f"  Median absolute fractional error: {np.median(np.abs(frac_diff_high)):.6f}")
print(f"  Max absolute fractional error: {np.abs(frac_diff_high).max():.6f}")

print("\nAll plots saved successfully!")
print("\nGenerated files:")
print("  - plot1_model_comparison.png")
print("  - plot2_truth_and_frac_diff.png")
print("  - plot3_predictions_and_frac_diff.png")
print("  - plot4_summary_statistics.png")

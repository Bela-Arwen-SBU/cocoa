#!/bin/bash
# Comprehensive overnight experiments - 500 epochs each with losses and testing metrics
echo "Starting comprehensive hyperparameter experiments at $(date)"

RESULTS_DIR="hyperparameter_results"
LOSSES_DIR="$RESULTS_DIR/losses"
METRICS_DIR="$RESULTS_DIR/testing_metrics"

# ============================================================================
# 1. WEIGHT DECAY SWEEP (5 tests)
# ============================================================================
echo ""
echo "=== WEIGHT DECAY SWEEP ==="
echo ""

echo "Running weight decay 0..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --weight_decay 0 --epochs 500
mv losses.txt $LOSSES_DIR/losses_wd_0.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_wd_0.txt

echo "Running weight decay 1e-2..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --weight_decay 1e-2 --epochs 500
mv losses.txt $LOSSES_DIR/losses_wd_1e2.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_wd_1e2.txt

echo "Running weight decay 1e-3..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --weight_decay 1e-3 --epochs 500
mv losses.txt $LOSSES_DIR/losses_wd_1e3.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_wd_1e3.txt

echo "Running weight decay 1e-4..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --weight_decay 1e-4 --epochs 500
mv losses.txt $LOSSES_DIR/losses_wd_1e4.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_wd_1e4.txt

echo "Running weight decay 1e-5..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --weight_decay 1e-5 --epochs 500
mv losses.txt $LOSSES_DIR/losses_wd_1e5.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_wd_1e5.txt

# ============================================================================
# 2. LEARNING RATE SWEEP (6 tests)
# ============================================================================
echo ""
echo "=== LEARNING RATE SWEEP ==="
echo ""

echo "Running learning rate 1e-2..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --learning_rate 1e-2 --epochs 500
mv losses.txt $LOSSES_DIR/losses_lr_1e2.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_lr_1e2.txt

echo "Running learning rate 5e-3..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --learning_rate 5e-3 --epochs 500
mv losses.txt $LOSSES_DIR/losses_lr_5e3.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_lr_5e3.txt

echo "Running learning rate 1e-3..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --learning_rate 1e-3 --epochs 500
mv losses.txt $LOSSES_DIR/losses_lr_1e3.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_lr_1e3.txt

echo "Running learning rate 5e-4..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --learning_rate 5e-4 --epochs 500
mv losses.txt $LOSSES_DIR/losses_lr_5e4.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_lr_5e4.txt

echo "Running learning rate 1e-4..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --learning_rate 1e-4 --epochs 500
mv losses.txt $LOSSES_DIR/losses_lr_1e4.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_lr_1e4.txt

echo "Running learning rate 1e-5..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --learning_rate 1e-5 --epochs 500
mv losses.txt $LOSSES_DIR/losses_lr_1e5.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_lr_1e5.txt

# ============================================================================
# 3. BATCH SIZE VALIDATION (2 tests)
# ============================================================================
echo ""
echo "=== BATCH SIZE VALIDATION ==="
echo ""

echo "Running batch size 32..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --batch_size 32 --epochs 500
mv losses.txt $LOSSES_DIR/losses_bs_32.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_bs_32.txt

echo "Running batch size 256..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --batch_size 256 --epochs 500
mv losses.txt $LOSSES_DIR/losses_bs_256.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_bs_256.txt

# ============================================================================
# 4. MIXED HYPERPARAMETER COMBINATIONS (8 tests)
# ============================================================================
echo ""
echo "=== MIXED HYPERPARAMETER COMBINATIONS ==="
echo ""

echo "Running WD=0 + LR=1e-3..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --weight_decay 0 --learning_rate 1e-3 --epochs 500
mv losses.txt $LOSSES_DIR/losses_wd0_lr1e3.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_wd0_lr1e3.txt

echo "Running WD=0 + LR=5e-4..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --weight_decay 0 --learning_rate 5e-4 --epochs 500
mv losses.txt $LOSSES_DIR/losses_wd0_lr5e4.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_wd0_lr5e4.txt

echo "Running WD=0 + LR=1e-4..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --weight_decay 0 --learning_rate 1e-4 --epochs 500
mv losses.txt $LOSSES_DIR/losses_wd0_lr1e4.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_wd0_lr1e4.txt

echo "Running WD=1e-5 + LR=1e-3..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --weight_decay 1e-5 --learning_rate 1e-3 --epochs 500
mv losses.txt $LOSSES_DIR/losses_wd1e5_lr1e3.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_wd1e5_lr1e3.txt

echo "Running WD=1e-5 + LR=5e-4..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --weight_decay 1e-5 --learning_rate 5e-4 --epochs 500
mv losses.txt $LOSSES_DIR/losses_wd1e5_lr5e4.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_wd1e5_lr5e4.txt

echo "Running WD=1e-5 + LR=1e-4..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --weight_decay 1e-5 --learning_rate 1e-4 --epochs 500
mv losses.txt $LOSSES_DIR/losses_wd1e5_lr1e4.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_wd1e5_lr1e4.txt

echo "Running WD=1e-4 + LR=1e-3..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --weight_decay 1e-4 --learning_rate 1e-3 --epochs 500
mv losses.txt $LOSSES_DIR/losses_wd1e4_lr1e3.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_wd1e4_lr1e3.txt

echo "Running WD=1e-4 + LR=5e-4..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --weight_decay 1e-4 --learning_rate 5e-4 --epochs 500
mv losses.txt $LOSSES_DIR/losses_wd1e4_lr5e4.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_wd1e4_lr5e4.txt

# ============================================================================
# 5. HIGH ACCURACY MODEL TESTS (3 tests)
# ============================================================================
echo ""
echo "=== HIGH ACCURACY MODEL TESTS ==="
echo ""

echo "Running high accuracy baseline..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_high_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --epochs 500
mv losses.txt $LOSSES_DIR/losses_high_accuracy_baseline.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_high_accuracy_baseline.txt

echo "Running high accuracy with WD=1e-5 + LR=5e-4..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_high_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --weight_decay 1e-5 --learning_rate 5e-4 --epochs 500
mv losses.txt $LOSSES_DIR/losses_high_accuracy_wd1e5_lr5e4.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_high_accuracy_wd1e5_lr5e4.txt

echo "Running high accuracy with WD=0 + LR=1e-3..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_high_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --save_testing_metrics True --weight_decay 0 --learning_rate 1e-3 --epochs 500
mv losses.txt $LOSSES_DIR/losses_high_accuracy_wd0_lr1e3.txt
mv testing_metrics.txt $METRICS_DIR/testing_metrics_high_accuracy_wd0_lr1e3.txt

# ============================================================================
# COMPLETE!
# ============================================================================
echo ""
echo "=========================================="
echo "All experiments completed at $(date)"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  Losses: $LOSSES_DIR"
echo "  Testing metrics: $METRICS_DIR"
echo ""
echo "Total files created:"
ls -1 $LOSSES_DIR/*.txt | wc -l
ls -1 $METRICS_DIR/*.txt | wc -l

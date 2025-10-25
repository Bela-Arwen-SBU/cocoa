#!/bin/bash

# Simple overnight experiments
echo "Starting experiments at $(date)"

# Weight decay experiments
echo "=== Weight Decay Experiments ==="

echo "Running weight decay 1e-1..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --weight_decay 1e-1 --epochs 300
mv losses.txt losses_wd_1e1.txt

echo "Running weight decay 1e-3..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --weight_decay 1e-3 --epochs 300
mv losses.txt losses_wd_1e3.txt

echo "Running weight decay 1e-5..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --weight_decay 1e-5 --epochs 300
mv losses.txt losses_wd_1e5.txt

echo "Running weight decay 0..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --weight_decay 0 --epochs 300
mv losses.txt losses_wd_0.txt

# Extended training experiments
echo "=== Extended Training Experiments ==="

echo "Running extended training 500 epochs..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --epochs 500
mv losses.txt losses_extended_500.txt

echo "Running extended training 600 epochs..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --epochs 600
mv losses.txt losses_extended_600.txt

# High accuracy test
echo "=== High Accuracy Test ==="

echo "Running high accuracy..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_high_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --epochs 300
mv losses.txt losses_high_accuracy.txt

# More batch size + weight decay combinations
echo "=== Batch Size + Weight Decay Combinations ==="

echo "Running batch 32 with weight decay 1e-3..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --batch_size 32 --weight_decay 1e-3 --epochs 300
mv losses.txt losses_bs32_wd1e3.txt

echo "Running batch 32 with weight decay 0..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --batch_size 32 --weight_decay 0 --epochs 300
mv losses.txt losses_bs32_wd0.txt

echo "Running batch 1000 with weight decay 1e-3..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --batch_size 1000 --weight_decay 1e-3 --epochs 300
mv losses.txt losses_bs1000_wd1e3.txt

echo "Running batch 1000 with weight decay 1e-5..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --batch_size 1000 --weight_decay 1e-5 --epochs 300
mv losses.txt losses_bs1000_wd1e5.txt

# Different starting learning rates
echo "=== Different Learning Rates ==="

echo "Running with learning rate 1e-4..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --learning_rate 1e-4 --epochs 400
mv losses.txt losses_lr1e4.txt

echo "Running with learning rate 5e-4..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --learning_rate 5e-4 --epochs 400
mv losses.txt losses_lr5e4.txt

# More extended training attempts
echo "=== More Extended Training ==="

echo "Running 700 epochs..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --epochs 700
mv losses.txt losses_extended_700.txt

echo "Running 800 epochs..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --epochs 800
mv losses.txt losses_extended_800.txt

# High accuracy with different settings
echo "=== High Accuracy Variations ==="

echo "Running high accuracy with 400 epochs..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_high_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --epochs 400
mv losses.txt losses_high_accuracy_400ep.txt

echo "Running high accuracy with batch 32..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_high_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --batch_size 32 --epochs 300
mv losses.txt losses_high_accuracy_bs32.txt

# Best combination attempts
echo "=== Best Combinations ==="

echo "Running batch 256, wd 1e-5, 400 epochs..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --batch_size 256 --weight_decay 1e-5 --epochs 400
mv losses.txt losses_best_combo1.txt

echo "Running batch 32, wd 0, 500 epochs..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --batch_size 32 --weight_decay 0 --epochs 500
mv losses.txt losses_best_combo2.txt

echo "All experiments completed at $(date)"
echo "Loss files created:"
ls -la losses_*.txt

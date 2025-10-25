#!/bin/bash
# Covariance squeezing experiments
echo "Starting covariance squeezing experiments at $(date)"

echo "=== Covariance Squeezing Tests ==="

echo "Running squeeze factor 2..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --squeeze_factor 2 --epochs 500
mv losses.txt hyperparameter_results/losses/losses_squeeze_2.txt

echo "Running squeeze factor 4..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --squeeze_factor 4 --epochs 500
mv losses.txt hyperparameter_results/losses/losses_squeeze_4.txt

echo "Running squeeze factor 8..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --squeeze_factor 8 --epochs 500
mv losses.txt hyperparameter_results/losses/losses_squeeze_8.txt

echo "Running squeeze factor 16..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --squeeze_factor 16 --epochs 500
mv losses.txt hyperparameter_results/losses/losses_squeeze_16.txt

echo "Running squeeze factor 32..."
python ./projects/lsst_y1/train_emulator.py --yaml ./projects/lsst_y1/xi_emulator_low_accuracy.yaml --probe cosmic_shear --device cpu --save_losses True --squeeze_factor 32 --epochs 500
mv losses.txt hyperparameter_results/losses/losses_squeeze_32.txt

echo "All covariance squeezing experiments completed at $(date)"
echo "Loss files created:"
ls -la hyperparameter_results/losses/losses_squeeze_*.txt

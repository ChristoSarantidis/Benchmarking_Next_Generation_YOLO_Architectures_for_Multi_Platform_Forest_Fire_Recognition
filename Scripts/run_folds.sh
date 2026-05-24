#!/bin/bash
echo "Running all folds separately to avoid memory issues..."


    python3 raspi_single_fold.py 1
    echo "Fold 1 complete, memory freed."
    sleep 5  # short pause between folds

    python3 raspi_single_fold.py 2
    echo "Fold 2 complete, memory freed."
    sleep 5  # short pause between folds

    python3 raspi_single_fold.py 3
    echo "Fold 3 complete, memory freed."
    sleep 5  # short pause between folds

    python3 raspi_single_fold.py 4
    echo "Fold 4 complete, memory freed."
    sleep 5  # short pause between folds

    python3 raspi_single_fold.py 5
    echo "Fold 5 complete, memory freed."
    sleep 5  # short pause between folds

echo "All folds complete!"
python3 merge_results.py
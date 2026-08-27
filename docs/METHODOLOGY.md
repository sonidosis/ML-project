# Methodology

This document describes only behavior traceable to the preserved 2022 notebooks and `Report.pdf`. It was added during the 2026 portfolio documentation update; it does not alter the original implementation.

## Problem Definition

The project is a ten-class image-classification task for 300 x 300 RGB traffic-sign and traffic-light images. The classes encoded as integers 0 through 9 are Stop, Yield, Red Light, Green Light, Roundabout, Right Turn Only, Do Not Enter, Crosswalk, Handicap Parking, and No Parking.

## Data Preparation

`Final Project - Training Data.ipynb` loads `data_train.npy` and `labels_train.npy`, counts samples by class, and displays examples from every class. The report states that the original set contained 6,194 images, that 30 labels were manually corrected, and that 97 difficult/noisy images were removed during preprocessing. The repository does not contain the arrays needed to independently verify or rerun those operations.

`Model_Train.ipynb` loads `data_train_vars.npy` and `labels_train_corr_vars.npy`, retains the first 5,500 samples, reshapes images to 300 x 300 x 3 TensorFlow tensors with `float16` dtype, and divides pixel values by 255.

## Original Partition Logic

The training notebook uses scikit-learn's `train_test_split` twice with shuffling and stratification:

1. Fifteen percent of the 5,500 samples becomes an internal test partition.
2. Twenty percent of the remaining 85% becomes validation data.
3. The remainder becomes training data.

No `random_state` is specified, so the original split is not deterministic. The internal test partition is prepared but is not evaluated by the notebook's returned metric; the function returns accuracy on the training partition. Separately, `Model_Test.ipynb` evaluates the first 1,000 samples from the same source arrays, which can overlap the development data.

## CNN Architecture

The preserved training notebook defines this sequence:

1. Random horizontal/vertical flip and random rotation by factor 0.2
2. Conv2D: 64 filters, 3 x 3 kernel, stride 2, ReLU, same padding
3. Dropout: 0.2
4. MaxPool2D: pool size 2
5. Conv2D: 128 filters, 3 x 3 kernel, stride 2, ReLU, same padding
6. Conv2D: 128 filters, 3 x 3 kernel, stride 2, ReLU, same padding
7. MaxPool2D: pool size 2
8. Conv2D: 256 filters, 3 x 3 kernel, stride 2, ReLU, same padding
9. Conv2D: 256 filters, 3 x 3 kernel, stride 2, ReLU, same padding
10. MaxPool2D: pool size 2
11. Flatten
12. Dense: 80 ReLU units, L1/L2 regularization of 0.001/0.001
13. Dropout: 0.1
14. Dense: 80 ReLU units, L1/L2 regularization of 0.001/0.001
15. Dropout: 0.1
16. Dense: 10 softmax outputs

The model uses sparse categorical cross-entropy, Nadam with a 0.001 learning rate, and accuracy as the training metric. Fitting is configured for 50 epochs with batch size 100, validation monitoring, early stopping with patience 30, and `ModelCheckpoint('CNN.h5', save_best_only=True)`.

The report describes the wider experimental process that led to the final architecture, including GridSearchCV experiments, overlapping max pooling, dropout, L1/L2 regularization, and TensorFlow/Keras 2.7 on the University of Florida HiPerGator environment. Some report settings differ from the final preserved notebook; both are retained as historical records rather than reconciled speculatively.

## Corrected Evaluation Design for a Future Rerun

When the complete, authorized original dataset is available, a rigorous reproduction should:

1. Define one fixed random seed and one deterministic stratified split into training, validation, and independent holdout partitions.
2. Fit preprocessing and the model using training data only.
3. Use validation data only for early stopping, model selection, and tuning.
4. Keep holdout samples unavailable to training, tuning, and model selection.
5. Evaluate the selected model on the holdout set exactly once.
6. Report overall accuracy, precision, recall, F1 score, a confusion matrix, and per-class support, together with the seed, split sizes, environment, and model-selection rule.

Execution of this corrected design remains pending because the dataset is absent from the repository.


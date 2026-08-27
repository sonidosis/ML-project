# Results and Limitations

This 2026 documentation distinguishes preserved academic outputs from any future reproduction. No new experiment was run for this update.

## Original Recorded Results

`Model_Test.ipynb` records:

```text
Model Accuracy: 0.956
```

This value comes from evaluating the first 1,000 samples of `data_train_vars.npy` and `labels_train_corr_vars.npy` with `CNN_overlap.h5`.

The original report also describes iterative model-development results of approximately 96% test accuracy on the preprocessed data and 94% on the original correctly labeled data. It reports class-wise precision means and standard deviations across 16 model changes. These values are historical project records, not metrics reproduced during the 2026 documentation update.

## Why 95.6% Is Not an Independent Holdout Claim

`Model_Train.ipynb` uses the first 5,500 samples from `data_train_vars.npy` during model development. `Model_Test.ipynb` then evaluates the first 1,000 samples from that same array. Those ranges overlap. The repository also does not establish that the loaded `CNN_overlap.h5` was selected without exposure to those samples.

For these reasons, the recorded 95.6% is retained as a legacy evaluation result but is not presented as independent holdout accuracy.

## Missing Artifacts

The repository does not distribute:

- `data_train.npy`
- `labels_train.npy`
- `data_train_vars.npy`
- `labels_train_corr_vars.npy`
- `CNN.h5`
- `CNN_overlap.h5`

Without the underlying arrays, dataset provenance and preprocessing cannot be independently reconstructed from this repository alone. Without the model file, the legacy evaluation cannot be rerun exactly. Exact versions of NumPy, Matplotlib, scikit-learn, Jupyter, and Python from the 2022 environment are also not recorded in the repository. The report documents TensorFlow 2.7 with Keras 2.7.

## Requirements for a Rigorous Rerun

A clean reproduction requires authorized access to the complete original images and labels (before any development split), a traceable implementation of the documented label corrections and exclusions, and a deterministic stratified partition fixed before model development. Training and tuning must use only training/validation data; the independent holdout must be evaluated exactly once after the final model is selected.

The rerun report should include the random seed, exact class counts and split sizes, software/hardware environment, model checkpoint-selection rule, accuracy, precision, recall, F1 score, confusion matrix, and per-class support. It should label all such outputs as a new reproduction, separate from the 2022 results.

## Current Status

No new independent test accuracy is claimed because the original dataset was not available for a clean rerun. No synthetic metrics or performance plots were created.


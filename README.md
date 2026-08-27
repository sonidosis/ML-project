# Traffic Sign Classification with a Convolutional Neural Network

**Graduate Machine Learning Project - University of Florida, EEL 5840, Summer 2022**

> **Project provenance:** The original academic implementation was completed in Summer 2022 as part of EEL 5840 - Fundamentals of Machine Learning at the University of Florida. The original notebooks and report are preserved in this repository. Portfolio documentation and reproducibility improvements were added in 2026.

## Project Overview

This team project developed a convolutional neural network (CNN) to classify 300 x 300 color images into ten traffic-sign and traffic-light categories:

1. Stop
2. Yield
3. Red Light
4. Green Light
5. Roundabout
6. Right Turn Only
7. Do Not Enter
8. Crosswalk
9. Handicap Parking
10. No Parking

The original report credits Justin Gilmore, Ryan Heras, and Paul W. Davis as authors. This repository preserves the project artifacts; it does not claim that all team work was completed by one author.

## Engineering and ML Workflow

The preserved implementation covers image/data preparation, stratified train/validation/test partitioning, tensor conversion, pixel normalization, data augmentation, CNN construction, training, validation, checkpointing, evaluation, and classification analysis.

The training notebook loads the first 5,500 samples from `data_train_vars.npy`. It makes a stratified 85/15 development/test split, then a stratified 80/20 training/validation split of the development portion. The split does not set a random seed. Images are reshaped to 300 x 300 x 3 tensors and normalized by dividing pixel values by 255.

## CNN Architecture

`Model_Train.ipynb` defines a TensorFlow/Keras `Sequential` model with:

- Random horizontal/vertical flips and random rotation (`0.2`)
- Five 3 x 3 `Conv2D` layers with 64, 128, 128, 256, and 256 filters
- Three `MaxPool2D` layers
- Dropout after the first convolution (`0.2`) and after each dense layer (`0.1`)
- Two 80-unit ReLU dense layers with L1/L2 regularization (`0.001` each)
- A 10-unit softmax output
- Sparse categorical cross-entropy loss and the Nadam optimizer (learning rate `0.001`)
- Up to 50 epochs, batch size 100, `EarlyStopping` with patience 30, and best-model checkpointing to `CNN.h5`

These details describe the preserved training notebook. The report also documents the broader model-development experiments, which used some different training settings during iterative tuning.

## Technologies

- Python
- TensorFlow and Keras
- NumPy
- scikit-learn
- Matplotlib
- Jupyter Notebook
- Git and GitHub

## Results

### Original / Legacy Recorded Results

The preserved `Model_Test.ipynb` records `Model Accuracy: 0.956` for 1,000 samples. The legacy test notebook draws those samples from the same `data_train_vars.npy` source array used during model development; the training notebook uses the first 5,500 source samples. Consequently, these selections can overlap.

The recorded 95.6% value is preserved as an original project result, but it is **not presented as an independent holdout-test accuracy**. The original report separately describes iterative testing results of approximately 96% on preprocessed data and 94% on the original correctly labeled data; these are historical results, not new reproductions.

### Reproducible Holdout Evaluation

A new independent holdout result is intentionally not claimed because the complete original dataset and trained model are not currently distributed with this portfolio repository. The reproducibility work below documents the intended evaluation procedure without fabricating a replacement score.

## Repository Structure

```text
.
|-- Final Project - Training Data.ipynb  # Original data-inspection notebook (2022)
|-- Model_Train.ipynb                    # Original CNN training notebook (2022)
|-- Model_Test.ipynb                     # Original legacy evaluation notebook (2022)
|-- Report.pdf                           # Original team project report (2022)
|-- README.md                            # Portfolio overview (updated 2026)
|-- requirements.txt                     # Python dependency guidance (added 2026)
|-- environment.yml                      # Conda environment definition (added 2026)
|-- .gitignore                           # Local data/model exclusions (added 2026)
`-- docs/
    |-- METHODOLOGY.md                    # Traceable implementation notes (added 2026)
    `-- RESULTS_AND_LIMITATIONS.md        # Result interpretation and rerun criteria (added 2026)
```

## Reproducibility

The repository is **not fully executable as distributed**: it does not include the required NumPy arrays (`data_train.npy`, `labels_train.npy`, `data_train_vars.npy`, and `labels_train_corr_vars.npy`) or trained model (`CNN_overlap.h5`). If authorized copies of those original files are available, place them in the repository root without committing them.

Create an environment using either:

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
jupyter notebook
```

or:

```bash
conda env create -f environment.yml
conda activate traffic-sign-cnn
jupyter notebook
```

Open `Final Project - Training Data.ipynb` to inspect the original data, `Model_Train.ipynb` to train, and `Model_Test.ipynb` to reproduce the legacy evaluation only. TensorFlow/Keras 2.7 is documented in the original report, but the exact versions of the other 2022 packages and the original Python version were not recorded here. See [Methodology](docs/METHODOLOGY.md) and [Results and Limitations](docs/RESULTS_AND_LIMITATIONS.md) before attempting a rigorous rerun.

## Skills Demonstrated

- Python software development
- Machine-learning model development
- CNN architecture design
- Data preprocessing and augmentation
- Train/validation/test methodology
- Model verification and validation
- Quantitative performance analysis
- Technical documentation
- Git-based project organization

## Academic Context

- [EEL 5840 / EEE 4773 Summer C 2022 syllabus](https://github.com/EEL5840-EEE4773-Summer2022/Syllabus)
- [Official Summer 2022 course organization repositories](https://github.com/orgs/EEL5840-EEE4773-Summer2022/repositories)

These links provide course context only; the organization and its repositories are not presented as Ryan Heras's work.


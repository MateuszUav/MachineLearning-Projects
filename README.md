# MachineLearning-Projects

A collection of small machine learning exercises and projects written in Python (NumPy, scikit-learn, TensorFlow/Keras, PyTorch, Matplotlib). Each file is a standalone script covering a different ML concept, ranging from classic regression to CNNs and LSTMs.

## Contents

### `regresja.py` — Polynomial & Reciprocal Regression
Loads 2D data from `data/dane7.txt`, performs an 80/20 shuffle split, and fits two models via the normal equation / pseudoinverse: a cubic polynomial regression and a reciprocal model. Uses NumPy linear algebra directly (no scikit-learn) and plots the fitted curves.

### `softmax.py` — Softmax Classifier with Decision Regions
Implements a multi-class classifier and a `plot_decision_regions` helper that visualizes decision boundaries, markers, and color maps for each class. Focused on plotting and understanding decision surfaces.

### `klasyfikatory` — Classifier Comparison
Benchmarks several scikit-learn classifiers on the `make_moons` synthetic dataset (10k samples, noisy). Compares Decision Trees (entropy vs. gini, varying depths), Random Forests (varying number of trees), and combines models with a Voting Classifier alongside Logistic Regression and SVM. Reports train/test accuracy and cross-validation scores.

### `propagacjaWsteczna` — Backpropagation from Scratch
A neural network regression implemented in pure NumPy (a Python port of an Octave/MATLAB `siec.m` script). Single hidden layer with 100 neurons, using `arctan` as the activation function, manual forward/backward passes, and gradient descent. Trains on `data/dane8.txt`.

### `CNN-cifar` — CNN on CIFAR-10 (Binary)
Convolutional neural network using TensorFlow/Keras. Loads CIFAR-10, merges train+test into one pool, and reframes it as a binary problem (animals vs. vehicles) with a two-output softmax head. Uses a 30/70 train/test split and a reusable `run()` helper to compile, train, and evaluate models.

### `LSTM-AR` — LSTM Autoregressive Time-Series
The largest project (~400 lines). A PyTorch LSTM for autoregressive time-series forecasting. Loads a 2110×4 dataset, scales features with `MinMaxScaler`, builds sequences, trains on CPU, and evaluates with MSE, MAE, and R² metrics. Includes reproducible seeding and Matplotlib (Agg backend) plotting.

### `rockPaperSciz` — Rock-Paper-Scissors Markov Strategy
A probabilistic Rock/Paper/Scissors simulation. Models an opponent with a fixed move distribution, uses a transition/counts matrix to predict the opponent's next move, plays the counter-move, and tracks cumulative payoff over 10,000 rounds.

## Tech Stack
- **Python** (100%)
- NumPy, Matplotlib
- scikit-learn
- TensorFlow / Keras
- PyTorch, pandas

## Notes
- Several scripts expect local data files under a `data/` directory (`dane7.txt`, `dane8.txt`, etc.) that are not committed to the repo.
- Comments are a mix of English and Polish.

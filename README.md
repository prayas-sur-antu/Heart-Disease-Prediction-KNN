# Heart Disease Prediction (KNN, scikit-learn)

A clinical classification project that predicts whether a patient is likely to have heart disease, using a K-Nearest Neighbors model on a dataset of 918 patients. This started as a Dataquest guided project; I worked through it to practise the full supervised-learning workflow on real clinical data — EDA, cleaning, feature selection, scaling, and tuning.

## What it does

- Explores the dataset (11 clinical features such as age, sex, chest pain type, resting blood pressure, cholesterol, max heart rate, ST slope) with descriptive statistics and grouped count plots.
- Cleans clinically impossible values: removes the single row with a resting blood pressure of 0, and imputes the 172 zero-cholesterol values using group medians (separately for patients with and without heart disease).
- Selects features by correlation with the target, keeping the strongest predictors (ST slope, exercise-induced angina, Oldpeak, sex).
- One-hot encodes categorical variables and scales features with `MinMaxScaler`.
- Trains a KNN classifier, then tunes `k` and the distance metric with `GridSearchCV` (cross-validated).

## Results

| Stage | Accuracy |
|-------|----------|
| Best single-feature model (ST slope) | ~82% |
| Multi-feature scaled KNN | ~83% |
| After GridSearchCV tuning (k = 19, Minkowski) — test set | ~87% |

The tuned model reaches ~87% accuracy on the held-out test set.

## An honest caveat

The dataset is heavily imbalanced by sex — 724 male patients vs 193 female. Because sex is one of the model's features, that imbalance can bias performance and limits how well the model would generalise to a more balanced population. The high test accuracy should be read with that in mind. Flagging this kind of limitation matters more in clinical prediction than squeezing out an extra point of accuracy.

## Tech stack

Python, pandas, scikit-learn (KNeighborsClassifier, GridSearchCV, MinMaxScaler), seaborn, matplotlib.

## How to run

1. Install dependencies:
   ```
   pip install pandas scikit-learn seaborn matplotlib
   ```
2. Place `heart_disease_prediction.csv` in the working directory (Kaggle Heart Failure Prediction dataset).
3. Open the notebook and run the cells top to bottom.

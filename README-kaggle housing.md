# Housing Price Prediction: Kaggle Learn Exercises

Two notebooks from Kaggle's free Learn courses, both using the Ames, Iowa housing data from the [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/c/home-data-for-ml-course). They cover the basics of fitting a regression model and handling missing values with scikit-learn.

These are guided course exercises. The setup and answer-checking cells come from Kaggle; the fill-in code and the comparisons are mine.

## Notebooks

| Notebook | Course | Topic |
| --- | --- | --- |
| `01_first_model_decision_tree.ipynb` | [Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) | Select a target and features, fit a decision tree, make predictions |
| `02_missing_values.ipynb` | [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning) | Compare ways of handling missing values with a random forest |

## Data

The data is not included in this repository. Download `train.csv` and `test.csv` from the [competition page](https://www.kaggle.com/c/home-data-for-ml-course). The target is `SalePrice`.

## Notebook 1: First model (decision tree)

- **Target:** `SalePrice`
- **Features (7):** `LotArea`, `YearBuilt`, `1stFlrSF`, `2ndFlrSF`, `FullBath`, `BedroomAbvGr`, `TotRmsAbvGrd`
- **Model:** `DecisionTreeRegressor(random_state=1)`
- **Result:** predictions were made on the same data the model was trained on, so this notebook does not measure accuracy. An unpruned tree reproduces its training prices almost exactly. Model validation is the next step in the course.

## Notebook 2: Missing values

**Setup:** numerical predictors only (36 columns), an 80/20 train/validation split (1,168 training rows, `random_state=0`). Three columns have missing values, 276 missing entries in total:

| Column | Missing values |
| --- | --- |
| `LotFrontage` | 212 |
| `GarageYrBlt` | 58 |
| `MasVnrArea` | 6 |

**Model and metric:** `RandomForestRegressor(n_estimators=100, random_state=0)`, scored by mean absolute error (MAE) on the validation set.

| Approach | Validation MAE |
| --- | --- |
| Drop the 3 columns with missing values | $17,838 |
| Mean imputation (`SimpleImputer`) | $18,063 |

**Takeaway:** dropping the columns scored slightly better, but the gap is only about $225 and is probably partly noise. Mean imputation may also be a poor fit here. For example, a missing `GarageYrBlt` likely means the house has no garage, so a fill value like 0 or an indicator column might work better. I did not test those alternatives.

## Limitations

- Notebook 1 has no validation set, so it gives no estimate of accuracy.
- In Notebook 2, the test-set preprocessing cell (Step 4B) references `final_imputer`, which is never defined, so it raises a `NameError` and `submission.csv` is not produced. The validation comparison above is not affected.
- Only numerical columns are used, and there is no feature engineering or hyperparameter tuning.

## How to run

The notebooks import `learntools`, Kaggle's answer-checking package, so they run as-is only on Kaggle. To run them elsewhere, remove the `learntools` and `step_N.check()` cells and install the requirements:

```
pip install pandas scikit-learn
```

Then update the file paths to point to your local copy of `train.csv` and `test.csv`.

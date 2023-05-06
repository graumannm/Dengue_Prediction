# import
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from preproc import preprocess_data

# global variables (extra variable file?)
# seed = 42
test_ratio = 0.25

# load data
# load data
X_path = "data-processed/dengue_features_train.csv"
y_path = "data-processed/dengue_labels_train.csv"

[X_sj, y_sj, X_iq, y_iq] = preprocess_data(X_path, True)


# X_data = pd.read_csv("./data-processed/dengue_features_train.csv")
# y_data = pd.read_csv("./data-processed/dengue_labels_train.csv")

# split cities
# mask_sj = X_data.loc[:, "city"] == "sj"
# mask_iq = X_data.loc[:, "city"] == "iq"

# X_sj = X_data.loc[mask_sj, :]
# y_sj = y_data.loc[mask_sj, "total_cases"]
# X_iq = X_data.loc[mask_iq, :]
# y_iq = y_data.loc[mask_iq, "total_cases"]

X_sj_train, X_sj_test, y_sj_train, y_sj_test = train_test_split(
    X_sj, y_sj, test_size=test_ratio, shuffle=False
)
X_iq_train, X_iq_test, y_iq_train, y_iq_test = train_test_split(
    X_iq, y_iq, test_size=test_ratio, shuffle=False
)


########
# start Pipeline here
# preprocessing
# easy wrangle NANs
# df.fillna(method='ffill', inplace=True) # bfill: use next valid observation to fill gap
class FillImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method="ffill"):
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.fillna(method=self.method)
        return X


# transform datetime
# transform weeks
class SinCosTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, attr="weekofyear"):
        self.attr = attr

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.loc[:, "weekofyear_cos"] = np.cos(2 * np.pi * X.loc[:, self.attr] / 52)
        X.loc[:, "weekofyear_sin"] = np.sin(2 * np.pi * X.loc[:, self.attr] / 52)
        X = X.drop(columns=[self.attr])
        return X


# drop useless columns
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, attrs=["week_start_date", "city"]):
        self.attrs = attrs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(columns=self.attrs)
        return X


model_pipeline = Pipeline(
    [
        #         ("imputer", FillImputer()),
        #         ("transformer", SinCosTransformer()),
        #         ("dropper", DropColumns()),
        ("rf", RandomForestRegressor(n_estimators=10, random_state=0)),
    ]
)


# # fit and predict San Juan
# model_pipeline.fit(X_sj_train, y_sj_train)
# y_sj_pred = model_pipeline.predict(X_sj_test)
# my_mse_sj = metrics.mean_squared_error(y_sj_test, y_sj_pred)

# print(my_mse_sj)

# # fit and predict Iquitos
# model_pipeline.fit(X_iq_train, y_iq_train)
# y_iq_pred = model_pipeline.predict(X_iq_test)
# my_mse_iq = metrics.mean_squared_error(y_iq_test, y_iq_pred)

# print(my_mse_iq)
# model
# choose models

# do grit/random search

parameters_grid = [
    {
        "rf__n_estimators": [3, 10, 30, 100],
        "rf__criterion": ["squared_error", "poisson"],
        "rf__max_depth": [3, 5, 10, 15],
    }
]

# grid search
grid_search_pip = GridSearchCV(model_pipeline, parameters_grid, cv=3)

grid_search_pip.fit(X_sj_train, y_sj_train)
final_train_model_sj = grid_search_pip.best_estimator_
best_train_param_sj = grid_search_pip.best_params_
y_sj_pred = final_train_model_sj.predict(X_sj_test)
my_mse_sj = metrics.mean_squared_error(y_sj_test, y_sj_pred)

print("best estimator on sj: ", final_train_model_sj)
print("best param on sj: ", best_train_param_sj)
print("mse on sj: ", my_mse_sj)

grid_search_pip.fit(X_iq_train, y_iq_train)
final_train_model_iq = grid_search_pip.best_estimator_
best_train_param_iq = grid_search_pip.best_params_
y_iq_pred = final_train_model_iq.predict(X_iq_test)
my_mse_iq = metrics.mean_squared_error(y_iq_test, y_iq_pred)

print("best estimator on iq: ", final_train_model_iq)
print("best param on iq: ", best_train_param_iq)
print("mse on iq: ", my_mse_iq)

# evaluate models on whole train set
model_pipeline.set_params(**best_train_param_sj)
final_model_sj = model_pipeline.fit(X_sj, y_sj)
y_sj_pred = final_model_sj.predict(X_sj_test)
my_mse_sj = metrics.mean_squared_error(y_sj_test, y_sj_pred)
print("param on sj:", final_model_sj.get_params())
print("whole train set mse_sj: ", my_mse_sj)

model_pipeline.set_params(**best_train_param_iq)
final_model_iq = model_pipeline.fit(X_iq, y_iq)
y_iq_pred = final_model_iq.predict(X_iq_test)
my_mse_iq = metrics.mean_squared_error(y_iq_test, y_iq_pred)
print("param on iq:", final_model_iq.get_params())
print("whole train set mse_iq: ", my_mse_iq)
# track model parameters

# submission
sj_test, sj_y, iq_test, iq_y = preprocess_data(
    "data-processed/dengue_features_test.csv", False
)
sj_predictions = final_model_sj.predict(sj_test).astype(int)
iq_predictions = final_model_iq.predict(iq_test).astype(int)

submission = pd.read_csv("data-processed/submission_format.csv", index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv("data-processed/our_model001.csv")

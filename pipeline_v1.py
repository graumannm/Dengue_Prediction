# import
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

from preproc import preprocess_data

# global variables
# seed = 42
test_ratio = 0.25
data_path = "data-processed/"

interpolation = True
norm = True
mylag = 2
interaction = True
ylabels = True

preprocess_string = "All_lagged_and_mm"

# load data
X_path = "data-processed/dengue_features_train.csv"
y_path = "data-processed/dengue_labels_train.csv"

# preprocess data
[X_sj, y_sj, X_iq, y_iq] = preprocess_data(
    X_path=X_path,
    interpolation=interpolation,
    norm=norm,
    mylag=mylag,
    interaction=interaction,
    ylabels=True,
)

# split train test
X_sj_train, X_sj_test, y_sj_train, y_sj_test = train_test_split(
    X_sj, y_sj, test_size=test_ratio, shuffle=False
)
X_iq_train, X_iq_test, y_iq_train, y_iq_test = train_test_split(
    X_iq, y_iq, test_size=test_ratio, shuffle=False
)


# fill nans
class FillImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method="ffill"):
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.fillna(method=self.method)
        return X


# transform weeks, assuming that weekofyear is in the index
class SinCosWeekTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.loc[:, "weekofyear_cos"] = np.cos(
            2 * np.pi * X.index.get_level_values("weekofyear") / 52
        )
        X.loc[:, "weekofyear_sin"] = np.sin(
            2 * np.pi * X.index.get_level_values("weekofyear") / 52
        )
        # X = X.drop(columns=[self.attr])
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


# preprocessing
class Preproc(BaseEstimator, TransformerMixin):
    def __init__(self, attrs):
        self.attrs = attrs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(columns=self.attrs)
        return X


# model pipeline for grid search
model_pipeline = Pipeline(
    [
        #         ("imputer", FillImputer()),
        #         ("transformer", SinCosWeekTransformer()),
        #         ("dropper", DropColumns()),
        # ("rf", RandomForestRegressor(n_estimators=10, random_state=0)),
        # ("svr", SVR()),
        ("gbr", GradientBoostingRegressor())
    ]
)

#### grit search
# grid search parameters
parameters_grid_rf = [
    {
        "rf__n_estimators": [3, 10, 30, 100],
        "rf__criterion": ["squared_error", "poisson"],
        "rf__max_depth": [3, 5, 10, 15],
    }
]

parameters_grid_svr = [
    {
        "svr__gamma": ["scale", "auto"],
        "svr__kernel": ["poly", "linear", "rbf"],
    }
]

parameters_grid_gbr = [
    {
        "gbr__criterion": ["squared_error", "friedman_mse"],
        "gbr__max_depth": [3, 5, 10, 15],
        "gbr__n_estimators": [5, 10, 100],
        "gbr__max_features": [0.9, 1.0],
    }
]

# grid search
grid_search_pip = GridSearchCV(model_pipeline, parameters_grid_gbr, cv=3)

# evaluate models
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

# safe predictions
y_sj_test = y_sj_test.to_frame()
y_iq_test = y_iq_test.to_frame()
y_sj_test.loc[:, "pred_cases"] = y_sj_pred
y_iq_test.loc[:, "pred_cases"] = y_iq_pred
y_sj_test.to_csv("data-processed/" + "sj_" + preprocess_string + "_gb" + ".csv")
y_iq_test.to_csv("data-processed/" + "iq_" + preprocess_string + "_gb" + ".csv")

# submission
sj_test, sj_y, iq_test, iq_y = preprocess_data(
    "data-processed/dengue_features_test.csv",
    interpolation=interpolation,
    norm=norm,
    mylag=mylag,
    interaction=interaction,
    ylabels=False,
)
sj_predictions = final_model_sj.predict(sj_test).astype(int)
iq_predictions = final_model_iq.predict(iq_test).astype(int)

submission = pd.read_csv("data-processed/submission_format.csv", index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv("data-processed/" + preprocess_string + "_gb" + ".csv")


# # visualize results
# y_sj_p = y_sj.copy()
# y_sj_p.iloc[-len(y_sj_pred) :] = y_sj_pred
# total = pd.concat(
#     [
#         y_sj,
#         y_sj_p,
#     ],
#     axis=1,
# )
# plt.close()
# plt.clf()
# total.plot()
# save_fig("model_time_sj_real_vs_pred")

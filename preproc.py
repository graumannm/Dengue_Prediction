import pandas as pd
import numpy as np

# run all preprocessing steps on both cities, then in last step seperate them

# load data
# X_path = "data-processed/dengue_features_train.csv"
# y_path = "data-processed/dengue_labels_train.csv"


def preprocess_data(X_path, labels):
    # Input:
    # labels: logical true or false whether we have y (because submission data does not)

    # load data and set index to city, year, weekofyear and remove week_start_date
    df = pd.read_csv(X_path, index_col=[0, 1, 2])
    df = df.drop("week_start_date", axis=1)  # [features]

    ### fill missing values with interpolation ###

    df.fillna(
        method="bfill", inplace=True
    )  # bfill: use next valid observation to fill gap

    ### add lag of 3 weeks for 4 variables below ###

    # and remove original ones
    # max temp, precipitation, humidity and avgerage temp
    lag = 3
    var2change = [
        "station_max_temp_c",
        "station_precip_mm",
        "reanalysis_relative_humidity_percent",
        "station_avg_temp_c",
    ]

    new_names = []
    for i, j in enumerate(var2change):
        new_names.append(j + "_lag")

        df[new_names[i]] = df[var2change[i]].shift(lag)  # Lagged by 1 time step

        # remove original
        df = df.drop(var2change[i], axis=1)

    # remove missing values again because of lag
    df.fillna(method="bfill", inplace=True)

    ### create interaction features

    # humidity above 42 % temperature above 24 degrees
    df["Humid_X_Temp26"] = np.where(
        (df["reanalysis_relative_humidity_percent_lag"] >= 42)
        & (df["station_avg_temp_c_lag"] >= 24),
        1,
        0,
    )

    # below one did not work
    # temperature after rain. Use rain vs. lagged temp var
    # df['Temp_X_rain'] = np.where((df['station_precip_mm_lag'] >= 600) & (df['station_avg_temp_c_lag'] >= 24), 1, 0)

    # if its not submission, load y data
    if labels == True:
        # add predictor to DF to seperate the cities
        y_path = "data-processed/dengue_labels_train.csv"
        y = pd.read_csv(y_path, index_col=[0, 1, 2])
        df = df.join(y)

    # separate san juan and iquitos
    df_sj = df.loc["sj"]
    df_iq = df.loc["iq"]

    if labels == True:
        # remove y from X again
        # San Juan
        X_sj = df_sj.drop("total_cases", axis=1)
        y_sj = df_sj["total_cases"]

        # Iquitos
        X_iq = df_iq.drop("total_cases", axis=1)
        y_iq = df_iq["total_cases"]

    else:
        X_sj = df_sj
        X_iq = df_iq
        y_sj = []
        y_iq = []

    return X_sj, y_sj, X_iq, y_iq


# remove df later again, just for debugging
# [X_sj, y_sj, X_iq, y_iq] = preprocess_data(X_path, True)

import pandas as pd


def load_data(path, filename):
    file_path = path + filename
    df0 = pd.read_csv(file_path)
    df = df0.copy()
    return df


def run_data_pipeline(df):
    """
    Data pipeline is outlined by the following process:
        1) Sort the data by last trip date in ascending order
        2) Fill in missing ratings data with the average rating of all riders and drivers respectively
        3) Drop the remaining records that contain any sort of missing values (other than ratings)
        4) Convert last trip date from pandas object to datetime format
        5) Create a "churn" column of Boolean value to determine based on our company standards of active users
        6) Create column in Boolean value if app user's device is an iPhone
        7) Convert luxury car indicator to Boolean
        8) Create dummy variables for city
    """

    df = df.sort_values(by=['last_trip_date'])
    _fill_na_mean(df, 'avg_rating_of_driver')
    _fill_na_mean(df, 'avg_rating_by_driver')
    df.dropna(axis=0, inplace=True)
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df = _create_indicator(df, 'churn',  df['last_trip_date'] < '2014-06-01')
    df = _create_indicator(df, 'phone', df['phone'] == 'iPhone')
    df['luxury_car_user'] = df['luxury_car_user'] * 1
    df = pd.get_dummies(df, columns=['city'], drop_first=False)

    return df


def create_variables(df):

    X = df.drop(columns = ['signup_date', 'last_trip_date', 'churn'])
    y = df['churn']

    return X,y


def _fill_na_mean(df, column):
    df[column].fillna(df[column].mean(), inplace=True)


def _create_indicator(df, column, condition):
    df[column] = condition
    df[column] = df[column] * 1
    return df


from pathlib import Path
import pandas as pd

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

root_dir = str(get_project_root())
training_data = '/data/churn_train.csv'
test_data = '/data/churn_test.csv'

def load_train_data(filename):
    file_path = root_dir + filename
    df0 = pd.read_csv(file_path)
    df = df0.copy()
    return df

def _fill_na_mean(df, column):
    df[column].fillna(df[column].mean(), inplace=True)

def _create_indicator(df, column, condition):
    df[column] = condition
    df[column] = df[column] * 1
    return df


def run_data_pipeline(df):
    """
    Data pipeline is outlined by the following process:
        1) Sort the data by last trip date in ascending order
        2) Fill in missing ratings data points with the average rating of all records in data
        3) Drop the remaining records that contain any sort of missing values (other than ratings)
        4) Convert last trip date from ____ to datetime format
        5) Create a "churn" column of Boolean value to determine based on our standards
                if the user has not returned since a month of the latest date
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

if __name__ == "__main__":
    df = load_train_data(training_data)
    df_clean = run_data_pipeline(df)

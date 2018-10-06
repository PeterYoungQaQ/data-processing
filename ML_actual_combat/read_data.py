import pandas as pd
import os
import matplotlib.pyplot as plt

HOUSING_PATH = "datasets/housing"


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()


def test_data():
    print(housing.info())
    print("\n")

    print(housing["ocean_proximity"].value_counts())
    print("\n")
    m = housing.describe()
    print(m)

    housing.hist(bins=50, figsize=(20, 15))
    plt.show()


# test_data()

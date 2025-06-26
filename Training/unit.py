import pytest
import pandas as pd
from training import read_data, split_data, train_model
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sample_data = None
# @pytest.fixture
def test_read_data(data_path):
    global sample_data
    df = read_data(data_path)
    sample_data = df
    assert df.shape[1]==10

def test_train_model():
    X = sample_data.drop(columns=["Potability"])
    y = sample_data["Potability"]

    pipe = train_model(X, X, y, y)

    assert isinstance(pipe, Pipeline)
    assert isinstance(pipe.named_steps["Scale"], StandardScaler)
    assert isinstance(pipe.named_steps["Train"], RandomForestClassifier)


# if __name__ == "__main__":
#     data_path = r"D:\Azure MLOps learning\Custom project\water quality\Told\Water Quality Prediction\Water-quality-CICD\CI CD\Data\unittest\unittest_data.csv"
#     test_read_data(data_path)
#     test_train_model()
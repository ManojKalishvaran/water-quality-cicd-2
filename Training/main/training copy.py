import os
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier
import argparse
import mlflow
import pickle


def read_data(path):
    df = pd.read_csv(path)
    return df 


def split_data(data):
    X = data.drop(columns=['Potability'], axis=1)
    y = data['Potability']
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, random_state=87, \
                         shuffle=True, stratify=y)
    return X_train, X_test, y_train, y_test


def train_model(x_train, x_test, y_train, y_test):
    pipe = Pipeline(steps=[
        ("Scale", StandardScaler()), 
        ("Train", RandomForestClassifier(n_estimators=240, \
                                          n_jobs=-1, random_state=45))
    ])
    pipe.fit(x_train, y_train)
    print("Training Complete")
    
    # model_folder = "model"

    # os.makedirs(model_folder, exist_ok=True)
    # with open(os.path.join(model_folder, "model.pkl"), "wb") as file: 
    #     pickle.dump(pipe, file)
    # print(f'Model is saved...')
    return pipe

def main(args):
    mlflow.autolog()

    df = read_data(args.data_path)

    X_train, X_test, y_train, y_test = split_data(df)

    train_model(X_train, X_test, y_train, y_test)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", 
                        dest="data_path", type=str)
    
    args = parser.parse_args()

    return args 


if __name__ == "__main__":
    args = parse_args()
    main(args)
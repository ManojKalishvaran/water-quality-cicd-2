import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier
import io
import pickle
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

def read_reference_data():
    credential = DefaultAzureCredential()

    storage_account_name = "manojblob1"
    container_name = "training-waterquality"

    account_url = f"https://{storage_account_name}.blob.core.windows.net"

    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)

    container_client = blob_service_client.get_container_client(container_name)
    blob_name = "training_water.csv"
    blob_client = container_client.get_blob_client(blob_name)

    blob_data = blob_client.download_blob().readall()
    df = pd.read_csv(io.StringIO(blob_data.decode("utf-8")), sep=",")
    return df


def read_target_data():
    credential = DefaultAzureCredential()

    storage_account_name = "manojblob1"
    container_name = "data-log-test"

    account_url = f"https://{storage_account_name}.blob.core.windows.net"

    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)

    container_client = blob_service_client.get_container_client(container_name)
    blob_name = "logs.csv"
    blob_client = container_client.get_blob_client(blob_name)

    blob_data = blob_client.download_blob().readall()
    df = pd.read_csv(io.StringIO(blob_data.decode("utf-8")), sep=",")
    return df

def read_data():
    df_reference = read_reference_data()
    df_target = read_target_data()

    df_target.rename(columns={df_target.columns[-1]:df_reference.columns[-1]}, inplace=True)
    df_target = df_target[list(df_target.columns[1:])]
    print(f"{df_reference.columns = }, {df_target.columns = }")

    total_df = pd.concat([df_reference, df_target], axis=0)
    total_df.drop_duplicates(inplace=True)
    print(f"There are {len(total_df)} samples...")

    return total_df

def train_model(x_train, y_train):
    pipe = Pipeline(steps=[
        ("Scale", StandardScaler()), 
        ("Train", RandomForestClassifier(n_estimators=240, \
                                          n_jobs=-1, random_state=45))
    ])
    pipe.fit(x_train, y_train)
    print("Training Complete")

    credential = DefaultAzureCredential()

    storage_account = "manojblob1"
    container_name = "retrained-models"

    account_url = f"https://{storage_account}.blob.core.windows.net"
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)

    container_client = blob_service_client.get_container_client(container_name)
    try: 
        container_client.create_container()
    except Exception as e: 
        print(f"Container already exists")
    blob_name = "model-retrained.pkl"
    blob_client = container_client.get_blob_client(blob_name)

    model_bytes = pickle.dumps(pipe)
    blob_client.upload_blob(model_bytes, overwrite=True)

    print(f'fininshed uploading')
    print("Model retraining uploading completed !!!")


def main():
    data = read_data()
    input_data = data.drop(columns=["Potability"])
    target = data[["Potability"]]
    train_model(input_data, target)

if __name__=="__main__":
    main()
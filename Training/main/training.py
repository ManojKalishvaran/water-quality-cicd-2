import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier
import io
import pickle
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient


def read_data():
    # # Replace with your Key Vault name and secret name
    # keyvault_name = "manoj-key-water"
    # secret_name = "train-data-sas"  # e.g., "storage-container-sas-token"

    # Construct Key Vault URI
    # KVUri = f"https://{keyvault_name}.vault.azure.net"

    # Authenticate to Key Vault using DefaultAzureCredential
    credential = DefaultAzureCredential()
    # secret_client = SecretClient(vault_url=KVUri, credential=credential)

    # # Retrieve the SAS token from Key Vault
    # sas_token = secret_client.get_secret(secret_name).value

    # Construct the Blob service URL with SAS token
    storage_account_name = "manojblob1"
    container_name = "training-waterquality"

    # The account_url should be without SAS token
    account_url = f"https://{storage_account_name}.blob.core.windows.net"

    # Initialize BlobServiceClient with SAS token as credential
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)

    # Get container and blob clients
    container_client = blob_service_client.get_container_client(container_name)
    blob_name = "training_water.csv"
    blob_client = container_client.get_blob_client(blob_name)

    # Download blob content
    blob_data = blob_client.download_blob().readall()
    df = pd.read_csv(io.StringIO(blob_data.decode("utf-8")), sep=",")
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

    # keyvault_name = "manoj-key-water"
    # secret_name = "model-blob"

    # KVUri = f"https://{keyvault_name}.vault.azure.net"

    credential = DefaultAzureCredential()
    # secret_client = SecretClient(vault_url=KVUri, credential=credential)

    # sas_token = secret_client.get_secret(secret_name).value
    storage_account = "manojblob1"
    container_name = "trained-models"

    account_url = f"https://{storage_account}.blob.core.windows.net"
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)

    container_client = blob_service_client.get_container_client(container_name)
    try: 
        container_client.create_container()
    except Exception as e: 
        print(f"Container already exists")
    blob_name = "water_quality_classififer.pkl"
    blob_client = container_client.get_blob_client(blob_name)

    model_bytes = pickle.dumps(pipe)
    blob_client.upload_blob(model_bytes, overwrite=True)

    print(f'fininshed uploading')
    print("Model training uploading completed !!!")

def main():
    print(f'Provisioning Data...')
    df = read_data()
    print(f'Splitting Data...')
    X_train, X_test, y_train, y_test = split_data(df)
    print(f'Training Model...')
    train_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    # args = parse_args()
    main()
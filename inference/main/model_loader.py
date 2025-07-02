from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from datetime import datetime
import pickle
import pandas as pd
import io


def load():
    keyvault_name = "manoj-key-water"
    secret_name = "model-blob"

    KVUri = f"https://{keyvault_name}.vault.azure.net"
    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=KVUri, credential=credential)

    sas_token = secret_client.get_secret(secret_name).value
    storage_account = "manojblob1"
    container_name = "trained-models"

    account_url = f"https://{storage_account}.blob.core.windows.net"
    blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token)
    container_client = blob_service_client.get_container_client(container_name)

    # Step 1: List all blobs in the container with timestamps
    blobs = container_client.list_blobs()

    # Step 2: Find the blob with the latest last modified time
    latest_blob = max(blobs, key=lambda b: b.last_modified)

    model_info = f"Loading latest model: {latest_blob.name}, Last Modified: {latest_blob.last_modified}"
    print(f"Loading latest model: {latest_blob.name}, Last Modified: {latest_blob.last_modified}")

    # Step 3: Download the latest model blob
    blob_client = container_client.get_blob_client(latest_blob.name)
    blob_data = blob_client.download_blob().readall()

    # Step 4: Deserialize model
    model = pickle.loads(blob_data)
    return model, model_info

def load_blob_client():
    feature = ["timestamp", "ph","Hardness","Solids","Chloramines","Sulfate","Conductivity","Organic_carbon","Trihalomethanes","Turbidity", "prediction"]
    keyvault_name = "manoj-key-water"
    secret_name = "logs"  

    KVUri = f"https://{keyvault_name}.vault.azure.net"

    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=KVUri, credential=credential)

    sas_token = secret_client.get_secret(secret_name).value

    storage_account_name = "manojblob1"
    container_name = "data-log-test"

    account_url = f"https://{storage_account_name}.blob.core.windows.net"

    blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token)

    container_client = blob_service_client.get_container_client(container_name)
    blob_name = "logs.csv"
    blob_client = container_client.get_blob_client(blob_name)

    if not blob_client.exists():

        df = pd.DataFrame(columns= feature)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        blob_client.upload_blob(csv_buffer.getvalue(), overwrite=True)
    
    return blob_client, feature

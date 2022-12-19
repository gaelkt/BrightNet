import os
from azure.storage.blob import BlobServiceClient

from .helper_general import write_logs
import re
import boto3
from trainer.__init__ import file_log


def download_from_S3(aws_access_key_id, aws_secret_access_key, bucket_name, s3_folder, local_dir):
    """
    Download the contents of a folder directory from S3 Bucket
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """

    try:
        write_logs(file_log, controler='Dataset', action='download_from_S3', message='Starting download from S3 Bucket')
        local_path_dataset = local_dir + '/' + s3_folder
    
        if os.path.exists(local_path_dataset):
            write_logs(file_log, controler='Dataset', action='download_from_S3', message='Error. A local dataset already exists')
            raise Exception("Local dataset already exists")
        else:
            os.mkdir(local_path_dataset)
        
        s3 = boto3.resource('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

        bucket = s3.Bucket(bucket_name)

    
        folder_exist = False
        for obj in bucket.objects.filter(Prefix=s3_folder + '/'):
            
            if '.' not in obj.key:
                continue            

            target = obj.key if local_path_dataset is None \
            else os.path.join(local_path_dataset, os.path.relpath(obj.key, s3_folder))
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
                if obj.key[-1] == '/':
                    continue

            bucket.download_file(obj.key, target)

            folder_exist = True

        
        if not folder_exist:
            raise Exception(s3_folder + " folder does not exist in the container " + bucket_name)     
        
        write_logs(file_log, controler='Dataset', action='download_from_S3', message='Download from S3 Bucket finished')

    except Exception as e:
        write_logs(file_log, controler='Dataset', action='download_from_S3', message=str(e))
        raise Exception(e)
        

    
  
def download_from_blob(profile_id, project_name, my_connection_string, my_blob_container, blob_folder, local_blob_path):
    
    """
    Download the contents of a folder directory from azure storage
    Args:
        my_blob_container: the name of the container
        blob_folder: the folder name in the container
        local_blob_path: a relative or absolute directory path in the local file system where to download the dataset
    """    
    
    class AzureBlobFileDownloader:
        def __init__(self):
            # print("Intializing AzureBlobFileDownloader")
 
            # Initialize the connection to Azure storage account
            self.blob_service_client =  BlobServiceClient.from_connection_string(my_connection_string)
            self.my_container = self.blob_service_client.get_container_client(my_blob_container)
 
 
        def save_blob(self,file_name,file_content):
            # Get full path to the file
            download_file_path = os.path.join(local_blob_path, file_name)
 
            # for nested blobs, create local path as well!
            os.makedirs(os.path.dirname(download_file_path), exist_ok=True)
 
            with open(download_file_path, "wb") as file:
                file.write(file_content)
 
        def download_all_blobs_in_container(self):
            # print('Downloading all blobs in the container ...')
            my_blobs = self.my_container.list_blobs()
            folder_exist = False
            for blob in my_blobs:
                if re.match(blob_folder + '/', blob.name):
                    # print(blob.name)
                    folder_exist = True
                    bytes = self.my_container.get_blob_client(blob).download_blob().readall()
                    self.save_blob(blob.name, bytes)
            
            if not folder_exist:
                raise Exception(blob_folder + " folder does not exist in the container " + my_blob_container)
                

    # Initialize class and upload files
    try:
        write_logs(file_log, controler='Dataset',  action='download_from_blob', profile_id = profile_id, project_name=project_name, message= 'Start download from blob')

        azure_blob_file_downloader = AzureBlobFileDownloader()

        azure_blob_file_downloader.download_all_blobs_in_container()    
        write_logs(file_log, controler='Dataset', profile_id = profile_id, project_name=project_name, action='download_from_blob', message='Dataset downloaded from azure blob')
    except Exception as e:
        # print('Got exception for azure donwload')
        write_logs(file_log, controler='Dataset', profile_id = profile_id, project_name=project_name, action='download_from_blob', message=str(e))
        raise Exception(e)
    return 0

                  

def upload_folder_to_azure_blob(local_path, path_remove, conn_str,container_name):    

    
    '''
    Upload a local folder on the vm to the remote azure blob
    Inputs:
        local_path: local path of the project to upload
                    For example: AllMyProjects + '/' + profile_id + '/' + project_name
        
        path_remove: path to remove from local_path when creating remote path on azure
                    For example: AllMyProjects + '/'
        
        conn_str,container_name: credentials
    
    '''
    
    try:
        write_logs(file_log, controler='Model', action='upload_folder_to_azure_blob', message='Uploading model to azure blob')
        service_client=BlobServiceClient.from_connection_string(conn_str)
        container_client = service_client.get_container_client(container_name)  
    
    
        if not os.path.exists(local_path):
            raise Exception("Local path for project does not exists")
    

        for r,d,f in os.walk(local_path):
            if f:
                for file in f:
                    file_path_on_azure = os.path.join(r,file).replace(path_remove,"")
                    file_path_on_local = os.path.join(r,file)

                    blob_client = container_client.get_blob_client(file_path_on_azure)

                    with open(file_path_on_local,'rb') as data:
                        blob_client.upload_blob(data)
    except Exception as e:
        write_logs(file_log, controler='Model', action='upload_folder_to_azure_blob', message=str(e))


def unzipping_zipped_dataset(is_dataset_zipped, directory_dataset):
    
    '''
        Check if the dataset is zipped and then unzipped the dataset
        
        Input: is_dataset_zipped: Bool 
    '''
    
    return 0
"""
Celery config file

https://docs.celeryproject.org/en/stable/django/first-steps-with-django.html

"""


from __future__ import absolute_import
import os
import shutil
# Need to set DJANGO_SETTINGS_MODULE before django.setup()
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'aitrainer.settings.dev')
import django
django.setup()

from celery import Celery
from django.conf import settings
from celery import shared_task

from trainer.__init__ import AllMyProjects, local_directory_dataset
from trainer.__init__ import central_conn_str_azure, central_container_azure
from trainer.__init__ import file_log
from trainer.__init__ import initialization_vector, key

from .helper_general import write_logs
from .helper_general import TriggerWebHook, update_database, check_dataset, check_exits_train_validation, delete_all_data_current_folder
from .helper_general import dataset_augmentation
from .helper_general import check_celery_server

from .helper_encryption import encrypt_classification_dataset, encrypt_detection_dataset


from .helper_download import upload_folder_to_azure_blob, download_from_S3, download_from_blob

from aitrainer.training import start_classification_training
from aitrainer.training_detection import start_detection_training

import zipfile

from django import db

import time


# this code copied from manage.py
# set the default Django settings module for the 'celery' app.
# you change change the name here
app = Celery("aitrainer")

# read config from Django settings, the CELERY namespace would make celery 
# config keys has `CELERY` prefix
app.config_from_object('django.conf:settings', namespace='CELERY')

# load tasks.py in django apps
app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)




import logging.config
logging.config.dictConfig({
    'version': 1,
    # Other configs ...
    'disable_existing_loggers': True
})


@shared_task
def download_and_training(profile_id, project_name, model_type, json_data, URL, export, num_epochs, batch_size, lr, channel, AllDatasets, step_size = 10,
                                                       gamma = 0.5, img_size = 224, Data_Crypted = False, krow = 1, kcol = 1, ITER_MAX = 1):
    
    try:
        
        
        write_logs(file_log, controler='Torch',  action='download_and_training', profile_id = profile_id, project_name=project_name, message= 'Starting download/training')
        
        dezipped = False # Variable to know if we have to deleted zipped files
        
        delete_all_data_current_folder(AllDatasets)

        delete_all_data_current_folder(AllMyProjects)
        
        if "storage_type" not in json_data:

            raise Exception("Please specify storage type")
        
        else:
            storagetype = json_data['storage_type']
            
        # Check several types of storage
        if storagetype == "azure":
            
            my_connection_string = json_data["azure_connection_string"]
            my_blob_container = json_data["container"]
            blob_folder = json_data["folder"]
            local_blob_path = AllDatasets + my_blob_container
            directory_dataset = local_blob_path + '/' + blob_folder
                         
            if os.path.exists(directory_dataset):
                raise Exception("Local dataset exists on Disk")
            else:
                # print('Creating local folder with container and blob folder')
                os.makedirs(directory_dataset)
                # print('Local folder for dataset created')
            
           
            # Updating status before download
            update_database(profile_id=profile_id, project_name=project_name, status='DOWNLOADING', message='Downloading azure dataset ')
            
            # Download
            download_from_blob(profile_id, project_name, my_connection_string, my_blob_container, blob_folder, local_blob_path)

            # Updating status after download
            update_database(profile_id=profile_id, project_name=project_name, status='DOWNLOADED', message='Dataset azure downloaded ')

 
            # A zipped dataset is provied
            # if json_data["zipped"] ==  True or ('zip' in os.listdir(directory_dataset)[0] and len(os.listdir(directory_dataset)) == 1):
            if json_data["zipped"] ==  True or ('zip' in os.listdir(directory_dataset)[0] and len(os.listdir(directory_dataset)) == 1):
                
                write_logs(file_log, controler='Dataset',  action='download_and_training', profile_id = profile_id, project_name=project_name, message= 'Dezipping dataset')

                # The dataset is not a single file
                if len(os.listdir(directory_dataset)) != 1:
                    raise Exception('Zipped dataset should have only one file .zip')
                
                dezipped = True # Used later to delete file
                parent_zipped = local_blob_path + '/' + blob_folder
                download_file_path = parent_zipped + '/' + os.listdir(parent_zipped)[0]
                
                # The actual dataset is not , while it is supposed to be
                if "zip" not in download_file_path:
                    raise Exception('Unable to find a zip file .' + str(os.listdir(parent_zipped)[0]) + ' is not a zipped file' )                
                
                
                # Dezipping the dataset
                with zipfile.ZipFile(download_file_path, 'r') as zip_ref:
                    zip_ref.extractall(parent_zipped)
                
                # Removing the zip file
                os.remove(download_file_path)
                
                # Updated dataset path
                directory_dataset = parent_zipped + '/' + os.listdir(parent_zipped)[0]
   
                
            
        elif storagetype == "aws":
            
            aws_access_key_id = json_data["aws_access_key_id"]
            aws_secret_access_key = json_data["aws_secret_access_key"]
            bucket_name = json_data["container"]
            s3_folder = json_data["folder"]
            
            update_database(profile_id=profile_id, project_name=project_name, status='DOWNLOADING', message='Dowloading dataset ')
            

            try:
        
                download_from_S3(aws_access_key_id, aws_secret_access_key, bucket_name, s3_folder, AllDatasets)
            
            except Exception as e:

                update_database(profile_id=profile_id, project_name=project_name, status='FAILED', message='Dowloading failed ')
                
                raise Exception(e)
                
            directory_dataset = AllDatasets + s3_folder
            update_database(profile_id=profile_id, project_name=project_name, status='DOWNLOALED')

            
            
            # The dataset is zipped
            if json_data["zipped"] ==  True or ('zip' in os.listdir(directory_dataset)[0] and len(os.listdir(directory_dataset)) == 1):

                write_logs(file_log, controler='Dataset',  action='download_and_training', profile_id = profile_id, project_name=project_name, message= 'Dezipping dataset')
                # Zip verification
                if len(os.listdir(directory_dataset)) != 1:
                    raise Exception('Zipped dataset should have only one file .zip')
                    
               
                parent_zipped = directory_dataset
                download_file_path = parent_zipped + '/' + os.listdir(parent_zipped)[0]
                
                if "zip" not in download_file_path:
                    raise Exception('Unable to find a zip file .' + str(os.listdir(parent_zipped)[0]) + ' is not a zipped file' )    


                
                with zipfile.ZipFile(download_file_path, 'r') as zip_ref:
                    zip_ref.extractall(parent_zipped)
                
                os.remove(download_file_path)
                directory_dataset = parent_zipped + '/' + os.listdir(parent_zipped)[0]
          
        elif storagetype == 'local_windows':
            
            # Just copy dataset to avoid download from cloud
            update_database(profile_id=profile_id, project_name=project_name, status='DOWNLOADING', message='Copying data on local Windows ')
            source_dataset = local_directory_dataset + json_data["folder"]
            directory_dataset = AllDatasets + json_data["folder"]
            shutil.copytree(source_dataset, directory_dataset)



        elif storagetype == 'local_linux':
            
            # Just copy dataset to avoid download from cloud
            update_database(profile_id=profile_id, project_name=project_name, status='DOWNLOADING', message='Copying data on local Linux ')
            source_dataset = local_directory_dataset + json_data["folder"]
            directory_dataset = AllDatasets + json_data["folder"]
            shutil.copytree(source_dataset, directory_dataset)
        else:
            raise Exception("Storage not implemented")

         
         # We check if we have train and validation folder
        directory_dataset = check_exits_train_validation(directory_dataset, model_type)
            
        
        # Data Augmentation
        if model_type == 'classification':
            write_logs(file_log, controler='Dataset',  action='Augmentation', profile_id = profile_id, project_name=project_name, message= 'Augmenting dataset')
        
            update_database(profile_id=profile_id, project_name=project_name, status='DATAPROCESSING', message='Augmenting data ') 
        
        
            dataset_augmentation(directory_dataset)
        
            write_logs(file_log, controler='Dataset',  action='Augmentation', profile_id = profile_id, project_name=project_name, message= 'Dataset augmented. Done')
          
        check_dataset(directory_dataset, model_type)


        
        # Data Encryption

        update_database(profile_id=profile_id, project_name=project_name, status='DATAPROCESSING', message='Encrypting data ') 
        write_logs(file_log, controler='Dataset',  action='Encryption', profile_id = profile_id, project_name=project_name, message= 'Encrypting dataset')

        if model_type == 'classification':
            encrypt_classification_dataset(directory_dataset, key, initialization_vector)
        elif model_type == 'detection':
            encrypt_detection_dataset(directory_dataset, key, initialization_vector)
        else:
            raise Exception("Invalid model type. Should be classification or detection")
        write_logs(file_log, controler='Dataset',  action='Encryption', profile_id = profile_id, project_name=project_name, message= 'Dataset encrypted')
        
        
        if model_type == 'classification':
        
            start_classification_training(profile_id=profile_id, project_name=project_name,
                                  URL = URL,
                                  export = export,
                                  num_epochs=num_epochs,
                                  img_size = img_size,
                                  batch_size=batch_size,
                                  directory_dataset = directory_dataset,
                                  lr=lr,
                                  step_size = step_size,
                                  gamma = gamma, 
                                  channel = channel,
                                  Data_Crypted = Data_Crypted,
                                  krow = krow,
                                  kcol = kcol,
                                  ITER_MAX = ITER_MAX                                  
                                  )
            
        elif model_type == 'detection':
            
            start_detection_training(profile_id=profile_id, project_name=project_name,
                                  URL = URL,
                                  export = export,
                                  num_epochs=num_epochs,
                                  batch_size=1,
                                  directory_dataset = directory_dataset,
                                  lr=lr,
                                  step_size = step_size,
                                  gamma = gamma, img_size = img_size,
                                  channel = channel,
                                  krow = krow,
                                  kcol = kcol,
                                  ITER_MAX = ITER_MAX                                  
                                  )
            
        else:
            raise Exception("Current model type not implemented")
        

        write_logs(file_log, controler='Torch',  action='download_and_training', profile_id = profile_id, project_name=project_name, message= 'training itself is finished. Now we have to update the databse and clean the VM')
        delete_all_data_current_folder(AllDatasets)
          
        local_path_projects =  AllMyProjects + '/' + profile_id + '/' + project_name
        path_remove = AllMyProjects + '/'
        
        upload_folder_to_azure_blob(local_path_projects, path_remove, central_conn_str_azure, central_container_azure)
        
     
        delete_all_data_current_folder(AllMyProjects)
        
        update_database(profile_id=profile_id, project_name=project_name, status='SUCCESS', message='Training finished successfully ')    
  
        TriggerWebHook(profile_id, project_name, message='Training finished successfully', status='SUCCESS', URL=URL)

    except Exception as e:
        
        write_logs(file_log, controler='Torch',  action='download_and_training', profile_id = profile_id, project_name=project_name, message= str(e))

        TriggerWebHook(profile_id, project_name, message='Error during download/training. ' + str(e), status='FAILED', URL=URL)
    
          # # Update databse     

        update_database(profile_id=profile_id, project_name=project_name, status='FAILED', message=str(e))
         
        # delete_all_data_current_folder(AllDatasets)
        
        delete_all_data_current_folder(AllMyProjects)  
        
        # Raising exception for views
        raise Exception(e)
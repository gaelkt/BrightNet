from trainer.models import Training
from trainer.__init__ import *
import datetime
import sys
import os
import numpy as np
import shutil
import ast
import requests
import json
from django import db

from celery import Celery
import platform

from PIL import Image, ImageFilter, ImageChops, ImageEnhance

    
def TriggerWebHook(profile_id, project_name, message='Trainig Aborted', status='FAILED', URL=default_webhook_url):
      
    headers={'Authorization': str(authorization_token)}

    r = requests.post(URL, json={"profile_id": profile_id, "project_name": project_name, "status": status, "message": message}, headers=headers)
    
    write_logs(file_log, controler='', action='TriggerWebHook', profile_id=profile_id, project_name=project_name, message='Webhook sent to URL')
    
    return r



def write_logs(file_log, controler, action='', profile_id='', project_name='', message='', request_body='/'):
    
    data = {}
    data['controler'] = controler
    data['action'] = action
    data['profile_id'] = profile_id
    data['project_name'] = project_name
    data['message'] = message
    data['timestamp'] = str(datetime.datetime.now())
    data['request_body'] = str(request_body)
    
    f = open(file_log, 'a')
    
    try:
          
        json.dump(data, f)
        
    
        f.write('\n')
    
    except Exception as e:

        print('Exception when printing logs. Controler = ' + str(controler) + ' and action is ' + str(action)  + ' filelog = ' + str(file_log) + ' Message = ' + str(e))
      
    finally:
        
    
        f.close
        


def update_database(profile_id, project_name, status, message='', current_epoch=None, loss=None, accuracy=None, epoch_duration=None):
    
    '''
        Function to update the database
        
        Mandatory inputs: profile_id, project_name, status
    
    '''
    try:
        # Query the database
        stored_trainings_object = Training.objects.get(profile_id=profile_id, project_name=project_name)
        
        # General updates
        stored_trainings_object.status = status
        stored_trainings_object.lastupdate = datetime.datetime.now() 
        stored_trainings_object.message = message
        
        
        # Epoch end updates
        if current_epoch != None and loss != None and epoch_duration != None and accuracy != None:
            stored_trainings_object.accuracy = str(np.round(accuracy, 2))
            stored_trainings_object.loss = str(np.round(loss, 2))
            stored_trainings_object.epoch_duration = str(np.round(epoch_duration, 2))
            stored_trainings_object.epoch = current_epoch + 1  # current_epoch in [0, epoch_to_train-1]
                      
            # If it is the last epoch, we change the status to EXPORTING
            epoch_to_train = int(stored_trainings_object.epoch_to_train)
            if current_epoch == epoch_to_train-1:
                stored_trainings_object.status = "EXPORTING"
            
        # Save the updates
        stored_trainings_object.save()
        
        # Logging
        write_logs(file_log, controler='Database', action='update_database', profile_id = profile_id, project_name=project_name, message= 'Updated database. status= ' + str(status) + ' current_epoch =  ' + str(current_epoch))
        
    except Exception as e:
        write_logs(file_log, controler='Database', action='update_database', profile_id = profile_id, project_name=project_name, message= str(e))
        raise Exception(e)
        
    finally:
        
        # Closing the DB connection
        db.connections.close_all()
        


        
def check_dataset(directory_dataset, model_type):
    
    write_logs(file_log, controler='Dataset', action='check_dataset', profile_id = '', project_name='', message= 'Checking dataset')
    

    train_dataset = directory_dataset + '/train'
    val_dataset = directory_dataset + '/val'
       
    if sorted(os.listdir(directory_dataset)) !=  ['train', 'val']:
        raise Exception("Dataset should contain <train> and <val> folders. Instead we got: " + str(os.listdir(directory_dataset)))
    
    if model_type == 'classification':
        if len(os.listdir(train_dataset)) != len(os.listdir(val_dataset)):
            raise Exception("Validation and Train dataset should have the same number of classes")
         
        if len(os.listdir(train_dataset)) <= 1: 
            raise Exception("Dataset should have at least two classes")  
            
    if model_type == 'detection':
        if 'annotations.json' not in os.listdir(train_dataset):
            raise Exception("Cannot find annotations in <train> folder")
        if 'annotations.json' not in os.listdir(val_dataset):
            raise Exception("Cannot find annotations in <val> folder")   
            
    write_logs(file_log, controler='Dataset', action='check_dataset', profile_id = '', project_name='', message= 'Dataset ok !')

def CheckJsonTraining(Json):
    
    write_logs(file_log, controler='', action='CheckJsonTraining', profile_id = '', project_name='', message= 'Checking input training data')
    
    # 1. profile_id           
    if 'profile_id' in Json:
        profile_id = Json['profile_id']
    else:
        raise Exception("Please specicy a valid profile_id")
    # print('1. profile_id ok')
       
    # 2. Project name             
    if 'project_name' in Json:
        project_name = Json['project_name']
    else:
        raise Exception("Please specicy a valid project_name")
    # print('2. project_name ok')



    # 3. model_type           
    if 'model_type' in Json:
        model_type = Json['model_type']
    else:
        # print('Please specify a valide model type')
        raise Exception("Please specify a valid model type")
    
    if model_type!= 'classification' and model_type != 'detection':
        # print('model_type should classification or detection')
        raise Exception("model_type should classification or detection")
    # print('3. model_type ok')


    # 4. webhook_url:
    if 'webhook_url' in Json:
        webhook_url = Json['webhook_url']                   
    else:
        webhook_url = default_webhook_url 
    # print('3. webhook_url ok')
        
    # 5. Export                        
    if 'export' in Json:
        export_to = Json['export']  
        
        if type(export_to) != list:
            # print("Export should be a list")
            raise Exception("Export should be a list. For example ['coreml', 'onnx'] ") 
        
        for export in export_to:
            if export not in all_possible_export:
                # print(str(export) + " is not a valid export")  
                raise Exception(str(export) + " is not a valid export")
    else:
        export_to = default_export
  
    # print('5. export ok')
    
    # We put onnx at first index
    if 'onnx' not in export_to and 'tflite' in export_to:
        # print('Adding onnx export since tflite export...')
        export_to.insert(0, 'onnx')
    
    if 'tflite' in export_to:
        raise Exception("tflite not supported at this time.") 
        # index_tflite = export_to.index('tflite')
        # index_onnx = export_to.index('onnx')
        
        # if index_tflite < index_onnx:
        #     print('tflite comes before onnx ....................')
        #     print('Permuting tflite and onnx....................')
        #     export_to[index_tflite] = 'onnx'
        #     export_to[index_onnx] = 'tflite'
            
    # 6. num_epochs   
    if 'epochs' in Json:
        num_epochs = Json['epochs']                       
    else:
        num_epochs = default_num_epochs
    if type(num_epochs) != int:
        # print("number of epochs should be integer")
        raise Exception("number of epochs should be integer")
    

    if num_epochs < num_epochs_min or num_epochs > num_epochs_max:
        error_msg = 'number of epochs should between {} and {}'.format(num_epochs_min, num_epochs_max)
        # print(error_msg)
        raise Exception(error_msg)
    
    # print('6 . epochs ok')

    # 7. Batch size
    if 'batch_size' in Json:
        batch_size = Json['batch_size']                                            
    else:
        batch_size = default_batch_size        
    if type(batch_size) != int:
        raise Exception("Batch size should be integer")
    if batch_size < batch_size_min or batch_size > batch_size_max:
        error_msg = 'batch size should between {} and {}'.format(batch_size_min, batch_size_max)
        raise Exception(error_msg)
    # print('7. batch_size ok')    
        
    # 8. storage type
    if 'storage_type' not in Json:
        raise Exception("storagetype not found. Please add a fiel storagetype of value aws or azure.") 
    else:
        storage_type = Json['storage_type']
        
    if storage_type == 'aws':
        if 'aws_access_key_id' not in Json:
            raise Exception("Please enter aws_access_key_id") 
        if 'aws_secret_access_key' not in Json:
            raise Exception("Please enter aws_secret_access_key")    
    elif storage_type == 'azure':
        if 'azure_connection_string' not in Json:
            raise Exception("Please enter azure_connection_string") 
    elif storage_type != 'local_windows' and storage_type != 'local_linux':
        raise Exception("storage type should be aws or azure")   
    
    # print('8. storage_type ok') 
        
    # 9. container
    if 'container' not in Json:
        raise Exception("Please enter container") 
    else:
        container = Json['container']
    # print('9. container ok')    
     
    # 10. dataset
    if 'folder' not in Json:
        raise Exception("Please enter folder in the container") 
    else:
        dataset = Json["folder"]
    # print('10. dataset ok')         

    # 11. Learning rate
    if 'learning_rate' in Json:
        learning_rate = Json['learning_rate']
    else:  
        learning_rate = default_learning_rate
    
    if type(default_learning_rate) != int and type(default_learning_rate) != float:
        raise Exception("Learning rate should be float.")
    if default_learning_rate >= learning_rate_max or default_learning_rate < learning_rate_min:
        error_msg = 'Learning rate should between {} and {}'.format(learning_rate_min, learning_rate_max)
        raise Exception(error_msg) 
    # print('11. learning rate ok')     
        
    # 12. step size:
    if 'step_size' in Json:
        step_size = Json['step_size']                   
    else:
        step_size = default_step_size 
    if type(step_size) != int:
        raise Exception("Step size should be integer.")          
    if step_size <= step_size_min:
        error_msg = 'step size should greather than {} '.format(step_size_min)
        raise Exception("Step size should be greater than 1")   
      
    # 13. image_size:
    if 'image_size' in Json:
        image_size = Json['image_size']                   
    else:
        image_size = default_image_size  
    if type(image_size) != int:
        raise Exception("img size should be integer.") 
    if image_size >= image_size_max and image_size <= image_size_min:
        error_msg = 'image size should between {} and {}'.format(image_size_min, image_size_max)
        raise Exception(error_msg)
    # print('13. image_size ok') 
    
    # 14. gamma:
    if 'gamma' in Json:
        gamma = Json['gamma']                   
    else:
        gamma = default_gamma
    if type(gamma) != float:
        raise Exception("gamma should be float.")           
    if gamma >= gamma_max or gamma <= gamma_min:
        error_msg = 'gamma should between {} and {}'.format(gamma_min, gamma_max)

        raise Exception(error_msg)
    # print('14. gamma ok') 
        
        
    # 15. channel:
    if 'channel' in Json:
        channel = Json['channel']  
        
        if channel == 1:
            channel = 3
        
        if type(channel) != int:
            raise Exception('channel should be an integer: 1 or 3')
            
    else:
        channel = default_channel
        
    if channel != 1 and channel !=3:

        raise Exception('channel should 1 or 3 for gray or rgb')
    # print('15. channel ok') 

              
    # 16. Data crypted:
    if 'crypted' in Json:
        data_crypted = Json['crypted']                   
    else:
        data_crypted = default_data_crypted 
    if type(data_crypted) != bool:

        raise Exception("crypted should be bool.") 

    write_logs(file_log, controler='', action='CheckJsonTraining', message= 'Input training data ok')
    return profile_id, project_name, model_type, webhook_url, export_to, num_epochs, batch_size, storage_type, container, dataset, learning_rate, step_size, image_size, gamma, channel, data_crypted 


def delete_all_data_current_folder(directory):
    
    '''
    Remove all data in a sepcific directory
    
    '''

    if os.path.exists(directory):
        folders = os.listdir(directory)
        for folder in folders:
            path_data = directory + '/' + folder
            shutil.rmtree(path_data)
    


def split_train_validation(train_dir, val_dir = None, ratio=0.1):
    
    write_logs(file_log, controler='', action='split_train_validation', message= 'Spliting data into training and validation')
    
    try:
        
        if train_dir == None:
            sys.exit("Please specify training folder")
            
        if val_dir == None:
            val_dir = os.path.dirname(train_dir) + '/validation'
            os.mkdir(val_dir)
            
        if not os.path.exists(train_dir):
            sys.exit("Training Folder does not exist")
            
        
        if not os.path.exists(val_dir):
            os.mkdir(val_dir)
            
        try:
            subfolders = os.listdir(train_dir)
            # print('-------------------------------------------------------------------')
            # print('There are ' + str(len(subfolders)) + ' classes which are ', subfolders)
            # print('-------------------------------------------------------------------')
            
            for subfolder in subfolders:
                # print('      Doing class ' + subfolder)
                
                if not os.path.exists(val_dir + '/' + subfolder):
                    os.mkdir(val_dir + '/' + subfolder)
                    
                files_subfolder = os.listdir(train_dir + '/' + subfolder)
                np.random.shuffle(files_subfolder)
                    
                N = int(np.round(ratio*len(files_subfolder)))
                if N <= 0:
                    # print(subfolder + 'does not have enough files')
                    pass
                else:
                    
                    for i in range(N):
                        shutil.copyfile(train_dir + '/' + subfolder + '/' + files_subfolder[i], val_dir + '/' + subfolder + '/' + files_subfolder[i]) 
              
        except Exception as e:
            # print('Unable to split classes')
            # print(e)
            write_logs(file_log, controler='', action='split_train_validation', message= str(e))
            raise Exception(e)
    except Exception as e:
        write_logs(file_log, controler='', action='split_train_validation', message= str(e))
        raise Exception(e)
        # print('Unable to split train and validation data')
        # print(e)
    write_logs(file_log, controler='', action='split_train_validation', message= 'Split train/validation finished')


def check_exits_train_validation(dataset, model_type):
    '''
    Check if a train and validation folder exists.
    Otherwise create a new one
    Inputs:
        datset: orignal dataset
        output: new dataset with train and validation folder
    
    '''
    
    # check if train/val exists
    subfolders = sorted(os.listdir(dataset))
    parent, child = os.path.split(dataset)
    if subfolders != ['train', 'val']:
        if model_type == 'detection':
            raise Exception("Detection dataset should have <train> and <val> folders. Instead we got " + str(subfolders))
        elif model_type == 'classification':
            train_dir = parent + '/train'
            val_dir = parent + '/val'
            os.rename(dataset, train_dir)
            split_train_validation(train_dir, val_dir = val_dir, ratio=0.1)
            return parent
        else:
            raise Exception("Invalid model_type. Should be classification or detection")
    else:
        return dataset
    
    
def check_authorization_token(request, Authorization):
    # print(request.headers)
    JsonHeader = ast.literal_eval(str(request.headers))
    if 'Authorization' not in JsonHeader:
        
        try:
            write_logs(file_log, controler='Django', action='check_authorization_token', profile_id='', project_name='', message='No authoziation token', request_body=request.body)
            TriggerWebHook(profile_id='', project_name='', message='No authorization token', status='FAILED', URL=default_webhook_url)
            
        finally:
            raise Exception("No authorization header. Please provide one")
            
    elif JsonHeader['Authorization'] != Authorization:
        try:
            write_logs(file_log, controler='Django', action='check_authorization_token', profile_id='', project_name='', message='Invalid authoziation token', request_body=request.body)
            TriggerWebHook(profile_id='', project_name='', message='Invalid authorization token', status='FAILED', URL=default_webhook_url) 
        finally:

            raise Exception("Invalid authorization header")
    
    else:
        write_logs(file_log, controler='Django', action='check_authorization_token', profile_id='', project_name= '', message='Authorization token ok', request_body=request.body)

            
    
def check_celery_server():
    
    ''' 
        Check the status of celery server
        
    '''
    
    try:

        app = Celery('trainer',
        broker="amqp://guest@localhost//",
        backend="rpc")
            
        all_list = app.control.inspect()
            
        active_task = all_list.active()
            

        Workers = app.control.ping()
        
        # There is no workers
        if Workers == []:
            
            message = 'Celery not started'
            return {'status': 'BUSY', 'message': 'Celery not started', 'code':404}
            
        # There are some workers
        else:
    
            worker = list(Workers[0].keys())[0]
            aliveness = Workers[0][worker]['ok']
    
            # worker not alive
            if aliveness != 'pong':
                
                message = 'Celery not ready'
                return {'status': 'BUSY', 'message':message, 'code':404}
                
            else:
                pass

            celer_server_name = 'celery@'+ str(platform.node())
            if celer_server_name in active_task:
                current_active_tasks = active_task[celer_server_name]
            else:
                return {'status': 'BUSY', 'message':str(celer_server_name) + ' not in active task', 'code':404}

            
            # Some tasks are active      
            if current_active_tasks != []:
                
                task = current_active_tasks[0]

                if 'id' in task:

                    if 'args' in task:
                    
                        task_args=task['args']

                        profile_id = task_args[0]
                        project_name = task_args[1]
  
                    
                        return {'status': 'BUSY', 'message':'Django ready but Celery busy with ' + str(profile_id) + '/' + str(project_name), 'project_name': str(project_name), 'profile_id': str(profile_id), 'code':404}
                    else:

                        return {'status': 'BUSY', 'message':'Unidentified tasks (args) are running', 'code':404}
                        
                else:

                    return {'status': 'BUSY', 'message':'Unidentified tasks (id) are running', 'code':404}
            else:   
                     
                return {'status': 'READY', 'message':'Django and Celery are ready', 'code':200}
                     

    except Exception as e:
    
        return {'status': 'BUSY', 'message':str(e), 'code':404}    
    
    

def remove_from_string(name, expr):
    
    return name.replace('.', '_' + expr + '_.')

def augmentation(source_folder, augmented_folder, flip_top_bottom=True, flip_left_right=True, angles=[90, 180, 27], blur=False, 
                      
                      horizontal_shift=0, vertical_shift=0, brightness=1):
    
    '''
    Data augmentation function to augment images from a source folder to a destination folder
    
    Parameters
    ----------
    source_folder : STRING
        Folder containing images.
    augmented_folder : STRING
        Folder to put augmented images.

    flip_top_bottom : BOOL
        


    Returns
    -------
    None.

    '''
    
    

    noise = 0.6 # Gaussian noise
    image_name_list = os.listdir(source_folder)
    
    for i in range(len(image_name_list)):
        
        patch = Image.open(source_folder + '/' + image_name_list[i])
        
        # Convert RGBA and P mode to RGB
        if patch.mode in ("RGBA", "P"):
            patch = patch.convert("RGB")
        
        # Vertical
        if flip_top_bottom:
            patch_vertical_flip_top_bottom = patch.transpose(Image.FLIP_TOP_BOTTOM)
            patch_vertical_flip_top_bottom.save(augmented_folder + '/' + remove_from_string(image_name_list[i], 'top_bottom'))
            
            # Blur
            if blur:
                patch_vertical_flip_top_bottom_blur = patch_vertical_flip_top_bottom.filter(ImageFilter.GaussianBlur(noise))
                patch_vertical_flip_top_bottom_blur.save(augmented_folder + '/' +remove_from_string(image_name_list[i], 'blur_top_bottom'))
              
        # Horizontal
        if flip_left_right:
            patch_horizontal_flip_top_bottom = patch.transpose(Image.FLIP_LEFT_RIGHT)
            patch_horizontal_flip_top_bottom.save(augmented_folder + '/' + remove_from_string(image_name_list[i], 'left_right'))
            
            # Blur
            if blur:
                patch_horizontal_flip_top_bottom_blur = patch_horizontal_flip_top_bottom.filter(ImageFilter.GaussianBlur(noise))
                patch_horizontal_flip_top_bottom_blur.save(augmented_folder + '/' + remove_from_string(image_name_list[i], 'blur_left_right'))
        

        # Rotations
        for angle in angles:
            patch_rot = patch.rotate(angle)
            patch_rot.save(augmented_folder + '/' + remove_from_string(image_name_list[i], 'rot_' + str(angle)))
     
        # Horizontal shift
        if horizontal_shift != 0:
            patch_hor_shift_1 = ImageChops.offset(patch, horizontal_shift, 0)
            patch_hor_shift_1.save(augmented_folder + '/' + remove_from_string(image_name_list[i], 'hor_shift_1'))
            
            patch_hor_shift_2 = ImageChops.offset(patch, -horizontal_shift, 0)
            patch_hor_shift_2.save(augmented_folder + remove_from_string(image_name_list[i], 'hor_shift_2'))

        # Vertical shift
        if vertical_shift != 0:
            patch_ver_shift_1 = ImageChops.offset(patch, vertical_shift, 0)
            patch_ver_shift_1.save(augmented_folder + '/' + remove_from_string(image_name_list[i], 'vert_shift_1'))
            
            patch_ver_shift_2 = ImageChops.offset(patch, -vertical_shift, 0)
            patch_ver_shift_2.save(augmented_folder +  '/' + remove_from_string(image_name_list[i], 'vert_shift_2'))
            
        # Brightness
        if brightness != 1:
            patch_enhancer = ImageEnhance.Brightness(patch)
            patch_enhancer_brightness = patch_enhancer.enhance(brightness)
            patch_enhancer_brightness.save(augmented_folder +  '/' + remove_from_string(image_name_list[i], 'brightness'))



def dataset_augmentation(dataset):
    
    '''
    
    Function to augment a classification dataset. 
    
    The folder dataset must have a subfolder 'train'
    
    The subfolder 'train' must have subfolder representing classes.
    
    The supported operations are: (will be completed later)
    
    ***Input: dataset folder
    
    ***Return None
    
    '''
    
    train_dataset = dataset + '/train'
    
    classes_names = os.listdir(train_dataset)
    
    for classe in classes_names:
        
        source_folder = train_dataset + '/' + classe
        
        augmentation(source_folder, source_folder, flip_top_bottom=False, flip_left_right=True, angles=[], blur=False, 
                      
                      horizontal_shift=0, vertical_shift=0, brightness=1)
        
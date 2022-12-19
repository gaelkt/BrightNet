import json


from django.http import JsonResponse

from django.views.decorators.csrf import csrf_exempt

from .models import Training
from .__init__ import *
from aitrainer.helper_general import TriggerWebHook, CheckJsonTraining, update_database, check_authorization_token, write_logs
from aitrainer.helper_general import check_celery_server
from aitrainer.celery import download_and_training
from datetime import date
import datetime
import platform
from celery import Celery
from django import db
db.connections.close_all()


@csrf_exempt
def form_start_training(request):
        
    write_logs(file_log, controler='Django', action='form_start_training', profile_id = '', project_name='', message='Starting line 1', request_body=request.body)

    check_authorization_token(request, authorization_token)
    

    if request.method == 'POST':
                    
        json_data = json.loads(request.body)
        
        
        if len(json_data) < nb_required_parameters:
            raise Exception('The request does not have enough parameters')
           
        # We get the training parameters and check if they are ok
        profile_id, project_name, model_type, webhook_url, export_to, num_epochs, batch_size, storage_type, container, dataset, learning_rate, step_size, image_size, gamma, channel, data_crypted = CheckJsonTraining(json_data)
              

        # We query the database for previous simimar entries
        stored_trainings = Training.objects.filter(profile_id=profile_id).filter(project_name=project_name)
        
        # Number of previous entries with profile_id and project_name
        nb_previous_entry = stored_trainings.count()
        
        if nb_previous_entry >= 1:
            raise Exception('Found previous project with profile_id =  ' + profile_id + ' and project_name = ' + project_name)
            
        # Creating a new entry in the database
        mytraining = Training(created = date.today(), last_update = datetime.datetime.now(), profile_id = profile_id, project_name = project_name, model_type=model_type, message = 'ok', epoch = 0, loss = 'N/A', accuracy = 'N/A', epoch_to_train=int(num_epochs), epoch_duration = 'N/A')    
        mytraining.save()
        write_logs(file_log, controler='Django', action='form_start_training', profile_id = profile_id, project_name=project_name, message= 'New training created in the database', request_body=request.body)
             
        # Download and training

        download_and_training.delay(profile_id, project_name, model_type, json_data, webhook_url, export_to, num_epochs, batch_size, learning_rate, channel, AllDatasets, step_size = 10,
                                                       gamma = gamma, img_size = image_size, Data_Crypted = False)
        
        write_logs(file_log, controler='Django', action='form_start_training', profile_id = profile_id, project_name=project_name, message= 'download_and_training ok', request_body=request.body)

        # Jsonresponse
        return JsonResponse({'result': 'PENDING', 'project_name': project_name, 'profile_id': profile_id, 'dataset': dataset, 'message':'Training pending'})
                     
    # If not POST
    else:    
        raise Exception('Request should be POST')


            
            
@csrf_exempt
def check_server(request):
    
     
    try:
        write_logs(file_log, controler='django', action='check_authorization_token', profile_id='', project_name='', message='', request_body=request.body)
        check_authorization_token(request, authorization_token)
    except Exception as e:
        return JsonResponse({'status': 'BUSY', 'message': str(e)}, status=404)
        
    
    
    if request.method == 'GET':
           
        try:
            
            output_celery = check_celery_server()
            
            state, message, code = output_celery['status'], output_celery['message'], output_celery['code']
            
            
            if code == 200:
                
                return JsonResponse({'status': 'READY', 'message': 'Django and Celery are ready'})
            
            else:
                
                return JsonResponse({'status': state, 'message':message}, status=404)
                

        except Exception as e:
    
            return JsonResponse({'status': 'BUSY', 'message':str(e)}, status=404)

    else:
        return JsonResponse({'status': 'BUSY', 'message':'Checkserver accepts only GET Method'}, status=404)
            
        


@csrf_exempt
def form_status_training(request, profile_id=None, project_name=None):
    

    write_logs(file_log, controler='Django', action='form_status_training', profile_id='', project_name='', message='starting status training', request_body=request.body)
    
    check_authorization_token(request, authorization_token)

        
    if request.method == 'GET':
        
        if profile_id is None:
            raise Exception("profile_id is None")
            

        if project_name is None:
                raise Exception("project_name is None")       
                
   
        stored_trainings = Training.objects.filter(project_name=project_name, profile_id=profile_id)
        nb_previous_entry = stored_trainings.count()
                    
                        
        if nb_previous_entry < 1:
            raise Exception('The project with that profile_id and name does not exist')
            
        elif nb_previous_entry >= 2:
            raise Exception('Several projects with that profile_id and name  that name do exist')
                      
        
        stored_trainings = Training.objects.get(project_name=project_name, profile_id=profile_id)

                
        if stored_trainings != None:
                            
            result = {}
                        
            status = stored_trainings.status
            accuracy = stored_trainings.accuracy
            loss = stored_trainings.loss 
            status = stored_trainings.status
            last_update = stored_trainings.last_update
            epoch_duration = stored_trainings.epoch_duration
            message = stored_trainings.message
            current_epoch = stored_trainings.epoch 
            epoch_to_train = stored_trainings.epoch_to_train
                        
            result['result'] = status
            result['epochs'] = epoch_to_train
                        
            result['current_epoch'] = current_epoch 
            result['loss'] = loss
            result['accuracy'] = accuracy
            result['last_epoch_duration_minutes'] = epoch_duration
                    
            result['last_update'] = last_update
            result['message'] = str(message)
     
            write_logs(file_log, controler='Django', action='form_status_training', profile_id = profile_id, project_name=project_name, message='Returning training data', request_body=request.body)
            
            return JsonResponse(result)
                        
        else:
            raise Exception('Issue to get values in the database')

    else:
        raise Exception('Status training accept only GET method')





 


from django.http import JsonResponse
from django.utils.deprecation import MiddlewareMixin
from aitrainer.helper_general import TriggerWebHook
from django.conf import settings
from django.http import HttpResponse
import traceback
import json
from trainer.__init__ import default_webhook_url, file_log
from aitrainer.helper_general import TriggerWebHook, update_database, write_logs
from trainer.models import Training

class TrainerMiddleware(MiddlewareMixin):

    # Not necessary may be ?
    def __init__(self, get_response):
        self.get_response = get_response
        # self.exception = None


    def __call__(self, request):
        # print('Request received')
        # self.process_request(request)  # Call process_request()
        # self.process_exeption(request)  # Call process_request()
        response = self.get_response(request)
        return response
        
    def process_exception(self, request, exception):
            
        A = '/starttraining' in str(request)
        
        # Statustraining request
        if '/statustraining' in str(request):
            request_split = str(request).split("/")
            profile_id, project_name = request_split[2], request_split[3]
            write_logs(file_log, controler='Django', action='statustraining', profile_id = profile_id, project_name=project_name, message= str(exception), request_body=request.body)
            return JsonResponse({'result': 'FAILED', 'project_name': project_name, 'profile_id': profile_id, 'message':str(exception)}, status=404)
        
        # Checkserver request
        elif '/checkserver' in str(request):
            write_logs(file_log, controler='Django', action='checkserver', profile_id = '', project_name='', message= str(exception), request_body=request.body)
            return JsonResponse({'status': 'FAILED', 'project_name': '', 'profile_id': '', 'message':str(exception)}, status=404)
            
        # Starttraining request
        elif  '/starttraining' in str(request):
            
            try:
            
                # Body is empty
                if request.body == b'':
                    project_name, profile_id, webhook_url = '', '', default_webhook_url
                    write_logs(file_log, controler='Django', action='starttraining', profile_id = '', project_name='', message= str(exception), request_body=request.body)
                    TriggerWebHook(profile_id=profile_id, project_name=project_name, message=str(exception), status='FAILED', URL=webhook_url)
                    return JsonResponse({'result': 'FAILED', 'project_name': '', 'profile_id': '', 'message':str(exception)}, status=404)
            
                # Body is not empty
                else:
                
                    # We get the parameters
                    Json = json.loads(request.body)
                
                    if 'profile_id' in Json:
                        profile_id = Json['profile_id']
                    else:
                        profile_id = ''
           
                    if 'project_name' in Json:
                        project_name = Json['project_name']
                    else:
                        project_name = ''
            

                    if 'webhook_url' in Json:
                        webhook_url = Json['webhook_url']                   
                    else:
                        webhook_url = default_webhook_url 
                
                    # We write logs
                    write_logs(file_log, controler='Django', action='starttraining.py', profile_id = profile_id, project_name=project_name, message= str(exception), request_body=request.body)
        
                    # Update database eventually and send webhook eventually for training endpoint
                    if profile_id != '' and project_name != '':
            
                        # Update database

                        stored_trainings = Training.objects.filter(profile_id=profile_id).filter(project_name=project_name)

                        nb_previous_entry = stored_trainings.count()

                        if nb_previous_entry == 1:

                            update_database(profile_id=profile_id, project_name=project_name, status='FAILED', message=str(exception))   

            
                    # Send webhook
                    TriggerWebHook(profile_id=profile_id, project_name=project_name, message=str(exception), status='FAILED', URL=webhook_url)
                 
                    # Jsonresponse
                    return JsonResponse({'result': 'FAILED', 'project_name': project_name, 'profile_id': profile_id, 'message':str(exception)}, status=404)
            
            except Exception as e:
                
                try:
                
                    TriggerWebHook(profile_id=profile_id, project_name=project_name, message=str(exception) + '   ' + str(e), status='FAILED', URL=webhook_url)

                finally:
                    
                    return JsonResponse({'result': 'FAILED', 'project_name': project_name, 'profile_id': profile_id, 'message':str(exception) + '   ' + str(e)}, status=404)

    # exécutée quand Django reçoit une requête et doit décider la vue à utiliser
    def process_request(self, request):
        # print('Request processed')
        pass
     
    # exécutée lorsque Django apelle la vue. On peut donc récupérer les arguments de la vue
    # view_func est la fonction Python que Django est sur le point d'utiliser. 
    def process_view(self, request, view_func, view_args, view_kwargs):
        # print('View processed')
        pass
     

      
     
    # La vue a été executée mais pas encore de compilation de template 
    # ( il est encore possible de chager le template )
    def process_template_response(self, request, response):
        # print('RESPONSEEE2')
        return response
     
    # Tout est executée, dernier recours avant le retour client 
    def process_response(self, request, response):
        return response

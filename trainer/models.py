from django.db import models

# Create your models here.

class Training(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    last_update = models.DateTimeField(auto_now=True)
    project_name = models.CharField(max_length=100, blank=False, default='')
#    task_id = models.CharField(max_length=100, blank=False, default='')
    profile_id = models.CharField(max_length=100, blank=False, default='')
    model_type = models.CharField(max_length=100, blank=False, default='classification')
    status = models.CharField(max_length=20, blank=False, default='PENDING')
    message = models.CharField(max_length=1000, default='classification')
    epoch = models.IntegerField(default=0)
    epoch_to_train = models.IntegerField()
    loss = models.CharField(max_length=1000, default='N/A')
    accuracy = models.CharField(max_length=1000, default='N/A')
    epoch_duration = models.CharField(max_length=1000, default='N/A')
    class Meta:
        app_label = 'trainer'

    
    
    

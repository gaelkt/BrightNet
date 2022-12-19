from __future__ import print_function, division
import torch
#torch.backends.quantized.engine = 'qnnpack'
#print('Quantized backend set to qnnpack')
import os

import torch.nn as nn
import torch.optim as optim

from torchvision import models
from torch.utils.mobile_optimizer import optimize_for_mobile


import sys
import coremltools as ct



from .helper_training import data_loader, train_model, on_train_begin, on_epoch_end, on_train_end
from .helper_general import TriggerWebHook
from .helper_general import write_logs

from trainer.__init__ import AllMyProjects, AllTensorboard
from trainer.__init__ import all_possible_export
from trainer.__init__ import file_log
from trainer.__init__ import initialization_vector, key
from trainer.__init__ import num_workers

import onnx

from .helper_training_detection import EncryptedDetectionDataset, data_transformation_detection, build_detection_model
from .helper_training_detection import save_jit_model, _convert_slice_v9
# from .detection.engine import train_one_epoch, evaluate

# from .detection import utils
import time
import numpy as np







def start_detection_training(profile_id, project_name, URL = '',
    model_type = "detection", 
    directory_dataset = "Front",
    model='fasterrcnn_resnet50_fpn',
    lr=0.005,
    batch_size=24,
    num_epochs=2,
    img_size = 224,
    channel=3,
    gamma = 0.5,
    step_size = 10,
    export = ["coreml"],
    krow = 1,
    kcol = 1,
    ITER_MAX = 1):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    
    
    write_logs(file_log, controler='Torch',  action='start_classification_training', profile_id = profile_id, project_name=project_name, message= 'Starting training itself for classification')
    
    
# # Checking mandatory arguments  
#     if profile_id == None:
#         raise Exception("profile_id is None")  
        
#     if project_name == None:
#         raise Exception("Project name is None")

#     if directory_dataset == None:
#         raise Exception("Directory dataset is None")

#     if model_type == None:
#         raise Exception("model_type is None")


#     if model_type != 'detection':
#         raise Exception("Model type should be detection")        

#     if model not in ['fasterrcnn_resnet50_fpn']:
#         raise Exception("Invalid detection model")        
        
    
#     # Creating Project Folder
#     project_folder = AllMyProjects + profile_id + '/' + project_name 
#     if not os.path.exists(project_folder):
#         os.makedirs(project_folder)
        
        
#     # Export Name
#     dirname, basename = os.path.split(project_folder)
    
#     project_folder = project_folder 
#     if not os.path.exists(project_folder):
#         os.mkdir(project_folder)
    
#     # Tensorboard 
#     tensorboard_dir = AllTensorboard + profile_id + '/' + project_name 
#     if not os.path.exists(tensorboard_dir):
#         os.makedirs(tensorboard_dir)    
    
  
#     basename_version = basename 

#     # Saving names for export
#     model_name_keras =  basename_version + '.h5'
#     model_name_torch =  basename_version + '.pt'
#     model_name_tflite = basename_version + '.tflite' 
#     model_name_coreml = basename_version + '.mlmodel'  
#     model_name_frozen = basename_version + '.pb' 
#     model_name_onnx = basename_version + '.onnx' 
#     model_name_pytorchmobile = basename_version + '.ptl' 
#     name_log_file = basename_version + '.txt'
    
    
    

#     CURRENT_DIR = project_folder
    
#     # Creating models Folder
#     if not os.path.exists(CURRENT_DIR + '/models'):
#        os.mkdir(CURRENT_DIR + '/models')

#     # Creating a Save subfolder for each export type
#     # print('Export = ', export)
#     for export_type in export:
#        # print(export_type)
#        if export_type not in all_possible_export:
#            raise Exception("Export should be tflite, keras, coreml, frozen, torch, pytorchmobile or onnx")
#            # sys.exit("Please export should be tflite, keras and/or coreml")
#        PATH_SAVE_MODEL = CURRENT_DIR + '/models/' + export_type
#        if not os.path.exists((PATH_SAVE_MODEL)):
#            os.mkdir(PATH_SAVE_MODEL)
    
#     PATH_SAVE_LOGS = CURRENT_DIR + '/models/' + "logs" 
#     if not os.path.exists((PATH_SAVE_LOGS)):
#            os.mkdir(PATH_SAVE_LOGS)
           
#     log_file = PATH_SAVE_LOGS  + '/' + name_log_file




#     # Train and Val dataset
#     dataset_train = EncryptedDetectionDataset(dataset=directory_dataset + '/train', channel=channel, key=key, initialization_vector=initialization_vector, name='name', transformations=data_transformation_detection(train=True))
#     dataset_val = EncryptedDetectionDataset(dataset=directory_dataset + '/val', channel=channel, key=key, initialization_vector=initialization_vector, name='name', transformations=data_transformation_detection(train=False))

    

    
    
#     # Number of classes
#     if sorted(dataset_train.CLASS_NAMES) != sorted(dataset_val.CLASS_NAMES):
#         raise Exception("Validation and Train dataset should have the same number of classes")
        
#     num_classes = len(dataset_train.CLASS_NAMES)    
    
    
#     # define training and validation data loaders
#     data_loader_train = torch.utils.data.DataLoader(
#         dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
#         collate_fn=utils.collate_fn)

#     data_loader_val = torch.utils.data.DataLoader(
#         dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
#         collate_fn=utils.collate_fn)
    
    
#      # get the model using our helper function
#     model_detection = build_detection_model(num_classes)
    
#     # move model to the right device
#     model_detection.to(device)

#     # construct an optimizer
#     params = [p for p in model_detection.parameters() if p.requires_grad]
    
#     optimizer = torch.optim.SGD(params, lr=lr,
#                                 momentum=0.9, weight_decay=0.0005)

#     # and a learning rate scheduler which decreases the learning rate by
#     # 10x every 3 epochs
#     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                    step_size=step_size,
#                                                    gamma=gamma)
    
#     # On train begin
#     on_train_begin(profile_id, project_name, log_file)
#     #start training

#     for epoch in range(num_epochs):
        

        
#         since = time.time()
#         # train for one epoch, printing every 10 iterations

#         loss_value, accuracy_value = train_one_epoch(model_detection, optimizer, data_loader_train, device, epoch, print_freq=10)
        

#         # update the learning rate

#         lr_scheduler.step()
#         # evaluate on the test dataset

#         evaluate(model_detection, data_loader_val, device=device)

        
#         time_elapsed = np.round((time.time() - since)/60.0, 2)
        
#         # End of each epoch
#         on_epoch_end(profile_id, project_name, current_epoch=epoch, num_epochs=num_epochs, log_file=log_file, loss=float(loss_value), accuracy=accuracy_value, epoch_duration=time_elapsed)
    

    
#     for export_type in export:
#         if export_type == "torch":

#             PATH_SAVE_MODEL = CURRENT_DIR + '/models/' + export_type + '/' + model_name_torch

#             # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             device = torch.device("cpu")
            
#             model_detection = model_detection.to(device)
#             jit_model_detection = save_jit_model(model_detection, PATH_SAVE_MODEL, img_size=img_size, script=True, device=device)
#             # torch.save(model_detection, PATH_SAVE_MODEL)
#             write_logs(file_log, controler='Torch',  action='start_detection_training', profile_id = profile_id, project_name=project_name, message= 'Export done for torch')
            
            




#         if export_type == "onnx":
            

#             PATH_SAVE_MODEL_ONNX = CURRENT_DIR + '/models/' + export_type + '/' + model_name_onnx
#             dummy_input = torch.randn(1, channel, img_size, img_size).to(device)
#             model_detection = model_detection.to(device)
#             torch.onnx.export(model_detection, dummy_input, PATH_SAVE_MODEL_ONNX, verbose=False, input_names=['Image'], output_names=['Predictions'],
#                               opset_version=12)
                    
            
#             # onnx_model = onnx.load(PATH_SAVE_MODEL_ONNX)
#             # onnx.checker.check_model(onnx_model)
            
#             # # coreml_model = ct.converters.onnx._converter.convert(model=PATH_SAVE_MODEL_ONNX)
            
#             # coreml_model = ct.converters.onnx.convert(
#             #     onnx_model,
#             #     # disable_coreml_rank5_mapping=True,
#             #     custom_conversion_functions={"Slice": _convert_slice_v9}
#             #     )
#             # coreml_model.save('my_model.mlmodel')
            
            
#             # # coreml_model  = ct.converters.onnx.convert(model=PATH_SAVE_MODEL_ONNX)
#             # # ct.converters.onnx.convert(PATH_SAVE_MODEL_ONNX, minimum_ios_deployment_target="13")
            
#             # print('Good for coreml from ONNX ********************')
#             # print('***************************')
            
#             # Metadata
            
#             # description of the model

            
#             description = "Model to detect object in an images. Possible objects are " + str(dataset_train.CLASS_NAMES)
            
#             # Output
#             output_model = "Boxes, scores and classes"
            
#             # Input
#             input_model = "Array(int8) of shape [1, " + str(channel) + ", " + str(img_size) + ", " + str(img_size) + "] (channel first)"
            
#             # metadata
#             metadata = {"input": input_model, "output": output_model, "classes":dataset_train.CLASS_NAMES, "description": description}
            
#             # version and author
#             version = 0
#             author = 'SightCall'

#             # Writing metadata
#             modelonnx = onnx.load(PATH_SAVE_MODEL_ONNX)
#             modelonnx.model_version = version
#             modelonnx.doc_string = str(metadata)
#             modelonnx.producer_name = author

#             onnx.save(modelonnx, PATH_SAVE_MODEL_ONNX)           
                     
#             # Writing Logs when it is done
#             write_logs(file_log, controler='Torch',  action='start_classification_training', profile_id = profile_id, project_name=project_name, message= 'Export done for onnx')

#         if export_type == "pytorchmobile":
            
#             write_logs(file_log, controler='Torch',  action='start_detection_training', profile_id = profile_id, project_name=project_name, message= 'Export done for pytorchmobile')
            
     
#         if export_type == "coreml":
            

#             write_logs(file_log, controler='Torch',  action='start_classification_training', profile_id = profile_id, project_name=project_name, message= 'Export done for coreml')
      

       
#     on_train_end(profile_id, project_name, URL)

    
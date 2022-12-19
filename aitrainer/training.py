from __future__ import print_function, division
import torch
#torch.backends.quantized.engine = 'qnnpack'
#print('Quantized backend set to qnnpack')
import os

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

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

import onnx


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def start_classification_training(profile_id, project_name, URL = '',
    project_type = "classification", 
    directory_dataset = "Front",
    model='mobilenet',
    lr=0.005,
    batch_size=24,
    num_epochs=2,
    img_size = 224,
    channel=3,
    gamma = 0.5,
    step_size = 10,
    export = ["coreml"],
    Data_Crypted = False,
    krow = 1,
    kcol = 1,
    ITER_MAX = 1):
    
    write_logs(file_log, controler='Torch',  action='start_classification_training', profile_id = profile_id, project_name=project_name, message= 'Starting training itself for classification')
    

# Checking mandatory arguments  
    if profile_id == None:
        TriggerWebHook('', project_name, message='profile_id is none', status='FAILED', URL=URL)
        raise Exception("profile_id is None")  
        
    elif project_name == None:
        TriggerWebHook(profile_id, '', message='TrTraining Ended Successfully', status='FAILED', URL=URL)
        raise Exception("Project name is None")


# Checking dataset
    if directory_dataset == None:
        TriggerWebHook(profile_id, project_name, message='Directory is None', status='FAILED', URL=URL)
        raise Exception("Directory dataset is None")
        # sys.exit("Please specify train datasets")
        
    # directory_dataset = AllDatasets + directory_dataset 
        
    # if os.listdir(directory_dataset) !=  ['train', 'val']:
    #     sys.exit("Dataset should have train and validation set")
        
    # train_dataset = directory_dataset + '\\train'
    # val_dataset = directory_dataset + '\\val'
        
    # if len(os.listdir(train_dataset)) != len(os.listdir(val_dataset)):
    #     sys.exit("Train and Val datasets should have same size")
    
    # elif len(os.listdir(train_dataset)) <= 1:
    #     sys.exit("Dataset should have at least two classes")
        

    if project_type == None:
        TriggerWebHook(profile_id, project_name, message='Project is None', status='FAILED', URL=URL)
        raise     ("Project type is None")
        # sys.exit("Please type of projects")


    if project_type not in ['classification', 'detection', 'segmentation']:
        TriggerWebHook(profile_id, project_name, message='Project should be classification, detection or segmentation', status='FAILED', URL=URL)
        raise Exception("Project type should be classification, detection or segmentation")        
        # sys.exist('Please choose a project type between classification, detection and segmentation')
    

    if project_type == 'classification':
        if model not in ['resnet', 'mobilenet']:
            TriggerWebHook(profile_id, project_name, message='model should be resnet or mobilenet', status='FAILED', URL=URL)
            raise Exception("Project type should be resnet pr mobilenet")            
            # sys.exist('Please choose a model between mobilenet and resnet')
         

    # Creating Project Folder
    
    project_folder = AllMyProjects + profile_id + '/' + project_name 
    if not os.path.exists(project_folder):
        os.makedirs(project_folder)
        
        

    # Export Name
    dirname, basename = os.path.split(project_folder)
    
    project_folder = project_folder 
    if not os.path.exists(project_folder):
        os.mkdir(project_folder)
    
    # Tensorboard 
    tensorboard_dir = AllTensorboard + profile_id + '/' + project_name 
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)    
    
  
    basename_version = basename 

    # Saving names for export
    model_name_keras =  basename_version + '.h5'
    model_name_torch =  basename_version + '.pth'
    model_name_tflite = basename_version + '.tflite' 
    model_name_coreml = basename_version + '.mlmodel'  
    model_name_frozen = basename_version + '.pb' 
    model_name_onnx = basename_version + '.onnx' 
    model_name_pytorchmobile = basename_version + '.ptl' 
    name_log_file = basename_version + '.txt'
    
    
    

    CURRENT_DIR = project_folder
    
    # Creating models Folder
    if not os.path.exists(CURRENT_DIR + '/models'):
       os.mkdir(CURRENT_DIR + '/models')

    # Creating a Save subfolder for each export type
    # print('Export = ', export)
    for export_type in export:
       # print(export_type)
       if export_type not in all_possible_export:
           raise Exception("Export should be tflite, keras, coreml, frozen, torch, pytorchmobile or onnx")
           # sys.exit("Please export should be tflite, keras and/or coreml")
       PATH_SAVE_MODEL = CURRENT_DIR + '/models/' + export_type
       if not os.path.exists((PATH_SAVE_MODEL)):
           os.mkdir(PATH_SAVE_MODEL)
    
    PATH_SAVE_LOGS = CURRENT_DIR + '/models/' + "logs" 
    if not os.path.exists((PATH_SAVE_LOGS)):
           os.mkdir(PATH_SAVE_LOGS)
           
    log_file = PATH_SAVE_LOGS  + '/' + name_log_file

    # Classes
    
    class_names_train = os.listdir(directory_dataset + '/train') 
    class_names_val = os.listdir(directory_dataset + '/val') 
    
    if class_names_train != class_names_val:
        write_logs(file_log, controler='Torch',  action='start_classification_training', profile_id = profile_id, project_name=project_name, message= 'Training classes different than validation')
        TriggerWebHook(profile_id, project_name, message='Training classes different than validation', status='FAILED', URL=URL)
        raise Exception("Training classes different than Validation classes ...")
        # sys.exit('Training classes different than Validation classes ....')
    else:
        class_names = class_names_train
        
    classes = len(class_names)
    # print('-------------------------------------------------------------------')
    # print('There are ' + str(classes) + ' which are ', class_names)
    # print('-------------------------------------------------------------------')

    # Checking the number of classes
    if classes <= 1:
        write_logs(file_log, controler='Torch',  action='start_classification_training', profile_id = profile_id, project_name=project_name, message= 'Number of classes should be greather than 1')
        TriggerWebHook(profile_id, project_name, message='Number of classes should be greather than 1', status='FAILED', URL=URL)
        raise Exception("Number of classes should be greather than 1")
        sys.exit('Number of classes should be greater 1. Got ' + str(classes))

    
    if project_type != 'classification':
        write_logs(file_log, controler='Torch',  action='start_classification_training', profile_id = profile_id, project_name=project_name, message= 'Shoule be classifocation at this time')
        TriggerWebHook(profile_id, project_name, message='Only classification implemented at this time', status='FAILED', URL=URL)
        raise Exception("Only classification implemented at this time")
        sys.exit(project_name + ' is not yet implemented')

    # We define the model
    if model == 'mobilenet':
        

        model_ft = models.mobilenet_v2()
        model_ft.classifier[1] = torch.nn.Linear(in_features=model_ft.classifier[1].in_features, out_features=classes)
        model_ft.classifier[1] = nn.Sequential(
            model_ft.classifier[1],
            nn.Softmax(),
                    )
       
        criterion = nn.CrossEntropyLoss()

    elif model == 'resnet':
        raise Exception("Resnet not implemented yet")
        model_ft = ""
    else:
        TriggerWebHook(profile_id, project_name, message='Specify the model', status='FAILED', URL=URL)
        raise Exception("Specify mobile model")
        sys.exit('Specify the mobile model')    

    # Sending model to device
    model_ft = model_ft.to(device)

    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)


    # Decay LR by a factor of 0.5 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma, verbose=True)

    
    # We get dataloaders
    # dataloaders, class_names, dataset_sizes = data_loader(IMG_SIZE_X, IMG_SIZE_Y, directory_dataset)

    dataloaders, class_names, dataset_sizes = data_loader(directory_dataset, img_size, img_size, batch_size, channel, key=key, initialization_vector=initialization_vector)
    # We train the model

    model_ft = train_model(profile_id, project_name, model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=num_epochs, log_file=log_file)
    
    write_logs(file_log, controler='Torch',  action='start_classification_training', profile_id = profile_id, project_name=project_name, message= 'Training itslef finished in train_model')
    
#.....................................................................................................
#.....................................................................................................
#................................................ Saving models ......................................
#.....................................................................................................
#.....................................................................................................
    write_logs(file_log, controler='Torch',  action='start_classification_training', profile_id = profile_id, project_name=project_name, message= 'Exporting models')
    
    # Possible classes of the model
    possible_classes = str(class_names[0])
    for i in range(1, len(class_names)):
        possible_classes = possible_classes + class_names[i]    
    
    
    for export_type in export:
        if export_type == "torch":
            # Saving Torch model

            PATH_SAVE_MODEL = CURRENT_DIR + '/models/' + export_type + '/' + model_name_torch
            torch.save(model_ft, PATH_SAVE_MODEL)
            
            write_logs(file_log, controler='Torch',  action='start_classification_training', profile_id = profile_id, project_name=project_name, message= 'Export done for torch')
                      
        if export_type == "coreml":
            # Saving Frozen model
            PATH_SAVE_MODEL = CURRENT_DIR + '/models/' + export_type + '/' + model_name_coreml
            
            example_input = torch.rand(1, channel, img_size, img_size) # after test, will get 'size mismatch' error message with size 256x256
            
            example_input = example_input.cuda()
            
            model_ft = model_ft.to(device)
            
            traced_model = torch.jit.trace(model_ft, example_input)
            

            for i, label in enumerate(class_names):
                if isinstance(label, bytes):
                    class_names[i] = label.decode("utf8")
                 
            classifier_config = ct.ClassifierConfig(class_names)
            ctModel = ct.convert(traced_model.cpu(), inputs=[ct.ImageType(name="input_image", shape=example_input.shape)], output_names = ['Prediction'], classifier_config=classifier_config)

            ctModel.author ="SightCall"
            ctModel.license="Commercial"
            ctModel.short_description="Model to predict the class of a photo. Possible classes are " + str(possible_classes)
            ctModel.input_description['input_image']="Array(int8) of shape [1, " + str(channel) + ", " + str(img_size) + ", " + str(img_size) + "] (channel first)"
            ctModel.output_description['655']=str(['Prob(' + str(i) + ')' for i in class_names])
            ctModel.save(PATH_SAVE_MODEL)

            
            # Correcting Features in the Coreml file


            new_coreml_model = ct.models.MLModel(PATH_SAVE_MODEL)
            spec = new_coreml_model.get_spec()
            ct.utils.rename_feature(spec, '655', 'quality_prediction')
            new_coreml_model = ct.models.MLModel(spec)
            new_coreml_model.save(PATH_SAVE_MODEL)
            
            write_logs(file_log, controler='Torch',  action='start_classification_training', profile_id = profile_id, project_name=project_name, message= 'Export done for coreml')
      
        if export_type == "onnx":

            PATH_SAVE_MODEL_ONNX = CURRENT_DIR + '/models/' + export_type + '/' + model_name_onnx
            dummy_input = torch.randn(1, channel, img_size, img_size).to("cpu")
            model_ft = model_ft.to("cpu")
            torch.onnx.export(model_ft, dummy_input, PATH_SAVE_MODEL_ONNX, verbose=False, input_names=['Image'], output_names=['Predictions'])
                    

            # Metadata
            
            # description of the model

            
            description = "ONNX file to classify an image. Possible classes are " + str(possible_classes) + "."
            
            # Output
            output_model = ['Prob(' + str(i) + ')' for i in class_names]
            
            # Input
            input_model = "Array(int8) of shape [1, " + str(channel) + ", " + str(img_size) + ", " + str(img_size) + "] (channel first)"
            
            # metadata
            metadata = {"input": input_model, "output": output_model, "classes":class_names, "description": description}
            
            # version and author
            version = 0
            author = 'SightCall'

            # Writing metadata
            modelonnx = onnx.load(PATH_SAVE_MODEL_ONNX)
            modelonnx.model_version = version
            modelonnx.doc_string = str(metadata)
            modelonnx.producer_name = author

            onnx.save(modelonnx, PATH_SAVE_MODEL_ONNX)
            
                     
            # Writing Logs when it is done
            write_logs(file_log, controler='Torch',  action='start_classification_training', profile_id = profile_id, project_name=project_name, message= 'Export done for onnx')

        if export_type == "pytorchmobile":
            
            # print("pytorch mobile export")

            PATH_SAVE_MODEL_PYTORCHMOBILE = CURRENT_DIR + '/models/' + export_type + '/' + model_name_pytorchmobile
            dummy_input = torch.rand(1, channel, img_size, img_size).to("cpu")
            model_ft = model_ft.to("cpu")
            traced_script_module = torch.jit.trace(model_ft, dummy_input)
            traced_script_module_optimized = optimize_for_mobile(traced_script_module)
            traced_script_module_optimized._save_for_lite_interpreter(PATH_SAVE_MODEL_PYTORCHMOBILE)   
            
            write_logs(file_log, controler='Torch',  action='start_classification_training', profile_id = profile_id, project_name=project_name, message= 'Export done for pytorchmobile')
            
   


        # if export_type == "tflite":
            
        #     import tensorflow as tf
    
        #     print('***************   TFLITE   *********************')
        #     PATH_SAVE_MODEL = CURRENT_DIR + '/models/' + export_type + '/' + model_name_tflite
            
        #     # Creating a folder for frozen export
        #     PATH_SAVE_MODEL_Frozen = CURRENT_DIR + '/models/' + 'frozen' 
        #     if not os.path.exists((PATH_SAVE_MODEL_Frozen)):
        #        os.mkdir(PATH_SAVE_MODEL_Frozen)
               
        #     print("Creating frozen directory")
        #     PATH_SAVE_MODEL_Frozen = CURRENT_DIR + '/models/' + 'frozen' +  '/' + model_name_frozen
        #     # Generating frozen graph
        #     os.system('onnx-tf convert -i' + PATH_SAVE_MODEL_ONNX + ' -o ' + PATH_SAVE_MODEL_Frozen)
        #     print("Frozen graph generated properly")


        #     # Generating tflite
        #     converter = tf.lite.TFLiteConverter.from_saved_model(PATH_SAVE_MODEL_Frozen)
        #     converter.allow_custom_ops = True 


        #     converter.target_spec.supported_ops = [
        #                 tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        #                 tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        #                 ]
        #     tflite_model = converter.convert()
            
        #     with open(PATH_SAVE_MODEL, 'wb') as f:
        #         f.write(tflite_model)

        #     print('*********************************************')
        #     print('*********************************************')
        #     print('********* TFLITE saved at ' + PATH_SAVE_MODEL)
        #     print('*********************************************')
        #     print('*********************************************')


        
    # print('*********************************************')
    # print('*********************************************')
    # print('*********************************************')
    # print('***** Updating with Finish Status************')
    on_train_end(profile_id, project_name, URL)
    # print('**** Updated with Finish Status !!!!!********')
    # print('*********************************************')
    # print('*********************************************')
    # print('*********************************************')
    # print('*********************************************')
    

def start_detection_training(profile_id, project_name, URL = '',
    project_type = "classification", 
    directory_dataset = "Front",
    model='mobilenet',
    lr=0.005,
    batch_size=24,
    num_epochs=2,
    img_size = 224,
    channel=3,
    gamma = 0.5,
    step_size = 10,
    export = ["coreml"],
    Data_Crypted = False,
    krow = 1,
    kcol = 1,
    ITER_MAX = 1):
    
    # start detection training here
    
    # complete later
    return 0
import numpy as np
import torch
from torchvision import datasets, transforms
import os
import copy
import time


from .helper_general import write_logs, update_database
from .helper_encryption import decryption_file, decryption_detection_annotations
from trainer.__init__ import file_log




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from PIL import Image
import torch
import torch.utils.data

import json
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# from .detection import transforms as T
import io


def build_detection_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    
    model_name = 'fasterrcnn_resnet50_fpn'
    model = getattr(torchvision.models.detection, model_name)(pretrained=True)
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def data_transformation_detection(train):
    # transforms = []
    # # converts the image, a PIL image, into a PyTorch Tensor
    # transforms.append(T.ToTensor())
    # if train:
    #     # during training, randomly flip the training images
    #     # and ground-truth for data augmentation
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    return 0






class EncryptedDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset=None, channel=3, key=None, initialization_vector=None, name='name', transformations=None):
        self.dataset = dataset
        self.key = key
        self.initialization_vector = initialization_vector 
        self.transformations = transformations
        self.annotations_file = os.path.join(dataset, "annotations.json")
        self.channel = channel

        
        
        # decrypt annotation file
        decryption_detection_annotations(self.annotations_file, self.key, self.initialization_vector)
        
        annotations1 = json.load(open(self.annotations_file))
        self.annotations = list(annotations1.values())  # don't need the dict keys
        self.CLASS_NAMES = ['__background__']
        
        # Getting class names  
        for k in range(len(self.annotations)):
            for i in range(len(self.annotations[k]["regions"])):
                name = self.annotations[k]["regions"][i]["region_attributes"]["name"]
                if name not in self.CLASS_NAMES:
                    self.CLASS_NAMES.append(name)
        
    
    def __getitem__(self, idx):

        # load images ad masks
        img_name = self.annotations[idx]["filename"]
        img_path = os.path.join(self.dataset, img_name)  
        
        # Decryption
   
        img_data = decryption_file(img_path, self.key, self.initialization_vector)
        
        if self.channel == 3:
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            
        elif self.channel == 1:
            img = Image.open(io.BytesIO(img_data)).convert('L')
        
        else:
            raise Exception("Undefined channel number")        
        

        # first id is the background, objects count from 1
        obj_ids = np.array(range(len(self.annotations[idx]["regions"]))) +1
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        
        if num_objs == 0:
            raise Exception("No bounding boxes for image " + str(img_name
                                                                 ))
            
        boxes = []
        label_list = []
        # if num_objs > 0:
        for i in range(num_objs):
            x = np.min(self.annotations[idx]["regions"][i]["shape_attributes"]["x"])
            y = np.max(self.annotations[idx]["regions"][i]["shape_attributes"]["y"])
            width = np.min(self.annotations[idx]["regions"][i]["shape_attributes"]["width"])
            height = np.max(self.annotations[idx]["regions"][i]["shape_attributes"]["height"])
            
            object_name = self.annotations[idx]["regions"][i]["region_attributes"]["name"]
            
            if object_name not in self.CLASS_NAMES:
                raise Exception(' Unknow class name ' + str(object_name))
                
            label_list.append(self.CLASS_NAMES.index(object_name))
            
            xmin = x
            ymin = y
            xmax = xmin + width
            ymax = ymin + height
            
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # there is only one class
        labels = torch.as_tensor(label_list, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        # print(label_list)

        if self.transformations is not None:
            # print(img)
            img, target = self.transformations(img, target)
            img, target = img, target

        return img, target

    def __len__(self):
        return len(self.annotations)
    
    




def on_train_begin(profile_id, project_name, log_file):
    
    write_logs(file_log, controler='Torch', action='on_train_begin', profile_id=profile_id, project_name=project_name, message='starting on_train_begin')
    
    
    '''
    
    This function is going to be run at the begining of each epoch
    
    INPUTS:
        
        * task_id: Task of the current job
    
        * log_file: File to store stats
    
    RETURN:
        Nothing
    
    '''
    
    # We write the first line of the stat file
    line_to_write = "Status" + ',' + "Epoch"  + "," + "Loss" + "," + "Accuracy" + '\n'
    f = open(log_file, "a")
    f.write(line_to_write)
    f.close()
    
    update_database(profile_id=profile_id, project_name=project_name, status='RUNNING', message='Starting training', current_epoch=None, loss=None, accuracy=None, epoch_duration=None)
    


def on_epoch_end(profile_id, project_name, current_epoch, num_epochs, log_file, loss, accuracy, epoch_duration):
    
    '''
    
    This function is going to be run at the end of each epoch
    
    INPUTS:
        * task_id: Task of the current job
    
        * current_epoch: Epoch we are processing
    
        * last_epoch: Total number of epochs
    
        * log_file: File to store stats
    
        * loss: loss at the current epoch
    
        * accuracy: Accuracy at the current epochs
    
    RETURN:
        Nothing
    
    '''
        
    # print('Current epoch ' + str(current_epoch))
    
    status = "RUNNING"
    if current_epoch == num_epochs-1:
        status = "EXPORTING"
    elif current_epoch == 0:
        line_to_write = "Status" + ',' + "Epoch"  + "," + "Loss" + "," + "Accuracy" + '\n'
    
    line_to_write = status + ',' + str(current_epoch)  + "," + str(loss) + "," + str(accuracy) + '\n'
    
    f = open(log_file, "a")
    f.write(line_to_write)
    f.close()
                 
    update_database(profile_id=profile_id, project_name=project_name, status=status, message='Updating with stats at epochs ' + str(current_epoch + 1), current_epoch=current_epoch, loss=loss, accuracy=accuracy, epoch_duration=epoch_duration)


def on_train_end(profile_id, project_name, URL):
    
    '''
    
    This function is going to be run at the end of the training
    
    INPUTS:
        * task_id: Task of the current job
    

    RETURN:
        Nothing
    
    '''
        
    write_logs(file_log, controler='Torch', action='on_train_end', profile_id=profile_id, project_name=project_name, message= 'starring on_train_end')
    
    # stored_trainings = Training.objects.get(profile_id=profile_id, project_name=project_name)
            
            
    # try:
                
    #     if stored_trainings != None:
    #         # print('store_training not None')
    #         # loss, accuracy = logs["loss"], logs["accuracy"]
    #         stored_trainings.status = "SUCCESS"
    #         stored_trainings.lastupdate = datetime.datetime.now() #date.today()
    #         # print('Database updated with SUCCESS ...')
    #         # print('stored_trainings.project_name = ', stored_trainings.project_name)
    #         stored_trainings.save()
            
    #         write_logs(file_log, controler='Torch', action='on_train_end', profile_id=profile_id, project_name=project_name, message= 'on_train_end finished')
    # except Exception as e:
    #     # print('********************************************************************')
    #     # print('Failed with store training update ....')
    #     # print(e)
    #     # print('********************************************************************')
    #     write_logs(file_log, controler='Torch', action='on_train_end', profile_id=profile_id, project_name=project_name, message= str(e))
    #     TriggerWebHook(profile_id, project_name, message='Error at the end of training' + str(e), status='FAILED', URL=URL)
    #     sys.exit("Failed with store training update............................................")





import torch




def do_script(model, in_size=100):
    model_script = torch.jit.script(model)
    model_script.eval()
    return model_script


def do_trace(model, in_size=100, device=device):
    inp = torch.rand(1, 3, in_size, in_size)
    inp_ = inp.to(device)
    model_ = model.to(device)
    model_trace = torch.jit.trace(model_, inp_)
    model_trace.eval()
    return model_trace


def dict_to_tuple(out_dict):
    if "masks" in out_dict.keys():
        return (out_dict["boxes"], out_dict["scores"], out_dict["labels"], out_dict["masks"])
    return (out_dict["boxes"], out_dict["scores"], out_dict["labels"])


class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return dict_to_tuple(out[0])


def save_jit_model(model, name, img_size=224, script=False, device=device):
    
    X = True

    if X:
        
        model = model.to(device)

        model.eval()
        # in_size = 100
        inp = torch.rand(1, 3, img_size, img_size)
        
        inp = inp.to(device)

        with torch.no_grad():
            out = model(inp)

            if script:
                out = dict_to_tuple(out[0])
                script_module = do_script(model, img_size)
                script_out = script_module([inp[0]])[1]
                script_out = dict_to_tuple(script_out[0])
            else:
                script_module = do_trace(model, img_size, device=device)
                script_out = script_module(inp)

            assert len(out[0]) > 0 and len(script_out[0]) > 0

            # compare bbox coord
            print(np.max(np.abs(out[0].numpy() - script_out[0].numpy())))

            torch._C._jit_pass_inline(script_module.graph)
            torch.jit.save(script_module, name)
            print(f"{name} is tha path where model is saved")
            if os.path.exists(name):
                print('Name exists')
            else:
                print('Unable to save')
                
            return script_module
        


def _convert_slice_v9(builder, node, graph, err):
    '''
    convert to CoreML Slice Static Layer:
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#L5082
    ''' 
    
    INT_MAX = 2 ** 30
    data_shape = graph.shape_dict[node.inputs[0]]
    len_of_data = len(data_shape)
    begin_masks = [True] * len_of_data
    end_masks = [True] * len_of_data

    default_axes = list(range(len_of_data))
    default_steps = [1] * len_of_data
    
    ip_starts = node.attrs.get('starts')
    ip_ends = node.attrs.get('ends')
    axes = node.attrs.get('axes', default_axes)
    steps = node.attrs.get('steps', default_steps)

    starts = [0] * len_of_data
    ends = [0] * len_of_data

    for i in range(len(axes)):
        current_axes = axes[i]
        starts[current_axes] = ip_starts[i]
        ends[current_axes] = ip_ends[i]
        if ends[current_axes] != INT_MAX or ends[current_axes] < data_shape[current_axes]:
            end_masks[current_axes] = False

        if starts[current_axes] != 0:
            begin_masks[current_axes] = False

    builder.add_slice_static(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        begin_ids=starts,
        end_ids=ends,
        strides=steps,
        begin_masks=begin_masks,
        end_masks=end_masks
    )











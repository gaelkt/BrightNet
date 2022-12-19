import numpy as np
import torch
from torchvision import datasets, transforms
import os
import copy
import time


from trainer.__init__ import num_workers

from .helper_general import write_logs, update_database
from .helper_encryption import decryption_file
from trainer.__init__ import file_log




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import Dataset, DataLoader
from PIL import Image


import io


dataset_path = 'D:\\Data/WebCelery/DatasetsWebCelery/TestAug'

def data_transformation(image_size_x, image_size_y, normalization_ratio=1.0/255.0):
    
    '''
    Transform input images to a tensor of size IMG_SIZE_X x IMG_SIZE_Y
    '''
    
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((image_size_x,image_size_y),interpolation=Image.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [normalization_ratio, normalization_ratio, normalization_ratio]),
    ]),
    'val': transforms.Compose([
        transforms.Resize((image_size_x,image_size_y),interpolation=Image.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [normalization_ratio, normalization_ratio, normalization_ratio]),
    ]),
}
    
    return data_transforms

class EncryptedDataset(Dataset):
    def __init__(self, dataset=None, channel=3, key=None, initialization_vector=None, transforms=None):
        
        self.key = key
        self.initialization_vector = initialization_vector 
        self.imgs_path = dataset
        self.transforms = transforms
        self.channel = channel
        
        class_name = os.listdir(dataset)
        
        self.classes = class_name
        
        
        
        self.class_map = {class_name[i] : i for i in range(len(class_name))}
        
        self.data = []
        
        for classe in class_name:
            for file in os.listdir(dataset + '/' + classe):
                img_path = dataset + '/' + classe + '/' + file
                self.data.append([img_path, classe])
                
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        
        img_data = decryption_file(img_path, self.key, self.initialization_vector)
        
        if self.channel == 3:
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            
        elif self.channel == 1:
            img = Image.open(io.BytesIO(img_data)).convert('L')
        
        else:
            raise Exception("Undefined channel number")
            
        if self.transforms is not None:
            img = self.transforms(img)
        
        class_id = self.class_map[class_name]
        
        return img, class_id
    
    
def data_loader(directory, image_size_x, image_size_y, batch_size, channel, key=None, initialization_vector=None):
# def data_loader(IMG_SIZE_X, IMG_SIZE_Y, directory):

    '''
    This function is used to load the data from a directory

    Inputs:
        - image_size_x, image_size_y: Size of input image
        - directory: Directory where images are stored with a subfolder per class and a folder for train/val
    
    RETURN:
        - dataloaders
        - class_names
        - dataset_sizes
    

    '''
    
    # data transformation and preprocessing
    data_transforms = data_transformation(image_size_x, image_size_y, normalization_ratio=1.0/255.0)

    
    # dataset
    image_datasets = {x: EncryptedDataset(os.path.join(directory, x), channel, key, initialization_vector, data_transforms[x]) for x in ['train','val']}

    
    
    # We load the data
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)
              for x in ['train','val']}
    
    
    
    # Size of data
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val']}
    
    # class names
    class_names = image_datasets['train'].classes
    
    return dataloaders, class_names, dataset_sizes



def train_model(profile_id, project_name, model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs, log_file):
    
    write_logs(file_log, controler='Torch', action='train_model', profile_id=profile_id, project_name=project_name, message='starting training')
    


    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # At the begining of train
    on_train_begin(profile_id, project_name, log_file)
    
    print('********** DEVICE IN TRAIN MODEL *************')
    
    print(device)

    for epoch in range(num_epochs):
        since = time.time()
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:

                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #     phase, epoch_loss, epoch_acc))
            
            
            # Duration
            time_elapsed = np.round((time.time() - since)/60.0, 2)
            #? At the end of each epoch
            if phase == 'train':
                on_epoch_end(profile_id, project_name, current_epoch = epoch, num_epochs=num_epochs, log_file=log_file, loss=float(epoch_loss), accuracy=float(epoch_acc), epoch_duration=time_elapsed)
                # print('**************************')
                # print('*****Update Epoch in the DB *******')
                # print('**************************')
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # print()

    
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model




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

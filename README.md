# AI Trainer SAIT

## VM
- The aitrainer is located at https://40.113.14.226:8000


## Datasets 
The dataset should have the following structure：
```Shell
  dataset/
    train/
      class 1/
      class 2/
      .../
      class N//
    val/
      class 1/
      class 2/
      .../
      class N/

```

or 

```Shell
  dataset/
    class 1/

    class 2/
    
    ...


```
**train** is the folder for training and **val** is the validation folder. We must have a subfolder for each class. For example, we have a dataset having two classes: **Bad** and **Good**, representing the quality of a teeth photo.



## EndPoint Check Server

- -- **Address and Port** : https://40.113.14.226:8000/checkserver/
- -- **Header** : Authorization token


- Returns JSON:
- -- **result**: "Failed" or "Ok"
- -- **message**: Custom message


## EndPoint Start Training: 


- -- **URL and Port** : 40.113.14.226:8000
- -- **Header** : Authorization token

- JSON parameters in the body:
- -- **profile_id**  (str, mandatory): The Idprofile of the user. An error will be returned if a previous one exists

- -- **project_name**  (str, mandatory): The project name and it has to be unique. An error will be returned if a previous one exists

- -- **webhook_url**  (str, mandatory): The URL for the webhook. If nothing is provided, the default URL in dev/qualif settings will be used

- -- **epochs**  (int, optional): It is an integer representing the number of training cycles (>1 and < 100).  In one epoch, we use all of the dataset exactly once. The default value is 10.

- -- **learning_rate** (float, optional): The learning rate is a hyper-parameter that controls the weights of our neural network with respect to the loss gradient. It defines how quickly the neural network updates the concepts it has learned. A desirable learning rate is low enough that the network converges to something useful, but high enough that it can be trained in a reasonable amount of time. The learning rate is between 0 and 1. The default value is 0.01.

- -- **image_size** (int, optional): Image size we want to use for the model. Default value is 224. lel tasks. Maximum allowed value is 768.

- -- **batch_size** (int, optional): Number of images processed before the model is updated. The default value is 36. The maximum value allowed for the dev VM is 36. A high batch will train faster but will require more memory. 

- -- **step_size** (int, optional): Period of learning rate decay. Default value is 10. It means that, after 10 epochs the learning rate willl be decreased.

- -- **gamma** (float, optional): Multiplicative factor of learning rate decay after step_size epochs. The value of gamma is between 0 and 1. The default value is 0.1

- -- **export**: A list stating the desired exports. Defaut lis is ["onnx", "coreml"]. Possible export are "coreml", "onnx", "pytorchmobile", "coreml" and "torch".  

- -- **model_type**: (str, mandatory): 'classification' or 'detection'.

- -- **storage_type** (string, mandatory): The type of cloud storage for the dataset. Must be "aws" or "azure"

- -- **container** (string, mandatory): The name of container for the dataset

- -- **folder** (string, mandatory): The folder name in the dataset container

- -- **zipped** (bool, mandatory): True if the dataset is a zip file

- -- **azure_connection_string** (string, mandatory): If storagetype is azure, the connection string

- -- **aws_access_key_id** (string, mandatory): If storagetype is AWS, the access key id

- -- **aws_secret_access_key** (string, mandatory): If storagetype is AWS, the secret access key






Result is a JSON with the following parameters:

- -- **status**  "FAILED" or "PENDING" (The query was correct or not)
- -- **project_name**: The name of the project
- -- **profile_id**: The name of the project
- -- **dataset** :  The dataset used to train the model
- -- **message** : A custom message.






## EndPoint Get Training Status

- -- **URL and Port** : https://40.113.14.226:8000/statustraining/profile_id/project_name
- -- **profile_id** : profile_id in the url
- -- **project_name** : Project name in the url
- -- **Header** : Authorization token
An error will be returned if several training or no training exist with same Idprofile and projectname. We should have only one. 
- Returns:
- -- **result**  of training: PENDING, DOWNLOADIND, DOWNLOADED, RUNNING, EXPORTING, SUCCESS or FAILED
- -- **current_epoch** (int): Current epoch 
- -- **epochs** (int): Total number of epochs to train
- -- **loss** (float): Current error on the training set of data. When the loss is decreasing actively, it means that the model is getting better and better. When the loss is not decreasing anymore, it means that training is stagnating.
- -- **accuracy** (float):  Acuracy
- -- **lastupdate** (time): Last time the model was updated
- -- **last_epoch_duration_seconds** (float):  Duration of the last epoch in seconds



### Some Notes

import os, platform
from django.conf import settings
from Crypto.Random import get_random_bytes

# Paths for the datasets, projects and tensorboards
nameapi = 'aitrainer_package'
current = os.path.abspath(os.getcwd())
parent, child = os.path.split(current)
AllMyProjects = parent + '/' + nameapi + '/Projects/'
AllDatasets = parent + '/' + nameapi + '/Datasets/'
AllTensorboard = parent + '/' + nameapi + '/Tensorboard/'
zipped_directory = parent + '/' + nameapi + '/ZipFolder/'


if not os.path.exists(AllMyProjects):
    os.makedirs(AllMyProjects)

if not os.path.exists(AllDatasets):
    os.makedirs(AllDatasets)

if not os.path.exists(AllTensorboard):
    os.makedirs(AllTensorboard)

if not os.path.exists(zipped_directory):
    os.makedirs(zipped_directory)


# file for logging
file_log = getattr(settings, "FILE_LOG", None)
if file_log == "" or file_log == r'':
    file_log = parent + '/' + nameapi + '/aitrainer.log'

# Variable for the project

central_conn_str_azure = getattr(settings, "CENTRAL_CONN_STR_AZURE", None)
central_container_azure = getattr(settings, "CENTRAL_AZURE_CONTAINER", None)


# Authorization token
authorization_token = getattr(settings, "AUTHORIZATION", None)

# Default training parameters
default_num_epochs = getattr(settings, "DEFAULT_NUM_EPOCHS", None)
default_batch_size = getattr(settings, "DEFAULT_BATCH_SIZE", None)
default_learning_rate = getattr(settings, "DEFAULT_LEARNING_RATE", None)
default_step_size = getattr(settings, "DEFAULT_STEP_SIZE", None)
default_gamma = getattr(settings, "DEFAULT_GAMMA", None)
default_image_size = getattr(settings, "DEFAULT_IMAGE_SIZE", None)
default_channel = getattr(settings, "DEFAULT_CHANNEL", None)
default_data_crypted = getattr(settings, "DEFAULT_DATA_CRYPTED", None)
default_webhook_url = getattr(settings, "DEFAULT_WEBHOOK_URL", None)
default_export = getattr(settings, "DEFAULT_EXPORT", None)


# Parameters range
num_epochs_min = getattr(settings, "NUM_EPOCHS_MIN", None)
num_epochs_max = getattr(settings, "NUM_EPOCHS_MAX", None)
batch_size_min = getattr(settings, "BATCH_SIZE_MIN", None)
batch_size_max = getattr(settings, "BATCH_SIZE_MAX", None)
learning_rate_min = getattr(settings, "LEARNING_RATE_MIN", None)
learning_rate_max = getattr(settings, "LEARNING_RATE_MAX", None)
step_size_min = getattr(settings, "STEP_SIZE_MIN", None)
gamma_min = getattr(settings, "GAMMA_MIN", None)
gamma_max = getattr(settings, "GAMMA_MAX", None)
image_size_min = getattr(settings, "IMAGE_SIZE_MIN", None)
image_size_max = getattr(settings, "IMAGE_SIZE_MAX", None)
all_possible_export = getattr(settings, "ALL_POSSIBLE_EXPORT", None)


#• Number of required parameters
nb_required_parameters = getattr(settings, "NB_REQUIRED_PARAMETERS", None)

# For windows, num_workers should be equal to 0
# https://discuss.pytorch.org/t/errors-when-using-num-workers-0-in-dataloader/97564/4
if platform.system() == 'Windows':
    num_workers = 0
else:
    num_workers = 12
    

# Encryption keys
key = get_random_bytes(32)
initialization_vector = get_random_bytes(16)

# Local directory dataset
local_directory_dataset = getattr(settings, "LOCAL_DIRECTORY_DATASET", None)
    
    

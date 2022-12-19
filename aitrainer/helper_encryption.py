from Crypto.Cipher import AES

import os

from PIL import Image
import io


def encryption_file(file_to_encrypt_path, key, initialization_vector):
    
    # Reading input data
    file_to_encrypt = open(file_to_encrypt_path, 'rb')
    data = file_to_encrypt.read()
    file_to_encrypt.close()
    
    
    # Encrypting input data
    cfb_cipher = AES.new(key, AES.MODE_CFB, initialization_vector)
    data = cfb_cipher.encrypt(data)

    # Saving encrypted data
    file_to_encrypt = open(file_to_encrypt_path, 'wb')
    file_to_encrypt.write(data)
    file_to_encrypt.close()

    return 0

def decryption_file(file_to_decrypt_path, key, initialization_vector):
    
    
    # Reading encrypted data
    file_to_decrypt = open(file_to_decrypt_path, 'rb')
    data = file_to_decrypt.read()
    file_to_decrypt.close()
    
    # Decrypting data
    cfb_decipher = AES.new(key, AES.MODE_CFB, initialization_vector)
    data = cfb_decipher.decrypt(data)
    
    return data



def decryption_detection_annotations(file_to_decrypt_path, key, iv):
    
    ''' 
    Decrypt json annotations
    
    '''
    # Reading encrypted data
    file_to_decrypt = open(file_to_decrypt_path, 'rb')
    data = file_to_decrypt.read()
    file_to_decrypt.close()
    
    # Decrypting data
    cfb_decipher = AES.new(key, AES.MODE_CFB, iv)
    data = cfb_decipher.decrypt(data)

    # Writing decrypted data
    file_to_decrypt = open(file_to_decrypt_path, 'wb')
    file_to_decrypt.write(data)
    file_to_decrypt.close()

    return 0


def encrypting_folder(folder_path, key, initialization_vector):
    
    for file in os.listdir((folder_path)):
        
        encryption_file(folder_path + '/' + file, key, initialization_vector)
        
        
def decrypting_folder(folder_path, key, initialization_vector):
    
    for file in os.listdir((folder_path)):
        
        decryption_file(folder_path + '/' + file, key, initialization_vector)   
        
        
def encrypt_classification_dataset(dataset, key, initialization_vector):
    '''
    Encrypt entire classification dataset.
    Dataset should have a train and a val folder

    Inputs:
        datset: orignal dataset
        output: new encrypted dataset 
    
    '''
    
    for mode in ['train', 'val']:
        
        for classe in os.listdir(dataset + '/' + mode):
            
            encrypting_folder(dataset + '/' + mode + '/' + classe, key, initialization_vector)
    
            
def encrypt_detection_dataset(dataset, key, initialization_vector):
    '''
    Encrypt entire detection dataset.
    Dataset should have a train and a val folder

    Inputs:
        datset: orignal dataset
        output: new encrypted dataset 
    
    '''
    
    for mode in ['train', 'val']:

        encrypting_folder(dataset + '/' + mode, key, initialization_vector)
#essentials 
import os
import json
from glob import glob
#ds
from PIL import Image
import pandas as pd 
import numpy as np



DATA_DIR = "/media/data/chitb/study_zone/ML-_midterm_20212/data"
INTERIM_DIR = os.path.join(DATA_DIR, "interim")

def folder_to_class(dir=''):
    #dict for store classes:
    class_dict = {}
    #indexing 
    idx = 0
    for folder in list(os.listdir(dir)):
        
        #dir to images
        curr_dir = os.path.join(dir, folder)
        #for external
        if folder == "anger":
            folder = "angry"
        elif folder == "contempt":
            folder = "neutral"
        elif folder == "sadness":
            folder = "sad"
        for image in glob(os.path.join(curr_dir, "*.png")):
            #img dir
            img_dir = os.path.join(curr_dir, image)
            img = Image.open(img_dir)
            #convert to numpy and store
            class_dict[idx] = [np.asarray(img).tolist(), folder]
            img.close()
            idx += 1 
        #for raw
        for image in glob(os.path.join(curr_dir, "*.jpg")):
            #img dir
            img_dir = os.path.join(curr_dir, image)
            img = Image.open(img_dir)
            
            #convert to numpy and store
            class_dict[idx] = [np.asarray(img).tolist(), folder]
            img.close()
            idx += 1 
        #next
        
    return class_dict

def iterim_preprocess(process_dir='external'):
    #get the train and test, or only data for external
    data_dir = os.path.join(DATA_DIR, process_dir)
    #data_dict
    data_dict = {}
    if process_dir == "raw":

        train_dir, test_dir = os.listdir(os.path.join(DATA_DIR, process_dir))
        train_dir, test_dir = os.path.join(data_dir, train_dir), os.path.join(data_dir, test_dir)
        train_dict, test_dict = folder_to_class(train_dir), folder_to_class(test_dir)
        return train_dict, test_dict
        # for key in train_dict.keys():
        #     data_dict[key] = []
        #     data_dict[key].extend(train_dict[key])
        #     data_dict[key].extend(test_dict[key])
    else:
        ex_dir = os.listdir(os.path.join(DATA_DIR, process_dir))[0]
        ex_dir = os.path.join(data_dir, ex_dir)
        data_dict = folder_to_class(ex_dir)
        return data_dict
    

if __name__ == '__main__':
    print(os.listdir(DATA_DIR))
    train_dict, test_dict = iterim_preprocess(process_dir='raw')
    external_dict = iterim_preprocess(process_dir='external/ck')
    print("TRAIN")
    for key, item in train_dict.items():
        
        print(key, len(item))
        break
    print("EXTERNAL")
    for key, item in external_dict.items():
        
        print(key, len(item))  
        break  

    # #TEMP STORAGE
    # with open(os.path.join(INTERIM_DIR, 'train.json'), 'w') as f:
    #     json.dump(train_dict, f, indent=3)
    # with open(os.path.join(INTERIM_DIR, 'test.json'), 'w') as f:
    #     json.dump(test_dict, f, indent=3)
    # with open(os.path.join(INTERIM_DIR, 'external.json'), 'w') as f:
    #     json.dump(external_dict, f, indent=3)
    #orient for setting key as index

    train_df = pd.DataFrame.from_dict(train_dict, orient='index', columns=['pixel', 'label']).reset_index(drop=True)
    train_df.to_feather(os.path.join(INTERIM_DIR, 'train.ftr'))
    test_df = pd.DataFrame.from_dict(test_dict, orient='index', columns=['pixel', 'label']).reset_index(drop=True)
    test_df.to_feather(os.path.join(INTERIM_DIR, 'test.ftr'))
    external_df = pd.DataFrame.from_dict(external_dict, orient='index', columns=['pixel', 'label']).reset_index(drop=True)
    external_df.to_feather(os.path.join(INTERIM_DIR, 'external.ftr'))

    #check the len
    print("TRAIN", len(train_df))
    print("EXTERNAL", len(external_df))
    print("TEST", len(test_df))
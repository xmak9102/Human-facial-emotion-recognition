#essentials
import json
import os
import shutil
import pandas as pd
#ds
import numpy as np 
from sklearn.model_selection import train_test_split 

DATA_DIR = "/media/data/chitb/study_zone/ML-_midterm_20212/data"
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
label2class = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
def checkNan(df):
    for i in range(len(df)):
        if np.array(df['pixel'][i].tolist()).reshape(-1).sum() == 0:
            df.drop(i, inplace=True)
            print(f"DROP {i}")
    df.reset_index(inplace=True, drop=True)
    return df
if __name__ == "__main__":
    #df to feather
    train_df = pd.read_feather(os.path.join(INTERIM_DIR, "train.ftr"))
    external_df = pd.read_feather(os.path.join(INTERIM_DIR, "external.ftr"))
    test_df = pd.read_feather(os.path.join(INTERIM_DIR, "test.ftr"))
    raw_df = train_df
    concat_df = pd.concat([train_df, external_df], axis=0).reset_index(drop=True)
    print(concat_df['label'].unique())
    #map label to class 
    concat_df['label'] = concat_df['label'].map(label2class)
    test_df['label'] = test_df['label'].map(label2class)
    raw_df['label'] = raw_df['label'].map(label2class)
    external_df['label'] = external_df['label'].map(label2class)
    concat_df = checkNan(concat_df)
    test_df = checkNan(test_df)
    train_df, val_df = train_test_split(concat_df, test_size=0.2, stratify=concat_df['label'])
    train_df.reset_index(inplace=True, drop=True)
    val_df.reset_index(inplace=True, drop=True)


    # final_data
    #train+val, train, val, test from fer and ck, raw from fer, external from ck 
    concat_df.to_feather(os.path.join(INTERIM_DIR, "data.ftr"))
    train_df.to_feather(os.path.join(INTERIM_DIR, "train.ftr"))
    val_df.to_feather(os.path.join(INTERIM_DIR, "val.ftr"))
    test_df.to_feather(os.path.join(INTERIM_DIR, "test.ftr"))
    raw_df.to_feather(os.path.join(INTERIM_DIR, "raw.ftr"))
    external_df.to_feather(os.path.join(INTERIM_DIR, "external.ftr"))
    #move to finally processed
    shutil.copy((os.path.join(INTERIM_DIR, "data.ftr")), PROCESSED_DIR)
    shutil.copy((os.path.join(INTERIM_DIR, "train.ftr")), PROCESSED_DIR)
    shutil.copy((os.path.join(INTERIM_DIR, "val.ftr")), PROCESSED_DIR)
    shutil.copy((os.path.join(INTERIM_DIR, "test.ftr")), PROCESSED_DIR)
    shutil.copy((os.path.join(INTERIM_DIR, "raw.ftr")), PROCESSED_DIR)
    shutil.copy((os.path.join(INTERIM_DIR, "external.ftr")), PROCESSED_DIR)
    print(len(concat_df))
import albumentations as A
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def augmentData(data, augment_ratio=4):
    tf  = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2)])
    augmentDict = {'pixel': [], 'label': []}

    for i in range(len(data)):
        for j in range(augment_ratio):
            tf_image = tf(image=np.array(data['pixel'][i].tolist()).reshape(48, 48).astype(np.uint8))['image']
            augmentDict['pixel'].append(tf_image.tolist())
            augmentDict['label'].append(data['label'])
    print("label to augment", data['label'])
    return pd.DataFrame.from_dict(augmentDict)


if __name__ == '__main__':
    data = pd.read_feather("/media/data/chitb/study_zone/ML-_midterm_20212/data/processed/data.ftr")
    test = pd.read_feather("/media/data/chitb/study_zone/ML-_midterm_20212/data/processed/test.ftr")
    print("before", data['label'].value_counts())
    print("data", data)
    #label 1 is inferior in terms of num_of_points
    # #augment all
    # for i in range(7):
    #     ratio = 4 if i == 1 else 1
    #     augment_df = data.loc[data['label'] == i, :]
    #     print(len(augment_df))
    #     augment_df.reset_index(inplace=True, drop=True)
    #     augment_part = augmentData(augment_df, augment_ratio=ratio)
    #     data = pd.concat([augment_part, data], axis=0).reset_index(drop=True)
    #augment 1 label
    augment_df = data.loc[data["label"] == 1, :]
    augment_part = augmentData(augment_df)
    augment_df.reset_index(inplace=True, drop=True)
    data = pd.concat([augment_part, data], axis=0).reset_index(drop=True)
    print("after", len(data))
    train, val = train_test_split(data, stratify=data['label'], test_size=0.2)
    train, val = train.reset_index(drop=True), val.reset_index(drop=True)
    train.to_feather("/media/data/chitb/study_zone/ML-_midterm_20212/data/processed/train_augment_all.ftr")
    val.to_feather("/media/data/chitb/study_zone/ML-_midterm_20212/data/val_augment_all.ftr")



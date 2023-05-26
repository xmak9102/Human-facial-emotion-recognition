import pandas as pd

print("INTERIM")
print(pd.read_feather("/media/data/chitb/study_zone/ML-_midterm_20212/data/interim/train.ftr")['label'].value_counts())
print(pd.read_feather("/media/data/chitb/study_zone/ML-_midterm_20212/data/interim/val.ftr")['label'].value_counts())
print(pd.read_feather("/media/data/chitb/study_zone/ML-_midterm_20212/data/interim/data.ftr")['label'].value_counts())

print("PROCESSED")
print(pd.read_feather("/media/data/chitb/study_zone/ML-_midterm_20212/data/processed/train.ftr")['label'].value_counts())
print(pd.read_feather("/media/data/chitb/study_zone/ML-_midterm_20212/data/processed/val.ftr")['label'].value_counts())
print(pd.read_feather("/media/data/chitb/study_zone/ML-_midterm_20212/data/processed/data.ftr")['label'].value_counts())

print("EXTERNAL DATA")
print(len(pd.read_feather("/media/data/chitb/study_zone/ML-_midterm_20212/data/processed/external.ftr")))
print("LEN TRAIN")
print(len(pd.read_feather("/media/data/chitb/study_zone/ML-_midterm_20212/data/processed/train.ftr")))
import os 
from glob import glob 
import sys 
import pandas as pd
import numpy as np
sys.path.append('../../')
sys.path.append('../../../')
from sklearn.model_selection import StratifiedKFold

from data_mapping import id2_dataset_2_description_mapping_more_classes, id2_code_numbers, id2_dataset_2_description_mapping


n_splits = 2
original_dataset_df = pd.read_csv("../../new_pickle_files_with_image_data/pickle_df.csv")
original_dataset_df = original_dataset_df.drop(columns = ['Unnamed: 0','kfold'])

latest_dataset_df = pd.read_csv('../../../New ID2 Data/Use this dump for latest data/mapped_df.csv')

cols = ['path', 'label', 'code', 'nc_group_name','description']


original_dataset_df["label"] = pd.to_numeric(original_dataset_df["label"])
latest_dataset_df["label"] = pd.to_numeric(latest_dataset_df["label"])

def old_dataset_get_new_path(txt):
    """function to get path from given path from the original dataset

    Args:
        txt (string): Original Path (Absolute path of Origin)

    Returns:
        string: Relative Path relative to current directory)
    """
    x = list(txt.split("\\"))
    newlist = x[5:]
    path = os.path.join('..','..','..', *newlist)
    return path

original_dataset_df["path"] = original_dataset_df["path"].apply(old_dataset_get_new_path)

def latest_data_get_new_path(txt):
    """function to get path from given path from the latest dataset

    Args:
        txt (string): Original Path (Absolute path of Origin)

    Returns:
        string: Relative Path relative to current directory)
    """
    x = list(txt.split("\\"))
    newlist = ["New ID2 Data", "Use this dump for latest data", "latest_dump"]
    name  = x[3:][0]
    name, _ = list(name.split("."))
    path = os.path.join('..','..','..', *newlist, name+".pickle")
    return path

latest_dataset_df["path"] = latest_dataset_df["path"].apply(latest_data_get_new_path)

"New ID2 Data/Use this dump for latest data/latest_dump'"

latest_dataset_df.to_csv("latest_dataset_df.csv")


compiled_df = pd.concat([original_dataset_df, latest_dataset_df])
compiled_df = compiled_df.reset_index(drop=True)

print(compiled_df["label"].value_counts())
compiled_df["fold"] = np.nan
skf = StratifiedKFold(n_splits=n_splits)
skf.get_n_splits(compiled_df, compiled_df.label)
for fold, (train_index, test_index) in enumerate(skf.split(compiled_df, compiled_df.label)):
    compiled_df.loc[test_index,"fold"]  = int(fold)

compiled_df["id"] = compiled_df["path"].apply(lambda x :os.path.splitext(os.path.basename(x))[0])

compiled_df = compiled_df.rename(columns={'label': 'description', 'new_label': 'label'})
compiled_df['fold'] = compiled_df['fold'].astype('int')


print(compiled_df)
print(compiled_df.columns)

compiled_df = compiled_df.drop(columns = ['Unnamed: 0',  'file_name', 'rgb_path', 'folder',])
print(compiled_df)
compiled_df.to_csv("compiled_df_server.csv")



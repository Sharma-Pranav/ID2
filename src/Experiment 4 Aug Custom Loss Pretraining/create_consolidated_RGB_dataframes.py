import pickle
import pandas as pd 
import os 
from glob import glob
from tqdm import tqdm
base_path = os.path.join("..","..","..","..","..","isic_datasets", )

isic_2018 = "isic_2018"
isic_2019 = "isic_2019"
isic_2020 = "isic_2020"

df_2018 = pd.read_csv(os.path.join(base_path, isic_2018, "ISIC2018_Task3_Training_GroundTruth.csv"))

def add_path_to_df_2018(x):
    x = os.path.join(base_path, isic_2018, "train", x+".jpg")
    return x
df_2018["path"] = df_2018["image"].apply(add_path_to_df_2018)
df_2018 = df_2018.rename(columns ={ "AKIEC": "AK"})


df_2019 = pd.read_csv(os.path.join(base_path, isic_2019, "ISIC_2019_Training_GroundTruth.csv"))
def add_path_to_df_2019(x):
    x = os.path.join(base_path, isic_2019, "train", x+".jpg")
    return x

df_2019["path"] = df_2019["image"].apply(add_path_to_df_2019)
df_2019 = df_2019.drop(columns=["SCC", "UNK" ])


df_2020 = pd.read_csv(os.path.join(base_path, isic_2020, "ISIC_2020_Training_GroundTruth.csv"))

def add_path_to_df_2020(x):
    x = os.path.join(base_path, isic_2020, "train", x+".jpg")
    return x
df_2020["path"] = df_2020["image_name"].apply(add_path_to_df_2020)

one_hot_df = pd.get_dummies(df_2020['target'], prefix='target')
df_2020 = pd.concat([df_2020, one_hot_df], axis=1)
df_2020 = df_2020.rename(columns ={"image_name":"image", "target_0": "NV", "target_1": "MEL", })
df_2020 = df_2020.drop(columns=["patient_id" , "sex",  "age_approx", "anatom_site_general_challenge", "diagnosis", "benign_malignant","target"])


stacked_df = pd.concat([df_2018, df_2019, df_2020], axis=0)
stacked_df['description'] = stacked_df[["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC"]].idxmax(axis=1)
category_mapping = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AK': 3,'BKL': 4 ,'DF': 5, 'VASC': 5}
stacked_df['label'] = stacked_df['description'].map(category_mapping)

for index, row in tqdm(stacked_df.iterrows()):
    if not os.path.isfile(row["path"]):
        print(row["path"])
        
stacked_df.to_csv("rgb_data.csv")
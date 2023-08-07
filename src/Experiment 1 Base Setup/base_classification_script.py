import os 
import sys
import numpy as np
import pandas as pd 
import pickle
import mlflow

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
from torchvision.models import efficientnet_b3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PickleDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        data_path = self.dataframe["path"].iloc[idx]  # Assuming "path" column is the first column
        label = self.dataframe["label"].iloc[idx]  # Assuming "label" column is the second column
        data_path = os.path.join("..", data_path)
        
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        data_dictionary = data_dict['data_point']
        sorted_keys = sorted(list(data_dictionary.keys()), reverse=False)
        
        if self.transform:
            #data = self.transform(data)
            data = np.vstack([self.transform(data_dictionary[key][:3]) for key in sorted_keys])
        else: 
            data = np.vstack([data_dictionary[key][:3] for key in sorted_keys])
        return data, label
    
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_layers_except_first_m_last_n(model,m, n):
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the first n layers
    for i, (name, param) in enumerate(model.named_parameters()):
        if i < m:
            param.requires_grad = True
    
    # Unfreeze the last n layers
    for i, (name, param) in enumerate(reversed(list(model.named_parameters()))):
        if i < n:
            param.requires_grad = True
    
    return model


if __name__ == '__main__':
    n_splits = 5
    batch_size = 4#32
    num_epochs = 100
    df = pd.read_csv(os.path.join("..", "..","data", "compiled_df_server.csv"))
    label_to_remove = 5 # Only 2 data points available for this class

    df = df[df['label'] != label_to_remove]

    df.reset_index(inplace=True)
    
    df["fold"] = np.nan
    skf = StratifiedKFold(n_splits=n_splits)
    skf.get_n_splits(df, df.label)
    for fold, (train_index, test_index) in enumerate(skf.split(df, df.label)):
        df.loc[test_index,"fold"]  = int(fold)

    df["id"] = df["path"].apply(lambda x :os.path.splitext(os.path.basename(x))[0])

    df['fold'] = df['fold'].astype('int')
    val_df =df[df["fold"]==0]
    train_df = df[df["fold"]!=0]
    print(train_df.shape, val_df.shape)
    # Just using one fold for now 
    
    
    train_dataset = PickleDataset(dataframe=train_df)
    val_dataset = PickleDataset(dataframe=val_df)
    # Access dataset samples using indexing
    data_sample, label = train_dataset[0]
    print('Sample data:', data_sample.shape)
    print('Label:', label)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=32, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=32, shuffle=True)
    model = efficientnet_b3(pretrained=True)
    print("data_sample.shape[1] : ", data_sample.shape[0])
    model.features[0][0] = nn.Conv2d(data_sample.shape[0],
                                                40,
                                                kernel_size=(3,3),
                                                stride=(2,2),
                                                padding=(1,1),
                                                bias=False)
    total_classes = len(set(df['label'].values))
    model.classifier[1] = nn.Linear(model.classifier[1].in_features,
                                    total_classes,
                                    bias=True)
    
    print(f"No. of parameters before freezing layers : ", count_trainable_parameters(model))
    model = freeze_layers_except_first_m_last_n(model, 1, 9)
    print(f"No. of parameters after freezing layers : ", count_trainable_parameters(model))

    
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    #for batch_data, batch_labels in dataloader:
    #    print(batch_data.shape, batch_labels)
    # Training loop
    best_accuracy = 0.0  # Initialize best accuracy
    best_model_path = os.path.join("models", "best_model.pth")  # Path to save the best model
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            #print(f"Epoch [{epoch + 1}/{num_epochs}], Iteration [{i + 1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}")
        print(f"Running test loss for epoch {epoch} : {running_loss}")        
        running_loss = 0.0
        # Testing loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f"Accuracy on test image {accuracy}%, epoch : {epoch}")
            
            
            
            # Save the model if it's the best so far
            if accuracy > best_accuracy:
                print(f"New best accuracy! Saving the model to {best_model_path}")
                best_accuracy = accuracy
                torch.save(model.state_dict(), best_model_path)
    print("Finished Training")
    
    print("best_accuracy achieved : ", best_accuracy)


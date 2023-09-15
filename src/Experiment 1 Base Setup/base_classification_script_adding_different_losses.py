import os 
import sys
import numpy as np
import pandas as pd 
import pickle

import torch

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
import scipy.ndimage as ndi 

import mlflow
from mlflow import log_metric, log_param, log_params, log_artifacts
from mlflow.models import infer_signature

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
    
    def process_using_histogram(self, data):
        # Scale the data to match the range of the histogram bins and convert to int
        scaled_data = (data * 4096).astype(int)
        
        # Clip the values in scaled_data to ensure they are within the valid range
        scaled_data = np.clip(scaled_data, 0, 4095)
        
        # Compute the histogram of the scaled data
        hist = ndi.histogram(scaled_data, min=0, max=4095, bins=4096)
        
        # Calculate the cumulative distribution function (CDF)
        cdf = hist.cumsum() / hist.sum()
        
        # Use the CDF to equalize the scaled_data
        equalised = cdf[scaled_data]
        
        return equalised / 4096
        
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
            data = np.vstack([self.transform(self.process_using_histogram(data_dictionary[key][:3])) for key in sorted_keys])
        else: 
            data = np.vstack([self.process_using_histogram(data_dictionary[key][:3]) for key in sorted_keys])
        
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

def train_test_loop(model, optimizer, num_epochs, trainloader, valloader, fold, best_model_path = os.path.join("models", "best_model.pth") ):
    best_accuracy = 0.0  # Initialize best accuracy
    for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                
                inputs, labels = data
                inputs = inputs.float()
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
            running_loss = 0.0
            # Testing loop
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in valloader:
                    images, labels = data
                    images = images.float()
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                log_metric("val_loss_" + str(fold), running_loss, step=epoch)  
                accuracy = 100 * (correct / total)
                # Save the model if it's the best so far
                if accuracy > best_accuracy:
                    print(f"Running train loss for epoch {epoch} : {running_loss}") 
                    print(f"New best accuracy! Saving the model to {best_model_path}")
                    print(f"Accuracy on test image {accuracy}%, epoch : {epoch}")
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), best_model_path)
                    mlflow.pytorch.save_model(model, "model_fold_"+str(fold) +"_accuracy" + str(best_accuracy))
                    
    print("best accuracy : ", best_accuracy)
    return best_accuracy

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        ce_loss = nn.CrossEntropyLoss()(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss
    
class CombinedLoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.cross_entropy = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(alpha=1, gamma=2)  # Instantiate the FocalLoss class
        self.kldiv_loss = nn.KLDivLoss(reduction='batchmean')  # Add reduction argument
        
    def forward(self, y_pred, y_true):
        ce_loss = self.cross_entropy(y_pred, y_true)
        focal_loss = self.focal_loss(y_pred, y_true)
        
        y_true_one_hot = torch.zeros_like(y_pred)
        y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1)  # Convert target to one-hot
        
        kldiv_loss = self.kldiv_loss(y_pred.log_softmax(dim=1), y_true_one_hot)
        
        combined_loss = self.alpha * ce_loss + self.beta * focal_loss + self.gamma * kldiv_loss
        
        return combined_loss

if __name__ == '__main__':
    with mlflow.start_run() as run:
        n_splits = 5
        batch_size = 4
        num_epochs = 500
        
        log_param("n_splits", n_splits)
        
        log_param("batch_size", batch_size)
        log_param("num_epochs", num_epochs)
        
        df_path = os.path.join("..", "..","data", "compiled_df_server.csv")
        log_param("df_path", df_path)
        df = pd.read_csv(df_path)
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
        for i in range(n_splits):
            val_df =df[df["fold"]==i]
            train_df = df[df["fold"]!=i]
            print(train_df.shape, val_df.shape)
            # Just using one fold for now 
            
            
            train_dataset = PickleDataset(dataframe=train_df)
            val_dataset = PickleDataset(dataframe=val_df)
            # Access dataset samples using indexing
            data_sample, label = train_dataset[0]

            trainloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=32, shuffle=True)
            valloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=32, shuffle=True)
            model = efficientnet_b3(pretrained=True)
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
            print(f"No. of parameters after freezing layers : ", count_trainable_parameters(model))

            model = model.to(device)
            
            criterion = CombinedLoss(alpha=0.5, beta=0.3, gamma=0.2)
            optimizer = optim.Adam(model.parameters(), lr=0.003)
            # Training loop
            
            best_model_path = os.path.join("models", "best_model.pth")  # Path to save the best model
            train_test_loop(model, optimizer, num_epochs, trainloader, valloader,i, best_model_path = os.path.join("models", "best_model.pth") )
            print(f"Finished Training fold : {i}")
            break
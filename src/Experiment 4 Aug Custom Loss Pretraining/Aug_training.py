import os 
import sys
import numpy as np
import pandas as pd 
import pickle
import mlflow
from tqdm import tqdm
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchmetrics import HingeLoss
from Aug import Aug_Hyperspectral_Data
from RGBDataset import RGBDataset

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
    """
    PickleDataset class for loading data from pickle files.

    Args:
        dataframe (DataFrame): The dataframe containing file paths and labels.
        shape (tuple, optional): Tuple representing the shape (height, width) of the data. Defaults to (512, 512).
        transform (Aug_Hyperspectral_Data, optional): Transformation to be applied to the data. Defaults to None.

    Attributes:
        dataframe (DataFrame): The dataframe containing file paths and labels.
        shape (tuple): Tuple representing the shape (height, width) of the data.
        transform (Aug_Hyperspectral_Data): Transformation to be applied to the data.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Loads and applies transformations to a data point.

    Examples:
        >>> dataset = PickleDataset(dataframe, shape=(256, 256), transform=Aug_Hyperspectral_Data())
        >>> data, label = dataset[0]
    """
    def __init__(self, dataframe, shape = (512, 512), transform=None):
        self.dataframe = dataframe
        self.shape = shape
        self.transform = transform
        if self.transform ==None:
            self.Aug = Aug_Hyperspectral_Data(self.shape, list_augmentations = [])
        else:
            self.Aug = Aug_Hyperspectral_Data(self.shape, list_augmentations = self.transform )
            
    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        Loads and applies transformations to a data point.

        Args:
            idx (int): Index of the data point in the dataset.

        Returns:
            data (Tensor): Transformed data point.
            label: The label associated with the data point.
        """
        data_path = self.dataframe["path"].iloc[idx]  # Assuming "path" column is the first column
        label = self.dataframe["label"].iloc[idx]  # Assuming "label" column is the second column
        data_path = os.path.join("..", data_path)
        
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        data_dictionary = data_dict['data_point']
        sorted_keys = sorted(list(data_dictionary.keys()), reverse=False)
        data = np.concatenate([data_dictionary[key][:3] for key in sorted_keys], axis = 0)
        data = np.transpose(data, (1, 2, 0))
        
        
        if self.transform is not None:
            data = self.Aug(data)

        data = torch.tensor(data.copy())
        
        data = data.permute(2, 0, 1)
        return data, label
    
def count_trainable_parameters(model):
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model: PyTorch model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_layers_except_first_m_last_n(model,m, n):
    """
    Freeze layers in a PyTorch model except for the first m and last n layers.

    Args:
        model: PyTorch model.
        m (int): Number of first layers to unfreeze.
        n (int): Number of last layers to unfreeze.

    Returns:
        model: Updated PyTorch model with frozen layers.
    """
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

def train_test_loop(model, optimizer, num_epochs, trainloader, valloader, fold, rgb_flag = True, best_model_path = os.path.join("models", "best_model.pth") ):
    """
    Train and test a PyTorch model.

    Args:
        model: PyTorch model to be trained.
        optimizer: PyTorch optimizer for training.
        num_epochs (int): Number of training epochs.
        trainloader (DataLoader): DataLoader for the training dataset.
        valloader (DataLoader): DataLoader for the validation dataset.
        fold (int): Fold number for tracking progress.
        rgb_flag (bool, optional): Flag to indicate whether the model is an RGB model. Defaults to True.
        best_model_path (str, optional): Path to save the best model checkpoint. Defaults to "models/best_model.pth".

    Returns:
        best_accuracy (float): Best accuracy achieved during training.
    """
    best_accuracy = 0.0  # Initialize best accuracy

    fold_desc = "Training on fold : "+ str(fold)
    for epoch in tqdm(range(num_epochs), desc=fold_desc):
            model.train()
            running_loss = 0.0
            train_desc = "Training on fold, epoch : "+ str(fold) + "_" + str(epoch)
            for i, data in tqdm(enumerate(trainloader, 0), desc=train_desc):
                inputs, labels = data
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
            test_desc = "Testing on fold, epoch : "+ str(fold) + "_" + str(epoch)
            with torch.no_grad():
                for data in tqdm(valloader, desc=test_desc):
                    
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item() 
                accuracy = 100 * correct / total
                
                # Save the model if it's the best so far
                if accuracy > best_accuracy:
                    print(f"Running train loss for epoch {epoch} : {running_loss}") 
                    print(f"New best accuracy! Saving the model to {best_model_path}")
                    print(f"Accuracy on test image {accuracy}%, epoch : {epoch}")
                    best_accuracy = accuracy
                    
                    torch.save(model.state_dict(), best_model_path)
                    if rgb_flag==True:
                        mlflow.pytorch.save_model(model, "model_fold_"+str(fold) +"_epoch_" +str(epoch) +"_accuracy_rgb" + str(best_accuracy))
                    else:
                         mlflow.pytorch.save_model(model, "model_fold_"+str(fold) +"_epoch_" +str(epoch) +"_accuracy" + str(best_accuracy))
    return best_accuracy


class FocalLoss(nn.Module):
    """
    Focal Loss implementation.

    Args:
        alpha (float, optional): Weighting factor for the positive class. Defaults to 1.
        gamma (float, optional): Focusing parameter. Defaults to 2.
    """
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        """
        Forward pass of the Focal Loss.

        Args:
            input: Model predictions.
            target: Ground truth labels.

        Returns:
            focal_loss: Computed focal loss.
        """
        ce_loss = nn.CrossEntropyLoss()(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined Loss implementation.

    Args:
        alpha (float): Weighting factor for cross-entropy loss.
        beta (float): Weighting factor for focal loss.
        gamma (float): Weighting factor for KL divergence loss.
    """
    def __init__(self, alpha, beta, gamma):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.cross_entropy = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(alpha=1, gamma=2)  # Instantiate the FocalLoss class
        self.kldiv_loss = nn.KLDivLoss(reduction='batchmean')  # Add reduction argument
        
    def forward(self, y_pred, y_true):
        """
        Forward pass of the Combined Loss.

        Args:
            y_pred: Model predictions.
            y_true: Ground truth labels.

        Returns:
            combined_loss: Computed combined loss.
        """
        ce_loss = self.cross_entropy(y_pred, y_true)
        focal_loss = self.focal_loss(y_pred, y_true)
        
        y_true_one_hot = torch.zeros_like(y_pred)
        y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1)  # Convert target to one-hot
        
        kldiv_loss = self.kldiv_loss(y_pred.log_softmax(dim=1), y_true_one_hot)
        
        combined_loss = self.alpha * ce_loss + self.beta * focal_loss + self.gamma * kldiv_loss
        
        return combined_loss


def get_model(in_channels, df):
    """
    Create a model for image classification.

    Args:
        in_channels (int): Number of input channels.
        df (DataFrame): Dataframe containing labels.

    Returns:
        model: Created model.
    """
    model = efficientnet_b3(pretrained=True)
    
    model.features[0][0] = nn.Conv2d(in_channels,  40,
                                                kernel_size=(3,3),
                                                stride=(2,2),
                                                padding=(1,1),
                                                bias=False)
    total_classes = len(set(df['label'].values))
    model.classifier[1] = nn.Linear(model.classifier[1].in_features,
                                    total_classes,
                                    bias=True)
    return model
if __name__ == '__main__':
    n_splits = 5
    batch_size = 4#32
    num_epochs = 20#20
    
    
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
    
    rgb_df = pd.read_csv("rgb_data.csv")

    train_rgb_df, test_rgb_df = train_test_split(rgb_df, test_size=0.1, stratify=rgb_df.label)
    train_rgb_dataset = RGBDataset(train_rgb_df)
    test_rgb_dataset = RGBDataset(test_rgb_df)
    rgb_train_loader = DataLoader(train_rgb_dataset, batch_size=batch_size, num_workers=32, shuffle=True)
    test_rgb_dataset = DataLoader(test_rgb_dataset, batch_size=batch_size, num_workers=32, shuffle=True)
    
    model = get_model(27, train_rgb_df)
    model.to(device)
    print(f"No. of parameters before freezing layers : ", count_trainable_parameters(model))
    #model = freeze_layers_except_first_m_last_n(model, 1, 9)
    print(f"No. of parameters after freezing layers : ", count_trainable_parameters(model))
    best_model_path_rgb = os.path.join("models", "best_model_rgb.pth")  # Path to save the best model
    criterion = CombinedLoss(alpha=0.5, beta=0.3, gamma=0.2)

    optimizer = optim.Adam(model.parameters(), lr=0.003)
    train_test_loop(model, optimizer, num_epochs, rgb_train_loader, test_rgb_dataset, 0, best_model_path = best_model_path_rgb )
    model.to("cpu")
    num_epochs = 300
    for i in range(n_splits):
        val_df =df[df["fold"]==i]
        train_df = df[df["fold"]!=i]
        print(train_df.shape, val_df.shape)
        train_dataset = PickleDataset(dataframe=train_df, transform= ["resize",#"definition_loss", "rotation", 
                                                        #"translation", "cut_out", #"shuffle_channels",
                                                        "horizontal_flip", "vertical_flip"])
        val_dataset = PickleDataset(dataframe=val_df, transform=None)
        # Access dataset samples using indexing
        data_sample, label = train_dataset[0]
        trainloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=32, shuffle=True)
        valloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=32, shuffle=True)

        new_model_top = get_model(27, train_rgb_df)
        total_classes = len(set(df['label'].values))
        new_model_top.load_state_dict(model.state_dict())
        
        new_model = nn.Sequential(new_model_top, nn.Linear(len(set(rgb_df['label'].values)), len(set(df['label'].values))))
        new_model = new_model.to(device)
        criterion = CombinedLoss(alpha=0.5, beta=0.3, gamma=0.2)
        #criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(new_model.parameters(), lr=0.003)
        # Training loop
        
        best_model_path = os.path.join("models", "best_model.pth")  # Path to save the best model
        
        train_test_loop(new_model, optimizer, num_epochs, trainloader, valloader, i, rgb_flag=False, best_model_path = best_model_path )
        print(f"Finished Training fold : {i}")
        
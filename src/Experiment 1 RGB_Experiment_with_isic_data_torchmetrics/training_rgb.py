import os 
import sys
import numpy as np
import pandas as pd 
import pickle
import mlflow
import cv2
from tqdm import tqdm
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchmetrics import HingeLoss
from mlflow import log_metric, log_param, log_params, log_artifacts
from mlflow.models import infer_signature
from glob import glob 
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 

from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassAveragePrecision, MulticlassCalibrationError, MulticlassSpecificity
from torchmetrics.classification import  MulticlassConfusionMatrix, MulticlassF1Score, MulticlassMatthewsCorrCoef, MulticlassPrecision, MulticlassRecall

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
from torchvision.models import efficientnet_b3



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class PickleDataset(Dataset):
    """
    Custom PyTorch dataset for loading and preprocessing pickled image data.

    Args:
        dataframe (DataFrame): The DataFrame containing image paths and labels.
        shape (tuple): The desired shape of the images (default is (512, 512)).
        transform (callable): A composition of image augmentation transformations (default is None).
    """
    def __init__(self, dataframe, shape = (512, 512), transform=None):
        self.dataframe = dataframe
        self.shape = shape
        self.transform = transform
        if self.transform ==None:
            self.Aug = A.Compose([
                        ToTensorV2()
                        ])
        else:
            self.Aug = transform
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        Loads and preprocesses an image at the given index.

        Args:
            idx (int): Index of the image in the dataset.

        Returns:
            img (Tensor): Preprocessed image.
            label (Tensor): Label associated with the image.
        """
        data_path = self.dataframe["path"].iloc[idx]  # Assuming "path" column is the first column
        folders = os.path.normpath(data_path).split("\\")
        label = self.dataframe["label"].iloc[idx]  # Assuming "label" column is the second column

        data_path_base = os.path.join("..","..","..",)
        data_path = os.path.join(data_path_base,  *folders)
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        img = data_dict['img']
        # Augment an image
        transformed = self.Aug(image=img)
        img = transformed["image"]
        return img, torch.tensor(label)


class RGBDataset(Dataset):
    """
    Custom PyTorch dataset for loading and preprocessing RGB image data.

    Args:
        dataframe (DataFrame): The DataFrame containing RGB image paths and labels.
        shape (tuple): The desired shape of the images (default is (512, 512)).
        transform (callable): A composition of image augmentation transformations (default is None).
    """
    def __init__(self, dataframe, shape = (512, 512), transform=None):
        self.dataframe = dataframe
        self.shape = shape
        self.transform = transform
        if self.transform ==None:
            self.Aug = A.Compose([
                        ToTensorV2()
                        ])
        else:
            self.Aug = transform
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        Loads and preprocesses an RGB image at the given index.

        Args:
            idx (int): Index of the RGB image in the dataset.

        Returns:
            img (Tensor): Preprocessed RGB image.
            label (Tensor): Label associated with the RGB image.
        """
        data_path = self.dataframe["path"].iloc[idx]  # Assuming "path" column is the first column
        
        
        folders = os.path.normpath(data_path).split("\\")
        label = self.dataframe["label"].iloc[idx]  # Assuming "label" column is the second column

        data_path_base = os.path.join("..","..","..",)
        data_path = os.path.join(data_path_base,  *folders )
        #print("data_path", data_path)
        img = cv2.imread(data_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, self.shape)
        # Augment an image
        transformed = self.Aug(image=img)
        img = transformed["image"]
        return img, torch.tensor(label)



 
def count_trainable_parameters(model):
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        int: The total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_layers_except_first_m_last_n(model,m, n):
    """
    Freeze layers in a PyTorch model, except for the first 'm' layers and the last 'n' layers.

    Args:
        model (nn.Module): The PyTorch model.
        m (int): Number of initial layers to leave unfrozen.
        n (int): Number of final layers to leave unfrozen.

    Returns:
        nn.Module: The modified PyTorch model with specified layers unfrozen.
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




def train_test_loop(model, optimizer, criterion, epoch, trainloader, valloader, fold, total_classes):
    """
    Perform the training and testing loop for a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to be trained and tested.
        optimizer (torch.optim.Optimizer): The optimizer used for model optimization.
        criterion (nn.Module): The loss function used for training.
        epoch (int): The current epoch number.
        trainloader (DataLoader): DataLoader for the training dataset.
        valloader (DataLoader): DataLoader for the validation dataset.
        fold (int): The fold number for cross-validation.
        total_classes (int): The total number of classes in the dataset.

    Returns:
        tuple: A tuple containing:
            - float: The running loss during training.
            - float: The accuracy achieved on the validation dataset.
            - dict: A dictionary containing various evaluation metrics.
    """
    metric_dictionary = get_metric_dictionary(total_classes)
    fold_desc = f"Training on fold : {fold} and epoch  {epoch}"
    for _ in tqdm(range(1), desc=fold_desc):
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
                #print("Testing images, labels : ", images.shape, labels.shape)
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)

                correct += (predicted == labels).sum().item() 
                
                # Update metrics for each sample
                for metric_name, metric_instance in metric_dictionary.items():
                    metric_instance.update(outputs.to(device), labels.to(device))

            accuracy = 100 * (correct / total)
            
    return running_loss, accuracy , metric_dictionary


class FocalLoss(nn.Module):
    """
    Focal Loss for classification tasks.

    Args:
        alpha (float): The weight factor for the rare class (default is 1.0).
        gamma (float): The focusing parameter (default is 2.0).

    Returns:
        torch.Tensor: The calculated focal loss.
    """
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        """
        Calculate the focal loss.

        Args:
            input (torch.Tensor): The model's predicted probabilities.
            target (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The calculated focal loss.
        """
        ce_loss = nn.CrossEntropyLoss()(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined Loss function for multi-objective learning.

    Args:
        alpha (float): Weight factor for cross-entropy loss.
        beta (float): Weight factor for focal loss.
        gamma (float): Weight factor for Kullback-Leibler Divergence loss.

    Returns:
        torch.Tensor: The combined loss.
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
        Calculate the combined loss.

        Args:
            y_pred (torch.Tensor): The model's predicted probabilities.
            y_true (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The combined loss.
        """
        ce_loss = self.cross_entropy(y_pred, y_true)
        focal_loss = self.focal_loss(y_pred, y_true)
        
        y_true_one_hot = torch.zeros_like(y_pred)
        y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1)  # Convert target to one-hot
        
        kldiv_loss = self.kldiv_loss(y_pred.log_softmax(dim=1), y_true_one_hot)
        
        combined_loss = self.alpha * ce_loss + self.beta * focal_loss + self.gamma * kldiv_loss
        
        return combined_loss


def get_model(total_classes):
    """
    Create a model for classification with a modified classifier head.

    Args:
        total_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The model with the modified classifier.
    """
    model = efficientnet_b3(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features,
                                    total_classes,
                                    bias=True)
    return model

def get_metric_dictionary(total_classes):
    """
    Create a dictionary of evaluation metrics for a multi-class classification task.

    Args:
        total_classes (int): The number of output classes.

    Returns:
        dict: A dictionary containing various evaluation metrics.
    """
    print("total_classes : ", total_classes, type(total_classes))
    metric_dictionary = {
        "MulticlassAccuracy" : MulticlassAccuracy(num_classes=total_classes).to(device),
        "MulticlassAUROC" : MulticlassAUROC(num_classes=total_classes).to(device),
        "MulticlassAveragePrecision" : MulticlassAveragePrecision(num_classes=total_classes).to(device),
        "MulticlassCalibrationError" : MulticlassCalibrationError(num_classes=total_classes, n_bins=30, norm='l2').to(device),
        "MulticlassConfusionMatrix" : MulticlassConfusionMatrix(num_classes=total_classes).to(device),
        "MulticlassF1Score" : MulticlassF1Score(num_classes=total_classes).to(device),
        "MulticlassMatthewsCorrCoef" : MulticlassMatthewsCorrCoef(num_classes=total_classes).to(device),
        "MulticlassPrecision" : MulticlassPrecision(num_classes=total_classes).to(device),
        "MulticlassRecall" :MulticlassRecall(num_classes=total_classes).to(device),
        "MulticlassSpecificity" :MulticlassSpecificity(num_classes=total_classes).to(device),
    }
    return metric_dictionary


if __name__ == '__main__':
    renaming_2018 = {'MEL':'melanoma',
             'NV':'melanocytic nevus',
             'BCC':'basal cell carcinoma',
             'AKIEC':'actinic keratosis',
             'BKL':'benign keratosis',
             'DF':'dermatofibroma',
             'VASC':'vascular lesion'}
    
    
    n_splits = 5
    batch_size = 16#32
    num_epochs = 20#20
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.Normalize(mean=mean, std=std),
                        A.RandomBrightnessContrast(p=0.1),
                        ToTensorV2()
                        ])
    
    test_transform = A.Compose([                        
                        A.Normalize(mean=mean, std=std),
                        ToTensorV2()
                        ])
    log_param("n_splits", n_splits)
        
    log_param("batch_size", batch_size)
    log_param("num_epochs", num_epochs)
    
    
    rgb_df = pd.read_csv(os.path.join("..","..","..","..","..", "isic_datasets","sorted_df.csv"))
    

    df = pd.read_csv(os.path.join("..","..","..","compiled_df_server_new.csv"))

    print(df[df.index.duplicated()])
    
    #matching labels from isic data and id2 data
    
    renaming_description = {'MM':'melanoma',
                'Basalzellkarzinom':'basal cell carcinoma',
                'BZK':'basal cell carcinoma',
                'Nävus':'melanocytic nevus',
                'N':'melanocytic nevus',
                'Melanoma':'melanoma',
                'Plattenephitelkarzinom':'squamous cell carcinoma',
                'PEK':'squamous cell carcinoma',
                "Seborrhoische Keratose": "benign keratosis"}
    df['description'] = df['description'].replace(renaming_description)
    
    
    
    renaming_labels = {3: 1, 2:0, 1:3, # the common between two lists
                       0:7 , 4:8, 
                       }
    
    condition = (df['description'] == "benign keratosis")

    df.loc[condition, 'label'] = 4 # labeling Seborrhoische Keratose to label of ISIC data  benign keratosis i.e 4
    
    df['description'] = df['description'].replace(renaming_description)

    label_to_remove = 5 # Only 2 data points available for this class

    df = df[df['label'] != label_to_remove]
    df['label'] = df['label'].replace(renaming_labels)

    df.reset_index(inplace=True)
    
    total_classes = int(max(set(df['label'].values))+1)

    df["fold"] = np.nan
    skf = StratifiedKFold(n_splits=n_splits)
    skf.get_n_splits(df, df.label)
    for fold, (train_index, test_index) in enumerate(skf.split(df, df.label)):
        df.loc[test_index,"fold"]  = int(fold)

    df["id"] = df["path"].apply(lambda x :os.path.splitext(os.path.basename(x))[0])

    df['fold'] = df['fold'].astype('int')
    
    
    num_epochs = 300
    for i in range(n_splits):
        val_df =df[df["fold"]==i]
        train_df = df[df["fold"]!=i]
        print("train_df.shape, val_df.shape :", train_df.shape, val_df.shape)
        train_dataset = PickleDataset(dataframe=train_df, transform = train_transform)# transform= ["resize",#"definition_loss", "rotation", 
                                                        #"translation", "cut_out", #"shuffle_channels",
                                                        #"horizontal_flip", "vertical_flip"])
        val_dataset = PickleDataset(dataframe=val_df, transform=test_transform)

        val_rgb_df =rgb_df[rgb_df["fold"]==i]
        train_rgb_df = rgb_df[rgb_df["fold"]!=i]
        print("val_rgb_df.shape, train_rgb_df.shape : ", val_rgb_df.shape, train_rgb_df.shape)
        train_rgb_dataset = RGBDataset(dataframe=train_rgb_df, transform = train_transform)# transform= ["resize",#"definition_loss", "rotation", 
                                                        #"translation", "cut_out", #"shuffle_channels",
                                                        #"horizontal_flip", "vertical_flip"])
        val_rgb_dataset = RGBDataset(dataframe=val_rgb_df, transform=test_transform)
        
        # Access dataset samples using indexing
        print("train_dataset : ", train_dataset)

        for data, label in train_dataset:
            print(data.shape, label.shape)
            break 
        
        # Access dataset samples using indexing
        print("train_rgb_dataset : ", train_rgb_dataset)

        for data, label in train_rgb_dataset:
            print(data.shape, label.shape)
            break 

        trainloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=32, shuffle=True)
        valloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=32, shuffle=True)
    
        
        trainrgbloader = DataLoader(train_rgb_dataset, batch_size=batch_size, num_workers=32, shuffle=True)
        valrgbloader = DataLoader(val_rgb_dataset, batch_size=batch_size, num_workers=32, shuffle=True)
        
        model = get_model(total_classes)
    
        model.to(device)
        best_model_path_rgb = os.path.join("models", "best_model_rgb.pth")  # Path to save the best model
        criterion = CombinedLoss(alpha=0.5, beta=0.3, gamma=0.2)

        optimizer = optim.Adam(model.parameters(), lr=0.003)

        # Training loop
        
        best_model_path = os.path.join("models", "best_model.pth")  # Path to save the best model
        
        best_accuracy = 0
        
        for epoch in range(num_epochs):
            running_loss_rgb, accuracy_rgb, rgb_metric_dictionary = train_test_loop(model, optimizer, criterion, 1, trainrgbloader, valrgbloader, i, total_classes)
            
            running_loss, accuracy, pickle_metric_dictionary = train_test_loop(model, optimizer, criterion, epoch, trainloader, valloader, i , total_classes)
            
            if accuracy > best_accuracy:
                print(f"Running train loss for epoch {epoch} : {running_loss}") 
                print(f"New best accuracy! Saving the model to {best_model_path}")
                print(f"Accuracy on test image {accuracy}%, epoch : {epoch}")
                print(f"Running RGB loss {running_loss_rgb}, RGB Accuracy {accuracy_rgb}")
                best_accuracy = accuracy
                torch.save(model.state_dict(), best_model_path)
                mlflow.pytorch.save_model(model, os.path.join(os.getcwd(),"model", "model_fold_"+str(i) +"_epoch_" +str(epoch) +"_accuracy" + str(best_accuracy)))
                print()
                print("RGB Statistics")
                for metric_name, metric_instance in rgb_metric_dictionary.items():
                    metric_value = metric_instance.compute()
                    print(f"{metric_name}: {metric_value}")
                    
                print()
                print("Pickle Statistics")
                for metric_name, metric_instance in pickle_metric_dictionary.items():
                    metric_value = metric_instance.compute()
                    print(f"{metric_name}: {metric_value}")
                
                print()
                

        print(f"Finished Training fold : {i}")
        break
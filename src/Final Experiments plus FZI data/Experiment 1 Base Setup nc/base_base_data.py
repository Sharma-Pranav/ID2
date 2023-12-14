import os 
import sys
import numpy as np
import pandas as pd 
import pickle
from glob import glob
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision.models import efficientnet_b3
from sklearn.model_selection import StratifiedKFold
import scipy.ndimage as ndi 
from torchvision import transforms
import mlflow
from mlflow import log_metric, log_param, log_params, log_artifacts
from mlflow.models import infer_signature

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 

from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassAveragePrecision, MulticlassCalibrationError, MulticlassSpecificity
from torchmetrics.classification import  MulticlassConfusionMatrix, MulticlassF1Score, MulticlassMatthewsCorrCoef, MulticlassPrecision, MulticlassRecall


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class PickleDataset(Dataset):
    """
    A PyTorch dataset for loading data stored in pickle files.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing 'path' and 'label' columns.
        shape (tuple): The desired shape of the data. Default is (512, 512).
        transform (Compose): An optional data augmentation transform using Albumentations. 
                            If None, a default transformation with ToTensorV2 is applied.

    Attributes:
        dataframe (pd.DataFrame): The input DataFrame containing file paths and labels.
        shape (tuple): The desired shape of the data.
        transform (Compose): The data augmentation transform applied during loading.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Loads and preprocesses the data at the given index.

    Example:
        dataset = PickleDataset(dataframe=my_dataframe, shape=(256, 256), transform=my_transform)
        sample_data, sample_label = dataset[0]
    """
    def __init__(self, dataframe, shape = (512, 512), transform=None):
        """
        Initializes the PickleDataset.

        Parameters:
            dataframe (pd.DataFrame): The DataFrame containing 'path' and 'label' columns.
            shape (tuple): The desired shape of the data. Default is (512, 512).
            transform (Compose): An optional data augmentation transform using Albumentations. 
                                If None, a default transformation with ToTensorV2 is applied.
        """
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
        Loads and preprocesses the data at the given index.

        Parameters:
            idx (int): Index of the desired sample.

        Returns:
            torch.Tensor: The preprocessed data.
            torch.Tensor: The corresponding label.
        """
        data_path = self.dataframe["path"].iloc[idx]  
        folders = os.path.normpath(data_path).split("\\")
        label = self.dataframe["label"].iloc[idx]  


        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
         
        data_dictionary = data_dict['data_point']
        sorted_keys = sorted(list(data_dictionary.keys()), reverse=False)
        
        data = np.vstack([data_dictionary[key][:3] for key in sorted_keys])

        
        return torch.tensor(data), torch.tensor(label)


def count_trainable_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.

    Parameters:
        model (nn.Module): The PyTorch model for which trainable parameters will be counted.

    Returns:
        int: The total number of trainable parameters in the model.

    Example:
        model = YourCustomModel()
        num_params = count_trainable_parameters(model)
        print(f'The model has {num_params} trainable parameters.')
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_layers_except_first_m_last_n(model,m, n):
    """
    Freezes all layers in a PyTorch model except for the first m and last n layers.

    Parameters:
        model (nn.Module): The PyTorch model to modify.
        m (int): Number of layers to unfreeze from the beginning.
        n (int): Number of layers to unfreeze from the end.

    Returns:
        nn.Module: The modified model with specified layers unfrozen.

    Example:
        model = YourCustomModel()
        model = freeze_layers_except_first_m_last_n(model, 2, 1)
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

def get_metric_dictionary(total_classes):
    """
    Creates and returns a dictionary of multiclass classification metrics for evaluation.

    Parameters:
        total_classes (int): The total number of classes in the classification problem.

    Returns:
        dict: A dictionary containing instances of various multiclass classification metrics.

    Example:
        total_classes = 10
        metrics_dict = get_metric_dictionary(total_classes)
    """
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




def train_test_loop(model, optimizer, criterion, epoch, trainloader, valloader, fold, total_classes):
    """
    Training and testing loop for a PyTorch model.

    Parameters:
        model (nn.Module): The PyTorch model to train and test.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        criterion: The loss function used for training and testing.
        epoch (int): The current epoch number.
        trainloader (DataLoader): DataLoader for the training dataset.
        valloader (DataLoader): DataLoader for the validation dataset.
        fold (int): The current fold number in cross-validation.
        total_classes (int): The total number of classes in the classification problem.

    Returns:
        float: The training loss.
        float: The accuracy on the validation set.
        dict: A dictionary containing instances of various multiclass classification metrics.

    Example:
        train_loss, val_accuracy, metrics_dict = train_test_loop(model, optimizer, criterion, 1, train_loader, val_loader, 1, 10)
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
    Focal Loss for binary or multiclass classification tasks.

    Parameters:
        alpha (float): Focusing parameter. Default is 1.
        gamma (float): Modulation factor for the steepness of the loss function. Default is 2.

    Methods:
        forward(input, target): Computes the Focal Loss given the model predictions and target labels.

    Example:
        loss_function = FocalLoss(alpha=1.5, gamma=2)
        output = your_model(input_data)
        loss = loss_function(output, target_labels)
    """

    def __init__(self, alpha=1, gamma=2):
        """
        Initializes the FocalLoss.

        Parameters:
            alpha (float): Focusing parameter. Default is 1.
            gamma (float): Modulation factor for the steepness of the loss function. Default is 2.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        """
        Computes the Focal Loss given the model predictions and target labels.

        Parameters:
            input (torch.Tensor): The raw output of the model.
            target (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The computed Focal Loss.
        """
        ce_loss = nn.CrossEntropyLoss()(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss
    
class CombinedLoss(nn.Module):
    """
    Combined Loss for multi-objective learning using a weighted combination of CrossEntropy, Focal, and KL Divergence losses.

    Parameters:
        alpha (float): Weight for CrossEntropy loss.
        beta (float): Weight for Focal loss.
        gamma (float): Weight for KL Divergence loss.

    Methods:
        forward(y_pred, y_true): Computes the combined loss given the model predictions and target labels.

    Example:
        loss_function = CombinedLoss(alpha=1.0, beta=2.0, gamma=0.5)
        output = your_model(input_data)
        loss = loss_function(output, target_labels)
    """
    def __init__(self, alpha, beta, gamma):
        """
        Initializes the CombinedLoss.

        Parameters:
            alpha (float): Weight for CrossEntropy loss.
            beta (float): Weight for Focal loss.
            gamma (float): Weight for KL Divergence loss.
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.cross_entropy = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(alpha=1, gamma=2)  # Instantiate the FocalLoss class
        self.kldiv_loss = nn.KLDivLoss(reduction='batchmean')  # Add reduction argument
        
    def forward(self, y_pred, y_true):
        """
        Computes the combined loss given the model predictions and target labels.

        Parameters:
            y_pred (torch.Tensor): The raw output of the model.
            y_true (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The computed combined loss.
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
    Retrieves an EfficientNet-B3 model pre-trained on ImageNet and adapts the output layer for a custom number of classes.

    Parameters:
        total_classes (int): The total number of classes in the classification problem.

    Returns:
        nn.Module: The configured EfficientNet-B3 model with the modified output layer.

    Example:
        model = get_model(total_classes=10)
    """
    model = efficientnet_b3(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features,
                                    total_classes,
                                    bias=True)
    return model
def append_fzi_path(x):
    """
    Appends a relative path to the 'data' directory at the beginning of the provided path.

    Parameters:
        x (str): The input path.

    Returns:
        str: The updated path with the 'data' directory added.

    Example:
        updated_path = append_fzi_path("your/original/path")
    """
    folders = os.path.normpath(x).split("\\")
    return os.path.join("..","..","..","data", *folders) 

def append_old_path(x):
    """
    Appends a relative path at the beginning of the provided path.

    Parameters:
        x (str): The input path.

    Returns:
        str: The updated path.

    Example:
        updated_path = append_old_path("your/original/path")
    """
    folders = os.path.normpath(x).split("\\")
    return os.path.join("..","..","..","..",*folders) 

if __name__ == '__main__':
    with mlflow.start_run(nested=True) as run:
        n_splits = 5
        batch_size = 4
        num_epochs = 300#500
        best_accuracy_list = []
        log_param("depth", 5)
        log_param("n_splits", n_splits)
        
        log_param("batch_size", batch_size)
        log_param("num_epochs", num_epochs)

        df = pd.read_csv(os.path.join("..","..","..","..","compiled_df_server_new.csv"))

        label_to_remove = 5 # Only 2 data points available for this class

        df = df[df['label'] != label_to_remove]
        df.reset_index(inplace=True)
        print(df)
        print(df.columns)
        fzi_df = pd.read_csv(os.path.join("..","..","..","data", "fzi.csv"))
        fzi_df = fzi_df[ ['description', 'path', 'label',]]
        fzi_df['path'] = fzi_df['path'].apply(append_fzi_path)
        df = df[ ['description', 'path', 'label',]]
        df['path'] = df['path'].apply(append_old_path)
        print(fzi_df)
        
        df = fzi_df.append(df[fzi_df.columns], ignore_index=True)
        print(df["label"].value_counts())
        print(df["description"].value_counts())

        print(df.shape, df.head())
        data_path_base = glob(os.path.join("..","..","..","data", "*"))
        print(data_path_base)
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

            best_model_path = os.path.join("models", "model_fold_"+ str(i) + "best_model.pth")  # Path to save the best model
            best_model_metric_path = os.path.join("models", "model_fold_"+ str(i) + "best_model_metrics.pkl")
            best_accuracy = 0
            
            for epoch in range(num_epochs):
                
                running_loss, accuracy, pickle_metric_dictionary = train_test_loop(model, optimizer, criterion, epoch, trainloader, valloader, i , total_classes)
                
                if accuracy > best_accuracy:
                    print(f"Running train loss for epoch {epoch} : {running_loss}") 
                    print(f"New best accuracy! Saving the model to {best_model_path}")
                    print(f"Accuracy on test image {accuracy}%, epoch : {epoch}")
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), best_model_path)
                    mlflow.pytorch.save_model(model, os.path.join(os.getcwd(),"model", "model_fold_"+str(i) +"_epoch_" +str(epoch) +"_accuracy" + str(best_accuracy)))
                    print()
                    print()

                    print("Pickle Statistics")
                    for metric_name, metric_instance in pickle_metric_dictionary.items():
                        metric_value = metric_instance.compute()
                        print(f"{metric_name}: {metric_value}")
                    
                    print()
                    # Pickle the dictionary
                    with open(best_model_metric_path, 'wb') as pickle_file:
                        pickle.dump(pickle_metric_dictionary, pickle_file)

                    print("Metrics pickled successfully!")
            best_accuracy_list.append(best_accuracy)
            print(f"Finished Training fold : {i}")

        avg_acc = sum(best_accuracy_list)/n_splits
        
        file_path = 'best_accuracy_average.txt'
        with open(file_path, 'w') as file:
            # Write the value to the file
            value_to_save = f'Kfold_Accuracy_'+str(avg_acc)
            file.write(value_to_save)
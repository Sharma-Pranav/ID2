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
from sklearn.model_selection import StratifiedKFold
from torchmetrics import HingeLoss
from Aug import Aug_Hyperspectral_Data


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
    Custom PyTorch dataset for loading data from pickle files.

    Args:
        dataframe (pd.DataFrame): DataFrame containing file paths and labels.
        shape (tuple): Tuple specifying the shape of the data.
        transform (list or None): List of augmentations to apply or None.

    Attributes:
        dataframe (pd.DataFrame): DataFrame containing file paths and labels.
        shape (tuple): Tuple specifying the shape of the data.
        transform (list or None): List of augmentations to apply or None.
        Aug (Aug_Hyperspectral_Data): Augmentation object.

    Methods:
        __len__(): Get the number of samples in the dataset.
        __getitem__(idx): Get a sample and its corresponding label.
    """
    def __init__(self, dataframe, shape = (512, 512), transform=None):
        """
        Initialize the PickleDataset.

        Args:
            dataframe (pd.DataFrame): DataFrame containing file paths and labels.
            shape (tuple, optional): Tuple specifying the shape of the data. Defaults to (512, 512).
            transform (list or None, optional): List of augmentations to apply or None. Defaults to None.
        """
        self.dataframe = dataframe
        self.shape = shape
        self.transform = transform
        if self.transform ==None:
            self.Aug = Aug_Hyperspectral_Data(self.shape, list_augmentations = [])
        else:
            self.Aug = Aug_Hyperspectral_Data(self.shape, list_augmentations = self.transform )
    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        Get a sample and its corresponding label.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the data and its label.
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
        
        
        return data, label
    
def count_trainable_parameters(model):
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): PyTorch model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_layers_except_first_m_last_n(model,m, n):
    """
    Freeze layers in a PyTorch model, except for the first m and last n layers.

    Args:
        model (nn.Module): PyTorch model.
        m (int): Number of initial layers to keep trainable.
        n (int): Number of final layers to keep trainable.

    Returns:
        nn.Module: Modified PyTorch model.
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

def train_test_loop(model, optimizer, num_epochs, trainloader, valloader, fold, best_model_path = os.path.join("models", "best_model.pth") ):
    """
    Train and evaluate a PyTorch model over multiple epochs.

    Args:
        model (nn.Module): PyTorch model.
        optimizer: PyTorch optimizer.
        num_epochs (int): Number of training epochs.
        trainloader (DataLoader): DataLoader for training data.
        valloader (DataLoader): DataLoader for validation data.
        fold (int): Fold number.
        best_model_path (str, optional): Path to save the best model checkpoint. Defaults to "models/best_model.pth".

    Returns:
        float: Best accuracy achieved during training.
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
                    mlflow.pytorch.save_model(model, "model_fold_"+str(fold) +"_epoch_" +str(epoch) +"_accuracy" + str(best_accuracy))
                    
                    
    print("best accuracy : ", best_accuracy)
    return best_accuracy


class FocalLoss(nn.Module):
    """
    Focal Loss is a custom loss function designed to address class imbalance in classification tasks.
    
    Args:
        alpha (float, optional): A hyperparameter that controls the weight assigned to each class.
            Defaults to 1.
        gamma (float, optional): A hyperparameter that controls the focusing effect of the loss.
            Higher values give more focus to hard-to-classify examples. Defaults to 2.
    
    Attributes:
        alpha (float): The alpha hyperparameter.
        gamma (float): The gamma hyperparameter.
    
    Methods:
        forward(input, target):
            Compute the Focal Loss given input predictions and target labels.
    
    Examples:
        >>> criterion = FocalLoss(alpha=1, gamma=2)
        >>> loss = criterion(predictions, labels)
    """
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        """
        Compute the Focal Loss given input predictions and target labels.
        
        Args:
            input (Tensor): The predicted class scores from the model.
            target (Tensor): The true class labels.
        
        Returns:
            Tensor: The computed Focal Loss.
        """
        ce_loss = nn.CrossEntropyLoss()(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss


class CombinedLoss(nn.Module):
    """
    CombinedLoss is a custom loss function that combines Cross-Entropy Loss, Focal Loss,
    and Kullback-Leibler Divergence Loss in a linear combination.
    
    Args:
        alpha (float): Coefficient for Cross-Entropy Loss.
        beta (float): Coefficient for Focal Loss.
        gamma (float): Coefficient for Kullback-Leibler Divergence Loss.
    
    Attributes:
        alpha (float): Coefficient for Cross-Entropy Loss.
        beta (float): Coefficient for Focal Loss.
        gamma (float): Coefficient for Kullback-Leibler Divergence Loss.
    
    Methods:
        forward(y_pred, y_true):
            Compute the Combined Loss given predicted class scores and true class labels.
    
    Examples:
        >>> criterion = CombinedLoss(alpha=0.5, beta=0.2, gamma=0.3)
        >>> loss = criterion(predictions, labels)
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
        Compute the Combined Loss given predicted class scores and true class labels.
        
        Args:
            y_pred (Tensor): The predicted class scores from the model.
            y_true (Tensor): The true class labels.
        
        Returns:
            Tensor: The computed Combined Loss.
        """
        ce_loss = self.cross_entropy(y_pred, y_true)
        focal_loss = self.focal_loss(y_pred, y_true)
        
        y_true_one_hot = torch.zeros_like(y_pred)
        y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1)  # Convert target to one-hot
        
        kldiv_loss = self.kldiv_loss(y_pred.log_softmax(dim=1), y_true_one_hot)
        
        combined_loss = self.alpha * ce_loss + self.beta * focal_loss + self.gamma * kldiv_loss
        
        return combined_loss



if __name__ == '__main__':
    n_splits = 5
    batch_size = 4#32
    num_epochs = 300#20
    
    
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
        train_dataset = PickleDataset(dataframe=train_df, transform= ["resize",#"definition_loss", "rotation", 
                                                        #"translation", "cut_out", #"shuffle_channels",
                                                        "horizontal_flip", "vertical_flip"])
        val_dataset = PickleDataset(dataframe=val_df, transform=None)
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
        train_test_loop(model, optimizer, num_epochs, trainloader, valloader, i, best_model_path = os.path.join("models", "best_model.pth") )
        print(f"Finished Training fold : {i}")
        
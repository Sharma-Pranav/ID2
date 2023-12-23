import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd as autograd
from tqdm import tqdm
import scipy.optimize as op
class ResNetLayer(nn.Module):
    """
    Implements a Residual Block layer for neural networks.

    Args:
        n_channels (int): The number of input channels to the layer.
        n_inner_channels (int): The number of inner channels used in the layer.
        kernel_size (int, optional): The size of the convolutional kernel. Default is 3.
        num_groups (int, optional): The number of groups for group normalization. Default is 8.

    Attributes:
        conv1 (torch.nn.Conv2d): The first convolutional layer.
        conv2 (torch.nn.Conv2d): The second convolutional layer.
        norm1 (torch.nn.GroupNorm): The first group normalization layer.
        norm2 (torch.nn.GroupNorm): The second group normalization layer.
        norm3 (torch.nn.GroupNorm): The third group normalization layer.

    Methods:
        __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8):
            Initializes a ResNetLayer instance with the specified parameters.

        forward(self, z, x):
            Performs a forward pass through the ResNetLayer.

    Example:
        >>> import torch.nn as nn
        >>> # Create a ResNetLayer with 64 input channels and 128 inner channels
        >>> layer = ResNetLayer(64, 128)
        >>> # Perform a forward pass through the layer
        >>> output = layer(input_z, input_x)
    """
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(n_inner_channels, n_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        
    def forward(self, z, x):
        """
        Forward pass through the ResNetLayer.

        Args:
            z (torch.Tensor): The input tensor to the layer.
            x (torch.Tensor): The input tensor that will be added to the output of the layer.

        Returns:
            torch.Tensor: The output tensor of the layer.

        Example:
            >>> output = layer(input_z, input_x)
        """
        y = self.norm1(F.relu(self.conv1(z)))
        return self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))
    
def chebyshev_acceleration(f, x0, max_iter=50, tol=1e-2):
    """
    Accelerates convergence using Chebyshev acceleration.

    This function iteratively applies Chebyshev acceleration to estimate the fixed point of the function `f`.

    Args:
        f (callable): The function for which the fixed point is sought.
        x0 (torch.Tensor): The initial guess for the fixed point.
        max_iter (int, optional): The maximum number of iterations. Default is 50.
        tol (float, optional): The tolerance for convergence. Default is 1e-2.

    Returns:
        torch.Tensor: The estimated fixed point.
        list: A list of relative residuals at each iteration.

    Example:
        >>> estimated_fixed_point, residuals = chebyshev_acceleration(f, x0, max_iter=100, tol=1e-3)
    """
    x = x0
    res = []
    for k in range(max_iter):
        x1 = f(x)
        x2 = f(x1)
        alpha = torch.norm(x1 - x2) ** 2 / (torch.norm(x2 - x) ** 2 + 1e-5)
        x_new = x2 + alpha * (x2 - x)
        res.append(torch.norm(x_new - x).item() / (1e-5 + torch.norm(x_new).item()))
        if res[-1] < tol:
            break
        x = x_new
    return x, res



def forward_iteration(f, x0, max_iter=50, tol=1e-2):
    """
    Iteratively applies a function to estimate its fixed point.

    This function iteratively applies the function `f` to estimate its fixed point starting from an initial guess `x0`.

    Args:
        f (callable): The function for which the fixed point is sought.
        x0 (torch.Tensor): The initial guess for the fixed point.
        max_iter (int, optional): The maximum number of iterations. Default is 50.
        tol (float, optional): The tolerance for convergence. Default is 1e-2.

    Returns:
        torch.Tensor: The estimated fixed point.
        list: A list of relative residuals at each iteration.

    Example:
        >>> estimated_fixed_point, residuals = forward_iteration(f, x0, max_iter=100, tol=1e-3)
    """
    f0 = f(x0)
    res = []
    for k in range(max_iter):
        x = f0
        f0 = f(x)
        res.append((f0 - x).norm().item() / (1e-5 + f0.norm().item()))
        if (res[-1] < tol):
            break
    return f0, res



X = torch.randn(10,64,32,32)
f = ResNetLayer(64,128)


"""
Z, res = chebyshev_acceleration(lambda Z : f(Z,X), torch.zeros_like(X), tol=1e-4)
plt.figure(dpi=150)
plt.semilogy(res)
plt.xlabel("Iteration")
plt.ylabel("Relative residual")

plt.show()

Z, res = forward_iteration(lambda Z : f(Z,X), torch.zeros_like(X), tol=1e-4)
plt.figure(dpi=150)
plt.semilogy(res)
plt.xlabel("Iteration")
plt.ylabel("Relative residual")
plt.show()
"""




class DEQFixedPoint(nn.Module):
    """
    A module for solving a fixed-point equation using differential equation solvers.

    This class represents a module that aims to find the fixed point of a function `f` by iteratively applying it.
    It utilizes a provided solver function to accelerate convergence.

    Args:
        f (callable): The function for which the fixed point is sought.
        solver (callable): The solver function used to accelerate convergence.
        **kwargs: Additional keyword arguments to be passed to the solver function.

    Attributes:
        f (callable): The function for which the fixed point is sought.
        solver (callable): The solver function used to accelerate convergence.
        kwargs (dict): Additional keyword arguments for the solver.

    Methods:
        forward(x): Compute the fixed point using the provided solver.

    Example:
        >>> f = ResNetLayer(64, 128)
        >>> deq = DEQFixedPoint(f, chebyshev_acceleration, tol=1e-10, max_iter=500)
        >>> estimated_fixed_point = deq(torch.randn(1, 64, 32, 32))
    """

    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs
        
    def forward(self, x):
        """
        Compute the fixed point of the function `f` using the provided solver.

        Args:
            x (torch.Tensor): The input to the function `f`.

        Returns:
            torch.Tensor: The estimated fixed point.

        Example:
            >>> estimated_fixed_point = deq(torch.randn(1, 64, 32, 32))
        """
        # Compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z : self.f(z, x), torch.zeros_like(x), **self.kwargs)
        z = self.f(z, x)
        
        # Set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)
        
        def backward_hook(grad_output):
            g, self.backward_res = self.solver(
                lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad_output,
                grad_output,
                **self.kwargs)
            return g
        
        # Apply the backward_hook to z to handle gradients properly
        z.register_hook(backward_hook)
        
        return z

from torch.autograd import gradcheck
# run a very small network with double precision, iterating to high precision
f = ResNetLayer(2,2, num_groups=2).double()
deq = DEQFixedPoint(f, chebyshev_acceleration, tol=1e-10, max_iter=500).double()
print(gradcheck(deq, torch.randn(1,2,3,3).double().requires_grad_(), eps=1e-5, atol=1e-3, check_undefined_grad=False))



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
chan = 48
f = ResNetLayer(chan, 64, kernel_size=3)
model = nn.Sequential(nn.Conv2d(1,chan, kernel_size=3, bias=True, padding=1),
                      nn.BatchNorm2d(chan),
                      DEQFixedPoint(f, chebyshev_acceleration, tol=1e-2, max_iter=25),
                      nn.BatchNorm2d(chan),
                      nn.AvgPool2d(8,8),
                      nn.Flatten(),
                      nn.Linear(432,10)).to(device)



from torchvision import datasets, transforms
from torch.utils.data import DataLoader

mnist_train = datasets.MNIST(".", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(".", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 10, shuffle=True, num_workers=0)
test_loader = DataLoader(mnist_test, batch_size = 10, shuffle=False, num_workers=0)

# standard training or evaluation loop
def epoch(loader, model, opt=None, lr_scheduler=None):
    """
    Run one epoch of training or evaluation on a given dataset.

    This function performs one pass (epoch) through a dataset, either for training or evaluation.
    It computes the loss, accuracy, and updates the model's weights if an optimizer is provided.

    Args:
        loader (torch.utils.data.DataLoader): The data loader for the dataset.
        model (torch.nn.Module): The neural network model.
        opt (torch.optim.Optimizer, optional): The optimizer for updating model weights during training.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler for optimizer.

    Returns:
        tuple: A tuple containing the accuracy and average loss for the epoch.

    Example:
        >>> train_accuracy, train_loss = epoch(train_loader, model, opt, scheduler)
        >>> test_accuracy, test_loss = epoch(test_loader, model)
    """
    total_loss, total_accuracy = 0., 0.
    model.eval() if opt is None else model.train()
    correct_predictions = 0
    total_samples = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()

        _, predicted = torch.max(yp, 1)
        correct_predictions += (predicted == y).sum().item()
        total_samples += y.size(0)

        total_loss += loss.item() * X.shape[0]

    accuracy = correct_predictions / total_samples
    return accuracy, total_loss / len(loader.dataset)

import torch.optim as optim
opt = optim.Adam(model.parameters(), lr=1e-3)
print("# Parmeters: ", sum(a.numel() for a in model.parameters()))

max_epochs = 50
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs*len(train_loader), eta_min=1e-6)

for i in tqdm(range(50)):
    print("train_accuracy, train_loss : ")
    print(epoch(train_loader, model, opt, scheduler))
    print("test_accuracy, test_loss : ")
    print(epoch(test_loader, model)) 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd as autograd
from tqdm import tqdm
import scipy.optimize as op
class ResNetLayer(nn.Module):
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
        y = self.norm1(F.relu(self.conv1(z)))
        return self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))
    
def chebyshev_acceleration(f, x0, max_iter=50, tol=1e-2):
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
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs
        
    def forward(self, x):
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
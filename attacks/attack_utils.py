# import apex.amp as amp
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.autograd import Variable
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)
# mnist_mean = (0.1307,)
# mnist_std = (0.3081,)

# mu = torch.tensor(mnist_mean).view(1,1,1).cuda()
# std = torch.tensor(mnist_std).view(1,1,1).cuda()

# upper_limit = ((1 - mu)/ std)
# lower_limit = ((0 - mu)/ std)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


# def get_loaders(dir_, batch_size):
#     train_transform = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(cifar10_mean, cifar10_std),
#     ])
#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(cifar10_mean, cifar10_std),
#     ])
#     num_workers = 2
#     train_dataset = datasets.CIFAR10(
#         dir_, train=True, transform=train_transform, download=True)
#     test_dataset = datasets.CIFAR10(
#         dir_, train=False, transform=test_transform, download=True)
#     train_loader = torch.utils.data.DataLoader(
#         dataset=train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         pin_memory=True,
#         num_workers=num_workers,
#     )
#     test_loader = torch.utils.data.DataLoader(
#         dataset=test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         pin_memory=True,
#         num_workers=2,
#     )
#     return train_loader, test_loader


def attack_pgd(model, X, y, epsilon=0, alpha=0, attack_iters=0, restarts=0, opt=None):
    # epsilon = (8 / 255.)-mean / std
    # alpha = (2 / 255.) / std
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            # if opt is not None:
            #     with amp.scale_loss(loss, opt) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts):
    print("new")
    epsilon = (8 / 255.)-mu / std
    alpha = (2 / 255.)-mu / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pred = torch.max(output, dim=1)[1]
            pgd_acc += (pred == y).sum().item()
            n += y.size(0)
    return pgd_loss/n, pgd_acc/n


def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n
def GA_PGD(model, data, target, epsilon, step_size, num_steps,rand_init):
    model.eval()
    Kappa = torch.zeros(len(data))

    x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    print(x_adv.shape)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        print(output.shape)
        predict = output.max(1, keepdim=True)[1]
        # Update Kappa
        for p in range(len(x_adv)):
            if predict[p] == target[p]:
                Kappa[p] += 1
        model.zero_grad()
        with torch.enable_grad():
           
            loss_adv = torch.nn.CrossEntropyLoss(reduction="mean")(output, target)
            
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = Variable(x_adv, requires_grad=False)
    return x_adv, Kappa

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"
sys.path.append("../")
from basic_models.resnet import *
from basic_models.vgg import *
from basic_models.lenet import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cinic_directory = '../data/zero-shot/data/CINIC-10'
cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]
# cinic_train = torch.utils.data.DataLoader(
#     torchvision.datasets.ImageFolder(cinic_directory + '/train',
#     	transform=transforms.Compose([transforms.ToTensor(),
#         transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
#     batch_size=128, shuffle=True)

cinic_test = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(cinic_directory + '/test',
    	transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
    batch_size=256, shuffle=True)
        
# weight = "../weights/cifar10/worse_label/supcontrast/resnet34/1_0.5_lr_splitby60/clean_sup_best.pth"
weight = "../weights/clean_ot_param/vgg16/best.pth"
# weight = "../weights/only_right/worse/round_199.pth"
# weight = "../weights/cifar10/clean/supcontrast/resnet34/clean_sup_best.pth"
encoder = vgg16()
encoder = encoder.to(device)
encoder.load_state_dict(torch.load(weight, map_location=device))
print('Loaded checkpoint successfully.')

# Freeze earlier layers
# for param in encoder.parameters():
#     param.requires_grad = False

# print(encoder)
# Modify the last fully connected layer
# num_features = encoder.linear.in_features
# print(num_features)
# encoder.linear = nn.Linear(num_features, 10)  # Assuming there are 10 classes

encoder = encoder.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(encoder.linear.parameters(), lr=0.1, momentum=0.9)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
encoder.eval()
for epoch in range(1):  # Adjust the number of epochs as needed
    running_loss = 0.0
    correct = 0
    total = 0
    # for i, data in enumerate(cinic_train, 0):
    #     inputs, labels = data[0].to(device), data[1].to(device)

    #     optimizer.zero_grad()

    #     outputs = encoder(inputs)
    #     loss = criterion(outputs, labels)
    #     loss.backward()
    #     optimizer.step()

    #     running_loss += loss.item()
    #     _, predicted = torch.max(outputs.data, 1)
    #     total += labels.size(0)
    #     correct += (predicted == labels).sum().item()

    #     if i % 100 == 99:
    #         print(f'Epoch: {epoch+1}, Batch: {i+1}, Loss: {running_loss/100:.3f}, Train Accuracy: {100 * correct / total:.2f}%')
    #         running_loss = 0.0
    #         correct = 0
    #         total = 0

    # Learning rate scheduler step
    # scheduler.step()

    # Evaluation on test set
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data in cinic_test:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = encoder(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    print(f'Epoch: {epoch+1}, Test Loss: {test_loss/len(cinic_test):.3f}, Test Accuracy: {100 * test_correct / test_total:.2f}%')

# Save the fine-tuned model
# torch.save(encoder.state_dict(), './fine_tuned_baseline_CINIC.pth')
# print('Fine-tuned model saved successfully.')

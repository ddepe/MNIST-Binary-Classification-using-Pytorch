"""
Linear SVM approach to MNIST Binary classification
Using Pytorch and a custom Hinge Loss
"""
# Don't change batch size
batch_size = 64

from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

dtype = torch.float
device = torch.device("cpu")

## USE THIS SNIPPET TO GET BINARY TRAIN/TEST DATA

train_data = datasets.MNIST('./pytorch/data/', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
test_data = datasets.MNIST('./pytorch/data/', train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
subset_indices = ((train_data.train_labels == 0) + (train_data.train_labels == 1)).nonzero().view(-1)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False,
                                           sampler=SubsetRandomSampler(subset_indices))

subset_indices = ((test_data.test_labels == 0) + (test_data.test_labels == 1)).nonzero().view(-1)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                          sampler=SubsetRandomSampler(subset_indices))


class HingeLoss(torch.nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, inputs, target):
        L = (1 - target * inputs).clamp(min=0)
        return torch.mean(L)


lr_list = [1e-2]
accuracy_list = []
loss_list = []
with open('losses.txt', 'w') as f:
    for learning_rate in lr_list:
        # learning_rate = 1e-1
        model = nn.Linear(28 * 28, 1)
        # criterion = nn.MarginRankingLoss(margin=1.0)
        criterion = HingeLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        # Training the Model
        # Notice that newest Pytorch merge tensor and variable, so the additional Variable wrapping is no longer required.
        num_epochs = 10
        epoch_loss = []
        for epoch in range(num_epochs):
            total_loss = 0
            # scheduler.step()
            for i, (images, labels) in enumerate(train_loader):
                images = Variable(images.view(-1, 28 * 28))  # 64x784
                # Convert labels from 0,1 to -1,1
                labels = Variable(2 * (labels.float() - 0.5))  # 64x1

                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                # For using built-in hinge loss, nn.MarginRankingLoss()
                # loss = criterion(outputs.t(), torch.zeros(outputs.size(1), 1, device=device, dtype=dtype,
                #                                           requires_grad=True), labels)
                loss = criterion(outputs.t(), labels)

                # Backward pass + Optimize
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print('Epoch: [% d/% d], Loss: %.4f'
                  % (epoch + 1, num_epochs, total_loss/198))
            # Print your results every epoch
            epoch_loss.append(total_loss/198)
            f.write("%f\t" % (total_loss/198))

            if total_loss/198 < 0.000001:
                break

            # Test the Model
            correct = 0.
            total = 0.
            for images, labels in test_loader:
                images = Variable(images.view(-1, 28 * 28))

                labels_test_pred = model(images)
                prediction = labels_test_pred.data.sign() / 2 + 0.5

                correct += (prediction.view(-1).long() == labels).sum()
                total += images.shape[0]

            print('Accuracy of the LinearSVM SGD model with step size %f on the test images: %f %%' % (learning_rate, 100 * (correct.float() / total)))
            accuracy_list.append(correct.float() / total)
            loss_list.append(epoch_loss)

            f.write("\n")
            f.write('Accuracy of the LinearSVM SGD model with step size %f on the test images: \t%f %%' % (learning_rate, 100 * (correct.float() / total)))





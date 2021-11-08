'''
This is Jinfei's Note for the DataCamp course: Introduction to Deep Learning with PyTorch

'''

#####################################
#       CH1: INTRO TO PYTORCH       #
#####################################


# Creating tensors in PyTorch
# Import torch
import torch

# Create random tensor of size 3 by 3
your_first_tensor = torch.rand(3, 3)

# Calculate the shape of the tensor
tensor_size = your_first_tensor.size()

# Print the values of the tensor and its shape
print(your_first_tensor)
print(tensor_size)


# Matrix Multiplication
# Create a matrix of ones with shape 3 by 3
tensor_of_ones = torch.ones(3, 3)

# Create an identity matrix with shape 3 by 3
identity_tensor = torch.eye(3)

# Do a matrix multiplication of tensor_of_ones with identity_tensor
matrices_multiplied = torch.matmul(tensor_of_ones, identity_tensor)
print(matrices_multiplied)

# Do an element-wise multiplication of tensor_of_ones with identity_tensor
element_multiplication = tensor_of_ones * identity_tensor
print(element_multiplication)

# Forward pass
# Initialize tensors x, y and z
x = torch.rand(1000, 1000)
y = torch.rand(1000, 1000)
z = torch.rand(1000, 1000)

# Multiply x with y
q = torch.matmul(x, y)

# Multiply elementwise z with q
f = z * q

mean_f = torch.mean(f)
print(mean_f)

# Back Propagation
# Initialize x, y and z to values 4, -3 and 5
x = torch.tensor(4., requires_grad=True)
y = torch.tensor(-3.,requires_grad=True)
z = torch.tensor(5., requires_grad=True)

# Set q to sum of x and y, set f to product of q with z
q = x + y
f = q * z

# Compute the derivatives
f.backward()

# Print the gradients
print("Gradient of x is: " + str(x.grad))
print("Gradient of y is: " + str(y.grad))
print("Gradient of z is: " + str(z.grad))


#Calculating Gradient
# Multiply tensors x and y
q = torch.matmul(x, y)

# Elementwise multiply tensors z with q
f = z * q

mean_f = torch.mean(f)

# Calculate the gradients
mean_f.backward()

# First Neural Network
# Initialize the weights of the neural network
weight_1 = torch.rand(784, 200)
weight_2 = torch.rand(200, 10)

# Multiply input_layer with weight_1
hidden_1 = torch.matmul(input_layer, weight_1)

# Multiply hidden_1 with weight_2
output_layer = torch.matmul(hidden_1, weight_2)
print(output_layer)

# First PyTorch Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Instantiate all 2 linear layers  
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
      
        # Use the instantiated layers and return x
        x = self.fc1(x)
        x = self.fc2(x)
        return x


#####################################
# CH2: Artificial Neural Networks   #
#####################################
'''
With just matrix multiplication, we can simplify any neural network
in a single layer network. -> neural network if not that powerful 
without activation function

Activation functions (non-linear):
- ReLU max(0, x)
- Sigmoid
- tanh
- ELU
- Maxout
- Leaky ReLU
'''
# Without Activation Function
# Calculate the first and second hidden layer
hidden_1 = torch.matmul(input_layer, weight_1)
hidden_2 = torch.matmul(hidden_1, weight_2)

# Calculate the output
print(torch.matmul(hidden_2, weight_3))

# Calculate weight_composed_1 and weight
weight_composed_1 = torch.matmul(weight_1,weight_2)
weight = torch.matmul(weight_composed_1, weight_3)

# Multiply input_layer with weight
print(torch.matmul(input_layer, weight))

#ReLU Activation
# Instantiate non-linearity
relu = nn.ReLU()

# Apply non-linearity on the hidden layers
hidden_1_activated = relu(torch.matmul(input_layer, weight_1))
hidden_2_activated = relu(torch.matmul(hidden_1_activated, weight_2))
print(torch.matmul(hidden_2_activated, weight_3))

# Apply non-linearity in the product of first two weights. 
weight_composed_1_activated = relu(torch.matmul(weight_1, weight_2))

# Multiply `weight_composed_1_activated` with `weight_3
weight = torch.matmul(weight_composed_1_activated, weight_3)

# Multiply input_layer with weight
print(torch.matmul(input_layer, weight))

# Instantiate ReLU activation function as relu
relu = nn.ReLU()

# Initialize weight_1 and weight_2 with random numbers
weight_1 = torch.rand(4, 6)
weight_2 = torch.rand(6, 2)

# Multiply input_layer with weight_1
hidden_1 = torch.matmul(input_layer, weight_1)

# Apply ReLU activation function over hidden_1 and multiply with weight_2
hidden_1_activated = relu(hidden_1)
print(torch.matmul(hidden_1_activated, weight_2))

'''
How to build neural network -> how to build them

Loss function: how wrong the model goes
Classification: softmax cross-entropy
More complicated questions -> more complicated loss function
Loss function should always be differentialble

'''
# Calculating loss function in PyTorch
# Initialize the scores and ground truth
logits = torch.tensor([[-1.2, 0.12, 4.8]])
ground_truth = torch.tensor([2])

# Instantiate cross entropy loss
criterion = nn.CrossEntropyLoss()

# Compute and print the loss
loss = criterion(logits, ground_truth)
print(loss)

# Loss function with random scores
# Import torch and torch.nn
import torch
import torch.nn as nn

# Initialize logits and ground truth
logits = torch.rand(1, 1000)
ground_truth = torch.tensor([111])

# Instantiate cross-entropy loss
criterion = nn.CrossEntropyLoss()

# Calculate and print the loss
loss = criterion(logits, ground_truth)
print(loss)

'''
Preparing a dataset in PyTorch
MINST: digits
CIFAR-10: natural images
'''
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
# Define a tranformation of images to torch tensors
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.48216, 0.44653),
                         (0.24703, 0.24349, 0.26159))]
    )
'''
batch_size = 32: 
The dataset is to large to use entirely,

'''
# Transform the data to torch tensors and normalize it 
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307), ((0.3081)))])

# Prepare training set and testing set
trainset = torchvision.datasets.MNIST('mnist', train=True, 
                                      download=True, transform=transform)
testset = torchvision.datasets.MNIST('mnist', train=False,
                download=True, transform=transform)

# Prepare training loader and testing loader
# Only the training dataset should be shuffled
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=0) 

# Compute the shape of the training set and testing set
trainset_shape = trainloader.dataset.train_data.shape
testset_shape = testloader.dataset.test_data.shape

# Print the computed shapes
print(trainset_shape, testset_shape)

# Compute the size of the minibatch for training set and testing set
trainset_batchsize = trainloader.batch_size
testset_batchsize = testloader.batch_size

# Print sizes of the minibatch
print(trainset_batchsize, testset_batchsize)

# Train Neural Network

'''
- Prepare the dataloaders
- Build a neural network
Loop over (repeat many times):
- Forward pass with Mini-batch
- Calculate loss function
- Calculate the gradients
- Change the weights based on gradients
(weight -= weight_gradient * learning_rate)

The learning rate should be neither too big (overshoot the minima) nor too small (slow).

<Gradient Descent>
By going in the direction of the gradient 
(subtracting the gradient of weights from weights), 
we are going in direction of the local minima of a function.
So we can minimize the Loss Function

Adam optimizer: a version of gradient descent - work well

'''

# Training the Neural Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

net = Net() # forward step
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=3e-4)

for epoch in range(10):  # loop over the dataset multiple times
    for i, data in enumerate(trainloader, 0):
        # Get the inputs
        inputs, labels = data
        inputs = inputs.view(-1, 32 * 32 * 3) # puts all entries of the images into vectors

        # Zero the parameter gradients
        # not accumulate gradients from the previous iteration
        optimizer.zero_grad() 

        # Forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step() #change the weight of optimizer

# Trained nets are used to make predictions on unseen images
correct, total = 0, 0
predictions = []
net.eval() # set the net in test(evaluation) mode
for i, data in enumerate(testloader, 0):
    inputs, labels = data
    inputs = inputs.view(-1, 32*32*3)
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    predictions.append(outputs)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print('The testing set accuracy of the network is: %d %%' % (100 * correct / total))

##############
# Exercise
##############
# Build a neural network - again
# Define the class Net
class Net(nn.Module):
    def __init__(self):    
        # Define all the parameters of the net
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28 * 1, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):    
        # Do the forward pass
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return 

# Instantiate the Adam optimizer and Cross-Entropy loss function
model = Net()   
optimizer = optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()
  
for batch_idx, data_target in enumerate(train_loader):
    data = data_target[0]
    target = data_target[1]
    data = data.view(-1, 28 * 28)
    optimizer.zero_grad()

    # Complete a forward pass
    output = model(data)

    # Compute the loss, gradients and change the weights
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# Set the model in eval mode
model.eval()

for i, data in enumerate(test_loader, 0):
    inputs, labels = data
    
    # Put each image into a vector
    inputs = inputs.view(-1, 28 * 28 *1)
    
    # Do the forward pass and get the predictions
    outputs = model(inputs)
    _, outputs = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (outputs == labels).sum().item()
# To decrease test error: larger dataset or deeper network
print('The testing set accuracy of the network is: %d %%' % (100 * correct / total))


##############################################
# CH3: Convolutional Neural Networks (CNN)   #
##############################################
'''
Fully-connected neural networks:
- Assume all relations between features
- big and very computationally inefficient
- so many parameters -> so overfit

CNNs:
- Units are connected with only a few units prom the previous layer
- Units share weights

Hyperparameter of CNN:
Filter/ Kernel Size: the size of the filter/kernel
Padding: adding 0 around the image
Stride: the number of pixel moved each time
'''

import torch
import torch.nn
image = torch.rand(16, 3, 32, 32)
conv_filter = torch.nn.Conv2d(in_channels=3,
             out_channels=1, kernel_size=5,
             stride=1, padding=0)
output_feature = conv_filter(image)
print(output_feature.shape)

'''
Two ways to do Convolutions in Pytorch
# OOP-based
- in_channels(int): number of channels in input
- out_channels(int): number of channels produced by the convolution
- kernel_size(int or tuple): size of the convolving kernel
- stride(int or tule, optional): default 1
- padding(int or tuple, optional): default 0
# Functional
'''
import torch
import torch.nn.functional as F
image = torch.rand(16, 3, 32, 32)
filter = torch.rand(1, 3, 5, 5)
out_feat_F = F.conv2d(image, filter,
                 stride=1, padding=0)
print(out_feat_F.shape)

# PRACTIC

# CONVOLUTION OPERATOR -OOP WAY
# Create 10 random images of shape (1, 28, 28)
images = torch.rand(10, 1, 28, 28)

# Build 6 conv. filters
# NOTE: kernel size is always a number -> for a k * k square
conv_filters = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)

# Convolve the image with the filters 
output_feature = conv_filters(images)
print(output_feature.shape)

# - Functional way
# Create 10 random images
image = torch.rand(10, 1, 28, 28)

# Create 6 filters
filters = torch.rand(6, 1, 3, 3)

# Convolve the image with the filters
output_feature = F.conv2d(image, filters, stride=1, padding=1)
print(output_feature.shape)

'''
Pooling operator -> Feature Selection
Choose the most dominant features from the image
or combine different features
- lower the resolution of the images
-> make computation more efficient
-> make the learning invariant to shifting and translation (more robost)

- Max pooling
(2*2) filters and stride 2

- Aveage pooling
Typically used in deep networks

'''
# Max pooling
# OOP
import torch
import torch.nn
im = torch.Tensor([[[[3, 1, 3, 5], [6, 0, 7, 9],
                   [3, 2, 1, 4], [0, 2, 4, 3]]]])
max_pooling = torch.nn.MaxPool2d(2)
output_feature = max_pooling(im)
print(output_feature)

# Functional
import torch
import torch.nn.functional as F
im = torch.Tensor([[[[3, 1, 3, 5], [6, 0, 7, 9],
                   [3, 2, 1, 4], [0, 2, 4, 3]]]])
output_feature_F = F.max_pool2d(im, 2)
print(output_feature_F)

# Average Pooling
# OOP
import torch
import torch.nn
im = torch.Tensor([[[[3, 1, 3, 5], [6, 0, 7, 9],
                   [3, 2, 1, 4], [0, 2, 4, 3]]]])
avg_pooling = torch.nn.AvgPool2d(2)
output_feature = avg_pooling(im)
print(output_feature)

# Functional way
import torch
import torch.nn.functional as F
im = torch.Tensor([[[[3, 1, 3, 5], [6, 0, 7, 9],
                   [3, 2, 1, 4], [0, 2, 4, 3]]]])
output_feature_F = F.avg_pool2d(im, 2)
print(output_feature_F)

### PRACTICE

# Build a pooling operator with size `2`.
max_pooling = torch.nn.MaxPool2d(2)

# Apply the pooling operator
output_feature = max_pooling(im)

# Use pooling operator in the image
output_feature_F = F.max_pool2d(im, 2)

# print the results of both cases
print(output_feature)
print(output_feature_F)

# Build a pooling operator with size `2`.
avg_pooling = torch.nn.AvgPool2d(2)

# Apply the pooling operator
output_feature = avg_pooling(im)

# Use pooling operator in the image
output_feature_F = F.avg_pool2d(im, 2)

# print the results of both cases
print(output_feature)
print(output_feature_F)

'''
Almost everything in computer vision is empowered by CNNs
If not, they at least play a large part on it
Detection, segmentation, recognition, autonomous driving
AlphaGo, Starcraft Zero - you name it.
'''
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        # call the superclass use super operator
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        # Define relu only once
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # Though we have three avgpool layer, they have the same kernel_size and stride
        # so we define it only once
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
net = AlexNet()

# Your first CNN __init__ method
# Wrong Answer
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Instantiate two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)
        
        # Instantiate the ReLU nonlinearity
        self.relu = nn.ReLU()
        
        # Instantiate a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Instantiate a fully connected layer
        # ATTENTION: THE FIRST ARGUMENT OF nn.Linear is num_channels*height*weight
        self.fc = nn.Linear(10, 10)
'''
Hint
Deduct the first size of the weights for the fully connected layers. Images start with shape (1, 28, 28) and two pooling operators (each halving the size of the image) are performed. What is the size of the image fed to the input layer (heigh * width * number_of_channels)?
In line 16, number_of_channels is the same as the number of channels in self.conv2.
MNIST images are black and white, so they contain one channel.
'''
# CORRECT ANSWER
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Instantiate two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)
        
        # Instantiate the ReLU nonlinearity
        self.relu = nn.ReLU()
        
        # Instantiate a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Instantiate a fully connected layer
        self.fc = nn.Linear(7 * 7 * 10, 10)

    def forward(self, x):

        # Apply conv followd by relu, then in next line pool
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # Apply conv followd by relu, then in next line pool
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Prepare the image for the fully connected layer
        x = x.view(-1, 7 * 7 * 10)

        # Apply the fully connected layer and return the result
        return self.fc(x)

'''
Training CNNs

Compared to fully connected layer (accuracy usually < 60%)
it is possible to train CNN to get 99% accuracy

'''
# import
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# dataloader
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)
# building a CNN
class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 4 * 4, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        return self.fc(x)

# Optimizer and Loss Function
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=3e-4)

# Train a CNN
for epoch in range(10):
    for i, data in enumerate(trainloader, 0):
        # Get the inputs
        inputs, labels = data
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
print('Finished Training')

# Evaluating the results
correct, total = 0, 0
predictions = []
net.eval()
for i, data in enumerate(testloader, 0):
    inputs, labels = data
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    predictions.append(outputs)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print('The testing set accuracy of the network is: %d %%' % (
        100 * correct / total))


#PRACTICE
for i, data in enumerate(train_loader, 0):
    inputs, labels = data
    optimizer.zero_grad()

    # Compute the forward pass
    outputs = net(inputs)
        
    # Compute the loss function
    loss = criterion(outputs, labels)
        
    # Compute the gradients
    loss.backward()
    
    # Update the weights
    optimizer.step()

    # Iterate over the data in the test_loader
for i, data in enumerate(test_loader, 0): #or enumerate(test_loader)

    # Get the image and label from data
    image, label = data

    # Make a forward pass in the net with your image
    output = net(image)

    # Argmax the results of the net
    _, predicted = torch.max(output.data, 1)
    if predicted == label:
        print("Yipes, your net made the right prediction " + str(predicted))
    else:
        print("Your net prediction was " + str(predicted) + ", but the correct label is: " + str(label))

##############################################
#               CH4: Using CNNs              #
##############################################

# Using Sequentials

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Declare all the layers for feature extraction
        self.features = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1), 
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1), 
                                      nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1),
                                      nn.MaxPool2d(2, 2), nn.ReLU(inplace=True))
        
        # Declare all the layers for classification
        self.classifier = nn.Sequential(nn.Linear(7 * 7 * 40, 1024), nn.ReLU(inplace=True),
                                        nn.Linear(1024, 2048), nn.ReLU(inplace=True),
                                        nn.Linear(2048, 10))
    def forward(self, x):
      
        # Apply the feature extractor in the input
        x = self.features(x)
        
        # Squeeze the three spatial dimensions in one
        x = x.view(-1, 7 * 7 * 40)
        
        # Classify the images
        x = self.classifier(x)
        return x

'''
Overfitting in the testing set
- Training set: train the model
- Validation set: selection the model (aka corss validation)
- Testing set: test the model

Validation set is to avoid test set been contaminated.

The networks are trained in the training set as before, 
and each of them is tested in the validation set.
Finally, the best performance model is tested in the testing set.

It is important the the testing set is used only once.
Otherwise, its result won't be trustworthy.
Also, train, validation, test set should not overlap.

To detect overfitting, we should use the error rate 
between training set and validation set.

'''

indices = np.arange(50000)
np.random.shuffle(indices)
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),
                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])),
    batch_size=1, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[:45000]))
val_loader = torch.utils.data.DataLoader(
                    datasets.CIFAR10(root='./data', train=True, download=True,
                                    transform=transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])), 
                    batch_size=1, 
                    shuffle=False, 
                    sampler=torch.utils.data.SubsetRandomSampler(indices[45000:50000]))

# PRACTICE
# Shuffle the indices
indices = np.arange(60000)
np.random.shuffle(indices)

# Build the train loader
train_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist', download=True, train=True,
                     transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                     batch_size=64, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[:55000]))

# Build the validation loader
val_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist', download=True, train=True,
                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                   batch_size=64, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[55000:60000]))

'''
Techniques to make training more efficiently with better predictions
- L2 Regularization
- dropout parameter
- optimizers (Adam vs gradient descent etc)
- batch-norm momentum and epsilon
- number of epochs for early stopping

Question: how to choose from those above hyperparameters?

Answer: Train many networks with different hyperparameters (typically with random values)
        and test them in the validation set. Then use the best performing net in the validation set
        to know the expected accuracy of the network in new data.

'''

'''
1. L2 regularization
- also used in algorithms like regression or SVM
- add a L2 regularization penalty term to the loss function
- PyTorch: add the weight_decay argument in the optimizer
'''
optimizer = optim.Adam(net.parameters(), lr=3e-4, weight_decay=0.0001)

'''
2. Dropout
- change the archetecture randomly to avoid dependency
- usually in full-connected layer instead of CNNs
- Parameter: p -> control the probability of units being droped
'''
self.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(256 * 6 * 6, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, num_classes),
)
'''
3. Batch Normalization
- compute the mean and variance of the minibatch for each feature
  and then it normalizes the features based on mean/var
- a must have for large neural networks nowaday
'''
self.bn = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.9)

'''
4. Early stopping
- Check the accuracy of the network in the validation set
  at the end each epoch
- if after n epochs the performance of the net hasn't increased
  (or it has decreased), then training is terminated

'''
# Train mode
model.train()
# Eval() mode
model.eval()

###### Practice

# L2-regularization

# Instantiate the network
model = Net()

# Instantiate the cross-entropy loss
criterion = nn.CrossEntropyLoss()

# Instantiate the Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.001)

# Dropout
class Net(nn.Module):
    def __init__(self):
        
        # Define all the parameters of the net
        self.classifier = nn.Sequential(
            nn.Linear(28*28, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(200, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10))
        
    def forward(self, x):
    
        # Do the forward pass
        return self.classifier(x)

# Batch Normalization
# NOTE: BatchNorm2d modules require the number of channels as argument
# Should BatchNorm before or after the activation function?
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Implement the sequential module for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2), 
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(10),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2), 
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(20))
        
        # Implement the fully connected layer for classification
        self.fc = nn.Linear(in_features=7*7*20, out_features=10)

'''
Transfer learning

Retraining / finetuning
With the layers of other pre-trained net such as ImageNet,
We can achieve good model on very small datasets.

Methods of finetuning:
1. Freeze most of the layers: not update them during back-propagation
   and fine tune only the last few layers (or only the last one)
2. Finetune everything (when dataset is larger)   

Typically, when dataset is small, it's better to freeze to avoid overfitting.

pre-trained model format: cifar10_net.pth
'''

# Fine tuning the CNNs
# Instantiate the model
model = Net()
# Load the parameters from the old model
model.load_state_dict(torch.load('cifar10_net.pth'))
# Change the number of out channels
model.fc = nn.Linear(4 * 4 * 1024, 100)
# Train and evaluate the model
model.train()

# Freeze the layer: set param.requires_grad = False 
# Instantiate the model
model = Net()
# Load the parameters from the old model
model.load_state_dict(torch.load('cifar10_net.pth'))
# Freeze all the layers bar the final one
for param in model.parameters():
    param.requires_grad = False
# Change the number of output units
model.fc = nn.Linear(4 * 4 * 1024, 100)
# Train and evaluate the model
model.train()

# Torch Vision Library
import torchvision
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(512, num_classes)



# Practice
# Create a new model
model = Net()

# Change the number of out channels
model.fc = nn.Linear(7 * 7 * 512, 26)

# Train and evaluate the model
model.train()
train_net(model, optimizer, criterion)
print("Accuracy of the net is: " + str(model.eval()))

# Create a model using
model = Net()

# Load the parameters from the old model
model.load_state_dict(torch.load('my_net.pth'))

# Change the number of out channels
model.fc = nn.Linear(7 * 7 * 512, 26)

# Train and evaluate the model
model.train()
train_net(model, optimizer, criterion)
print("Accuracy of the net is: " + str(model.eval()))

# Import the module
import torchvision

# Download resnet18
model = torchvision.models.resnet18(pretrained=True)

# Freeze all the layers bar the last one
for param in model.parameters():
    param.requires_grad = False

# Change the number of output units
model.fc = nn.Linear(512, 7)

from torch import nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, n_input_channels=3, n_output=10):
        super().__init__()

        ################################################################################
        # TODO:                                                                        #
        # Define 2 or more different layers of the neural network                      #
        ################################################################################

        # Channel/feature sizes
        conv1_in, conv1_out = n_input_channels, 12
        conv2_in, conv2_out = conv1_out, 24
        self.fc1_in, fc1_out = (conv2_out * 8 * 8), 80
        fc2_in, fc2_out = fc1_out, n_output

        self.conv1 = nn.Conv2d(in_channels=conv1_in, out_channels=conv1_out, kernel_size=5, padding=2)
        self.conv1_bn = nn.BatchNorm2d(conv1_out)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=conv2_in, out_channels=conv2_out, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(conv2_out)

        self.fc1 = nn.Linear(in_features=self.fc1_in, out_features=fc1_out)
        self.fc1_bn = nn.BatchNorm1d(fc1_out)

        self.fc2 = nn.Linear(in_features=fc2_in, out_features=fc2_out)

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

    def forward(self, x):
        ################################################################################
        # TODO:                                                                        #
        # Set up the forward pass that the input data will go through.                 #
        # A good activation function betweent the layers is a ReLu function.           #
        #                                                                              #
        # Note that the output of the last convolution layer should be flattened       #
        # before being inputted to the fully connected layer. We can flatten           #
        # Tensor `x` with `x.view`.                                                    #
        ################################################################################
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.maxpool(x)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.maxpool(x)
        x = x.view(-1, self.fc1_in)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

        return x

    def predict(self, x):
        logits = self.forward(x)
        return F.softmax(logits)


class ConvNet_Type2(nn.Module):
    # Tested with lr = 1e-3, l2_reg = 1e-2
    def __init__(self, n_input_channels=3, n_output=10):
        super().__init__()

        ################################################################################
        # TODO:                                                                        #
        # Define 2 or more different layers of the neural network                      #
        ################################################################################

        self.conv1 = nn.Conv2d(in_channels=n_input_channels, out_channels=24, kernel_size=5, padding=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=(12 * 8 * 8), out_features=96)
        self.fc2 = nn.Linear(in_features=96, out_features=n_output)

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

    def forward(self, x):
        ################################################################################
        # TODO:                                                                        #
        # Set up the forward pass that the input data will go through.                 #
        # A good activation function betweent the layers is a ReLu function.           #
        #                                                                              #
        # Note that the output of the last convolution layer should be flattened       #
        # before being inputted to the fully connected layer. We can flatten           #
        # Tensor `x` with `x.view`.                                                    #
        ################################################################################
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, 12 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

        return x

    def predict(self, x):
        logits = self.forward(x)
        return F.softmax(logits)


class ConvNet_Type3(nn.Module):
    # Tested with lr = 1e-3, l2_reg = 1e-2
    def __init__(self, n_input_channels=3, n_output=10):
        super().__init__()

        ################################################################################
        # TODO:                                                                        #
        # Define 2 or more different layers of the neural network                      #
        ################################################################################

        self.conv1 = nn.Conv2d(in_channels=n_input_channels, out_channels=48, kernel_size=5, padding=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=36, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=(36 * 8 * 8), out_features=288)
        self.fc2 = nn.Linear(in_features=288, out_features=n_output)

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

    def forward(self, x):
        ################################################################################
        # TODO:                                                                        #
        # Set up the forward pass that the input data will go through.                 #
        # A good activation function betweent the layers is a ReLu function.           #
        #                                                                              #
        # Note that the output of the last convolution layer should be flattened       #
        # before being inputted to the fully connected layer. We can flatten           #
        # Tensor `x` with `x.view`.                                                    #
        ################################################################################
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, 36 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

        return x

    def predict(self, x):
        logits = self.forward(x)
        return F.softmax(logits)
    
    

class ConvNet_Type4(nn.Module):
    # Tested with lr = 1e-3, l2_reg = 1e-2
    def __init__(self, n_input_channels=3, n_output=10):
        super().__init__()

        ################################################################################
        # TODO:                                                                        #
        # Define 2 or more different layers of the neural network                      #
        ################################################################################

        self.conv1 = nn.Conv2d(in_channels=n_input_channels, out_channels=32, kernel_size=5, padding=2)
        self.conv1_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=(64 * 8 * 8), out_features=96)
        self.fc2 = nn.Linear(in_features=96, out_features=n_output)

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

    def forward(self, x):
        ################################################################################
        # TODO:                                                                        #
        # Set up the forward pass that the input data will go through.                 #
        # A good activation function betweent the layers is a ReLu function.           #
        #                                                                              #
        # Note that the output of the last convolution layer should be flattened       #
        # before being inputted to the fully connected layer. We can flatten           #
        # Tensor `x` with `x.view`.                                                    #
        ################################################################################
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1_1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2_1(x))
        x = self.maxpool(x)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

        return x

    def predict(self, x):
        logits = self.forward(x)
        return F.softmax(logits)
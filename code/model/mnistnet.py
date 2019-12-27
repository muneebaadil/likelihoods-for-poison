import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):

    def __init__(self):
        super(MNISTNet, self).__init__()

        ## formula to make sure params work out
        ## W_out = (W_in - F)/S + 1
        ## from http://cs231n.github.io/convolutional-networks/

        ## we will have 2 conv layers

        # in_channel is 1 bcz MNIST is a BW dataset
        # from 1 input, we create 6 output feature maps, using 5x5 kernels
        # input is 28x28, output will be 24x24
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)

        # max pool over the 2x2 region.
        # input is 24x24, output will be 12x12
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # after max pool, out spatial dim is reduced, but num channels coming from
        # prev conv1 is still 6. We create 16 feature maps, again using 5x5 kernels
        # input is 12x12, output will be 8x8
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # max pool over the 2x2 region.
        # input is 8x8, output will be 4x4
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        ## followed by 2 FC layers, y = Wx + b
        # we take features computer after the pool2 step.
        # we have 16 maps, each of size 4x4. We will flatten them
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)

        ## and a last FC layer for predictions
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    # we define the fwd pass, backward pass is automatically defined
    # by PyTorch thru lineage

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # flatten
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.tanh(self.fc1(x))
        # x = F.tanh(self.fc2(x))
        x = self.fc3(x)

        return x
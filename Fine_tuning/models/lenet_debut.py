import torch
import torch.nn as nn
import torch.nn.functional as F

from debut.debut import DeBut
from debut.debut_conv import DeBut_2dConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeNet_DeBut(nn.Module):
    def __init__(self, R_shapes, DeBut_layer, **kwargs):
        super(LeNet_DeBut, self).__init__()

        # Original
        self.DeBut_layer = DeBut_layer
        self.R_shapes = R_shapes
        self.conv1 = nn.Conv2d(1, 8, 3)
        if 0 in self.DeBut_layer:
            self.conv2 = DeBut_2dConv(8, 16, 3, R_shapes = self.R_shapes[0],
                            bias=True, return_intermediates = False)
        else:
            self.conv2 = nn.Conv2d(8, 16, 3)
        
        if 1 in self.DeBut_layer:
            self.fc = DeBut(400, 128, R_shapes = self.R_shapes[1],
                            param = 'regular', bias=True, **kwargs)
        else:
            self.fc = nn.Linear(400, 128)
        
        if 2 in self.DeBut_layer:
            self.fc2 = DeBut(128, 64, R_shapes = self.R_shapes[2], 
                            param = 'regular', bias=True, **kwargs)
        else:
            self.fc2 = nn.Linear(128, 64)
        
        self.fc3   = nn.Linear(64, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.fc(out))

        out = self.fc3((self.fc2(out)))
        return out



import torch.nn as nn
from collections import OrderedDict
from debut.debut_conv import DeBut_2dConv
from debut.debut import DeBut


defaultcfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]

class VGG_DeBut(nn.Module):
    def __init__(self, R_shapes, DeBut_layer, cfg=None, num_classes=10, **kwargs):
        super(VGG_DeBut, self).__init__()

        if cfg is None:
            cfg = defaultcfg

        self.R_shapes = R_shapes
        self.DeBut_layer = DeBut_layer

        self.features = self._make_layers(cfg)
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cfg[-2], cfg[-1])),
            ('norm1', nn.BatchNorm1d(cfg[-1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(cfg[-1], num_classes)),
        ]))

    def _make_layers(self, cfg, **kwargs):
        print(self.DeBut_layer)
        layers = nn.Sequential()
        in_channels = 3
        r_shape_i = 0

        for i, x in enumerate(cfg):
            if x == 'M':
                layers.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                if i in self.DeBut_layer:
                    input_channel = cfg[i - 1] if cfg[i - 1] != 'M' else cfg[i]//2
                    output_channel = cfg[i]
                    if i == 14:
                        input_channel = cfg[i]
                    debut = DeBut_2dConv(input_channel, output_channel, 3, padding = 1, R_shapes = self.R_shapes[r_shape_i], bias=True, return_intermediates = False, **kwargs)
                    layers.add_module('debut%d' % i, debut)
                    layers.add_module('norm%d' % i, nn.BatchNorm2d(x))
                    layers.add_module('relu%d' % i, nn.ReLU(inplace=True))

                    r_shape_i += 1
                else: 
                    conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                    layers.add_module('conv%d' % i, conv2d)
                    layers.add_module('norm%d' % i, nn.BatchNorm2d(x))
                    layers.add_module('relu%d' % i, nn.ReLU(inplace=True))
                in_channels = x

        return layers

    def forward(self, x):
        x = self.features(x)

        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

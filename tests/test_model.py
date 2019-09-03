from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet


# -- fixtures -------------------------------------------------------------------------------------

@pytest.fixture(scope='module', params=[x for x in range(4)])
def model(request):
    return 'efficientnet-b{}'.format(request.param)


@pytest.fixture(scope='module', params=[True, False])
def pretrained(request):
    return request.param


@pytest.fixture(scope='function')
def net(model, pretrained):
    return EfficientNet.from_pretrained(model) if pretrained else EfficientNet.from_name(model)


# -- tests ----------------------------------------------------------------------------------------

@pytest.mark.parametrize('img_size', [224, 256, 512])
def test_forward(net, img_size):
    """Test `.forward()` doesn't throw an error"""
    data = torch.zeros((1, 3, img_size, img_size))
    output = net(data)
    assert not torch.isnan(output).any()


def test_dropout_training(net):
    """Test dropout `.training` is set by `.train()` on parent `nn.module`"""
    net.train()
    assert net._dropout.training == True


def test_dropout_eval(net):
    """Test dropout `.training` is set by `.eval()` on parent `nn.module`"""
    net.eval()
    assert net._dropout.training == False


def test_dropout_update(net):
    """Test dropout `.training` is updated by `.train()` and `.eval()` on parent `nn.module`"""
    net.train()
    assert net._dropout.training == True
    net.eval()
    assert net._dropout.training == False
    net.train()
    assert net._dropout.training == True
    net.eval()
    assert net._dropout.training == False


@pytest.mark.parametrize('img_size', [224, 256, 512])
def test_modify_dropout(net, img_size):
    """Test ability to modify dropout and fc modules of network"""
    dropout = nn.Sequential(OrderedDict([
        ('_bn2', nn.BatchNorm1d(net._bn1.num_features)),
        ('_drop1', nn.Dropout(p=net._global_params.dropout_rate)),
        ('_linear1', nn.Linear(net._bn1.num_features, 512)),
        ('_relu', nn.ReLU()),
        ('_bn3', nn.BatchNorm1d(512)),
        ('_drop2', nn.Dropout(p=net._global_params.dropout_rate / 2))
    ]))
    fc = nn.Linear(512, net._global_params.num_classes)

    net._dropout = dropout
    net._fc = fc

    data = torch.zeros((2, 3, img_size, img_size))
    output = net(data)
    assert not torch.isnan(output).any()


@pytest.mark.parametrize('img_size', [224, 256, 512])
def test_modify_pool(net, img_size):
    """Test ability to modify pooling module of network"""

    class AdaptiveMaxAvgPool(nn.Module):

        def __init__(self):
            super().__init__()
            self.ada_avgpool = nn.AdaptiveAvgPool2d(1)
            self.ada_maxpool = nn.AdaptiveMaxPool2d(1)

        def forward(self, x):
            avg_x = self.ada_avgpool(x)
            max_x = self.ada_maxpool(x)
            x = torch.cat((avg_x, max_x), dim=1)
            return x

    avg_pooling = AdaptiveMaxAvgPool()
    fc = nn.Linear(net._fc.in_features * 2, net._global_params.num_classes)

    net._avg_pooling = avg_pooling
    net._fc = fc

    data = torch.zeros((2, 3, img_size, img_size))
    output = net(data)
    assert not torch.isnan(output).any()

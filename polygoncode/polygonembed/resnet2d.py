import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

class ResNet2D(torchvision.models.resnet.ResNet):
    '''
    See the torchvision version of ResNet
    https://github.com/pytorch/vision/blob/21153802a3086558e9385788956b0f2808b50e51/torchvision/models/resnet.py
    '''
    def __init__(self, block, layers, in_channels = 1, num_classes=1000, zero_init_residual=False):
        super(ResNet2D, self).__init__(block, layers, num_classes, zero_init_residual)
        self.conv1 = torch.nn.Conv2d(in_channels, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), bias=False)


def get_resnet_model(resnet_type, num_classes = 20, in_channels = 1):
	if resnet_type == "resnet18":
		model = ResNet2D(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], in_channels = in_channels, num_classes = num_classes)
	elif resnet_type == "resnet34":
		model = ResNet2D(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], in_channels = in_channels, num_classes = num_classes)
	elif resnet_type == "resnet50":
		model = ResNet2D(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], in_channels = in_channels, num_classes = num_classes)
	elif resnet_type == "resnet101":
		model = ResNet2D(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], in_channels = in_channels, num_classes = num_classes)
	elif resnet_type == "resnet152":
		model = ResNet2D(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], in_channels = in_channels, num_classes = num_classes)

	return model

# def resnet18(pretrained=False, **kwargs):
#     """Constructs a ResNet-18 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet2D(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], **kwargs)
#     # if pretrained:
#     #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
#     return model


# def resnet34(pretrained=False, **kwargs):
#     """Constructs a ResNet-34 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet2D(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], **kwargs)
#     # if pretrained:
#     #     model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
#     return model


# def resnet50(pretrained=False, **kwargs):
#     """Constructs a ResNet-50 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet2D(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], **kwargs)
#     # if pretrained:
#     #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#     return model


# def resnet101(pretrained=False, **kwargs):
#     """Constructs a ResNet-101 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet2D(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], **kwargs)
#     # if pretrained:
#     #     model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
#     return model


# def resnet152(pretrained=False, **kwargs):
#     """Constructs a ResNet-152 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet2D(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], **kwargs)
#     # if pretrained:
#     #     model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
#     return model
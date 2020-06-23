from efficientnet_pytorch import EfficientNet as _EfficientNet

dependencies = ['torch']


def _create_model_fn(model_name):
    def _model_fn(num_classes=1000, in_channels=3, pretrained='imagenet'):
        """Create Efficient Net.

        Described in detail here: https://arxiv.org/abs/1905.11946

        Args:
            num_classes (int, optional): Number of classes, default is 1000.
            in_channels (int, optional): Number of input channels, default
                is 3.
            pretrained (str, optional): One of [None, 'imagenet', 'advprop']
                If None, no pretrained model is loaded.
                If 'imagenet', models trained on imagenet dataset are loaded.
                If 'advprop', models trained using adversarial training called
                advprop are loaded. It is important to note that the
                preprocessing required for the advprop pretrained models is
                slightly different from normal ImageNet preprocessing
        """
        model_name_ = model_name.replace('_', '-')
        if pretrained is not None:
            model = _EfficientNet.from_pretrained(
                model_name=model_name_,
                advprop=(pretrained == 'advprop'),
                num_classes=num_classes,
                in_channels=in_channels)
        else:
            model = _EfficientNet.from_name(
                model_name=model_name_,
                override_params={'num_classes': num_classes},
            )
            model._change_in_channels(in_channels)

        return model

    return _model_fn

for model_name in ['efficientnet_b' + str(i) for i in range(9)]:
    locals()[model_name] = _create_model_fn(model_name)

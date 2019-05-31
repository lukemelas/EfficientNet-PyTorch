# EfficientNet PyTorch
This repository contains an op-for-op PyTorch reimplementation of EfficientNet, the new convolutional neural network architecture from [EfficientNet](https://arxiv.org/abs/1905.11946) ([TensorFlow implementation](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)). 

At the moment, you can: 
 * 
 * Load pretrained EfficientNet models
 * Evaluate and train on ImageNet with the models

_Upcoming features_:  In the next few days, you will be able to:
 * Install this repository via `pip` 
 * Easily load pretrained models without having to manually run `download.sh` in `pretrained/pretrained_pytorch`    
 * Easily predict on your own images
 * Easily replicate the paper's results by training from scratch 

### Table of contents
 1. [About EfficientNet](#about_efficientnet_tf)
 2. [About EfficientNet-PyTorch](#about_efficientnet_pytorch)
 2. [Examples](#examples)
 3. [Models](#models) 
 4. [Installation](#installation) 

### About EfficientNet Models <a name="about_efficientnet_tf"></a>

If you're new to EfficientNets, here is an explanation straight from the official TensorFlow implementation: 

EfficientNets are a family of image classification models, which achieve state-of-the-art accuracy, yet being an order-of-magnitude smaller and faster than previous models. We develop EfficientNets based on AutoML and Compound Scaling. In particular, we first use [AutoML Mobile framework](https://ai.googleblog.com/2018/08/mnasnet-towards-automating-design-of.html) to develop a mobile-size baseline network, named as EfficientNet-B0; Then, we use the compound scaling method to scale up this baseline to obtain EfficientNet-B1 to B7.

<table border="0">
<tr>
    <td>
    <img src="https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/params.png" width="100%" />
    </td>
    <td>
    <img src="https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/flops.png", width="90%" />
    </td>
</tr>
</table>

EfficientNets achieve state-of-the-art accuracy on ImageNet with an order of magnitude better efficiency:


* In high-accuracy regime, our EfficientNet-B7 achieves state-of-the-art 84.4% top-1 / 97.1% top-5 accuracy on ImageNet with 66M parameters and 37B FLOPS, being 8.4x smaller and 6.1x faster on CPU inference than previous best [Gpipe](https://arxiv.org/abs/1811.06965).

* In middle-accuracy regime, our EfficientNet-B1 is 7.6x smaller and 5.7x faster on CPU inference than [ResNet-152](https://arxiv.org/abs/1512.03385), with similar ImageNet accuracy.

* Compared with the widely used [ResNet-50](https://arxiv.org/abs/1512.03385), our EfficientNet-B4 improves the top-1 accuracy from 76.3% of ResNet-50 to 82.6% (+6.3%), under similar FLOPS constraint.



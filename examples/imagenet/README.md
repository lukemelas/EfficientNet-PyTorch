### Imagenet

This is a preliminary directory for evaluating the model on ImageNet. It is adapted from the standard PyTorch Imagenet script. 

For now, only evaluation is supported, but I am currently building scripts to assist with training new models on Imagenet. 

The evaluation results are slightly different from the original TensorFlow repository, due to differences in data preprocessing. For example, with the current preprocessing, `efficientnet-b3` gives a top-1 accuracy of `80.8`, rather than `81.1` in the paper. I am working on porting the TensorFlow preprocessing into PyTorch to address this issue.   

To run on Imagenet, place your `train` and `val` directories in `data`. 

Example commands: 
```bash
# Evaluate small EfficientNet on CPU
python main.py data -e -a 'efficientnet-b0' --pretrained 
```
```bash
# Evaluate medium EfficientNet on GPU
python main.py data -e -a 'efficientnet-b3' --pretrained --gpu 0 --batch-size 128
```
```bash
# Evaluate ResNet-50 for comparison
python main.py data -e -a 'resnet50' --pretrained --gpu 0
```

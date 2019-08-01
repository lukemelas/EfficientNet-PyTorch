### TensorFlow to PyTorch Conversion

This directory is used to convert TensorFlow weights to PyTorch. It was hacked together fairly quickly, so the code is not the most beautiful (just a warning!), but it does the job. I will be refactoring it soon. 

I should also emphasize that you do *not* need to run any of this code to load pretrained weights. Simply use `EfficientNet.from_pretrained(...)`. 

That being said, the main script here is `convert_to_tf/load_tf_weights.py`. In order to use it, you should first download the pretrained TensorFlow weights:
 ```bash
cd pretrained_tensorflow
./download.sh efficientnet-b0
cd ..
```
Then
```bash
mkdir -p pretrained_pytorch
cd convert_tf_to_pt
python load_tf_weights.py \
    --model_name efficientnet-b0 \
    --tf_checkpoint ../pretrained_tensorflow/efficientnet-b0/ \
    --output_file ../pretrained_pytorch/efficientnet-b0.pth
``` 

<!-- Here is a helpful utility:
mv efficientnet-b0.pth efficientnet-b0-$(sha256sum efficientnet-b0.pth | head -c 8).pth
-->

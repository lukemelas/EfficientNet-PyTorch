import os
import numpy as np
import PIL
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet

from efficientnet_pytorch import EfficientNet

from sotabencheval.image_classification import ImageNetEvaluator
from sotabencheval.utils import is_server

if is_server():
    DATA_ROOT = DATA_ROOT = os.environ.get('IMAGENET_DIR', './imagenet')  # './.data/vision/imagenet'
else: # local settings
    DATA_ROOT = os.environ['IMAGENET_DIR']
    assert bool(DATA_ROOT), 'please set IMAGENET_DIR environment variable'
    print('Local data root: ', DATA_ROOT)

model_name = 'EfficientNet-B5'
model = EfficientNet.from_pretrained(model_name.lower())
image_size = EfficientNet.get_image_size(model_name.lower())

input_transform = transforms.Compose([
    transforms.Resize(image_size, PIL.Image.BICUBIC),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = ImageNet(
    DATA_ROOT,
    split="val",
    transform=input_transform,
    target_transform=None,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

model = model.cuda()
model.eval()

evaluator = ImageNetEvaluator(model_name=model_name,
                              paper_arxiv_id='1905.11946')

def get_img_id(image_name):
    return image_name.split('/')[-1].replace('.JPEG', '')

with torch.no_grad():
    for i, (input, target) in enumerate(test_loader):
        input = input.to(device='cuda', non_blocking=True)
        target = target.to(device='cuda', non_blocking=True)
        output = model(input)
        image_ids = [get_img_id(img[0]) for img in test_loader.dataset.imgs[i*test_loader.batch_size:(i+1)*test_loader.batch_size]]
        evaluator.add(dict(zip(image_ids, list(output.cpu().numpy()))))
        if evaluator.cache_exists:
            break

if not is_server():
    print("Results:")
    print(evaluator.get_results())

evaluator.save()

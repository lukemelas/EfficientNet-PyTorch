import os
import sys
import getopt
import requests

import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image

from flask import Flask
from flask import request
import traceback
import json
import re
import traceback
from uuid import uuid4

app = Flask(__name__)


def download_image(image_url, filename):
    img_data = requests.get(image_url).content
    with open(filename, 'wb') as handler:
        handler.write(img_data)

    return filename


@app.route("/detect", methods=["POST"])
def detect():
    filename = str(uuid4())
    filename = os.path.join(directory, filename + ".jpg")

    try:
        image_url = request.json["url"]
        top_k = request.json["top_k"]
        filename = download_image(image_url, filename)
        
        img = Image.open(filename)

        tfms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), 
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        img = tfms(img).unsqueeze(0)
 
        with torch.no_grad():
            logits = model(img)

        preds = torch.topk(logits, k=top_k).indices.squeeze(0).tolist()

        results = []

        for idx in preds:
            prob = torch.softmax(logits, dim=1)[0, idx].item()
            results.append({'label': labels_map[idx][1], 'label_id': labels_map[idx][0], 'score': '{0:.2f}'.format(prob*100)})

        return json.dumps(results), 200
        
    except:
        traceback.print_exc()
        return {'message': 'input error'}, 400

    finally:
        if os.path.exists(filename):
            os.remove(filename)


def main(argv):
    usage = """ python3 app.py -m <model_type>
for instance :
python3 app.py -m 0 for b0 efficient net
Details about the models are below:
             
Name              # Params   Top-1 Acc.  Pretrained?
efficientnet-b0     5.3M       76.3        ✓
efficientnet-b1     7.8M       78.8        ✓
efficientnet-b2     9.2M       79.8        ✓
efficientnet-b3     12M        81.1        ✓
efficientnet-b4     19M        82.6        ✓
efficientnet-b5     30M        83.3        ✓
efficientnet-b6     43M        84.0        -
efficientnet-b7     66M        84.4        -
"""

    global model, labels_map, image_size, directory

    try:
        opts, args = getopt.getopt(argv, "hm:", ["help", "model"])

    except getopt.GetoptError:
      print(usage)
      sys.exit(2)
    for opt, arg in opts:
      print(opt)
      print(arg)
      if opt in ("-h", "--help"):
        print(usage)
        sys.exit()
      elif opt in ("-m", "--model"):
          model_type = int(arg)

    directory = 'upload'

    if not os.path.exists(directory):
       os.makedirs(directory)

    try:
        model_name = 'efficientnet-b' + str(model_type)
    except:
        print("Error : missing model number")
        print(usage)
        sys.exit(2)
    
    print("API running with " + model_name)
    model = EfficientNet.from_pretrained(model_name)
    image_size = EfficientNet.get_image_size(model_name)

    labels_map = json.load(open('labels_map.txt'))
    labels_map = [labels_map[str(i)] for i in range(1000)]


    model.eval()

    port = 5000
    host = '0.0.0.0'

    app.run(host=host, port=port, threaded=True)

if __name__ == '__main__':
    main(sys.argv[1:])


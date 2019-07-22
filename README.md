# Cats vs. Dogs Classifier with Model Pruning and Deployment

![](https://raw.githubusercontent.com/nlml/cats-vs-dogs-classifier-pruned-and-served/master/cat.jpeg) ![](https://raw.githubusercontent.com/nlml/cats-vs-dogs-classifier-pruned-and-served/master/dog.jpeg)

In this repo, we:

- Finetune a Squeezenet trained on Imagenet to specialise in classifying cats versus dogs,

- Iteratively prune this finetuned model, reducing its size from 2.9MiB to 1.7MiB while still maintaining the 98 per cent accuracy rate achieved before any pruning,

- Serve this pruned model for prediction using Flask and Docker.

All of these steps are reproducible following the instructions below.

## Training the model

You will need `unzip` and `jhead` installed to unzip and prepare the cats vs. dogs dataset.

```
sudo apt-get update && apt-get install -y unzip jhead
```

Then to train the model:

```
cd /path/to/repo/train_model

# Make a  virtual environment with the required packages (or just `pip install -r requirements.txt`)
mkvirtualenv -r requirements.txt -p python3.6 catsdogs

# Ensure you are in the correct virtualenv, then run the training script
python train.py
```

You should get around 98% final validation accuracy.

## Model serving

### Preparation

After training the model, the final weights will be saved to `final_pruned_model.pth` in the repo root. You need to copy these to the `serve_model` folder:

```
cd /path/to/repo/train_model
cp final_pruned_model.pth serve_model/
```

Once that's done, you just need to build the image and run it.

You will need `nvidia-docker` installed if you want to use GPU, otherwise just `docker`. If you want to use just CPU instead of GPU, replace `FROM ufoym/deepo:pytorch-py36-cu100` with `FROM ufoym/deepo:pytorch-py36-cpu` in the first line of `serve_model/Dockerfile`.

Last step is to build and run the image:

```
cd /path/to/repo/serve_model
nvidia-docker build -t cats-vs-dogs .
nvidia-docker run -p 5000:5000 cats-vs-dogs
```

### /upload classification UI endpoint

Point your browser to http://localhost:5000/upload where you can upload an image to classify through the model.

### /classify POST endpoint

You can also use `localhost:5000/classify` endpoint to POST data to be classified. It accepts JSON format data and binary data.

Here is an example of how you might post some binary data in Python:

```
import requests

with open('dog.jpeg', 'rb') as f:
	data = f.read()
r = requests.post(
    url='http://localhost:5000/classify',
    data=data,
    headers={'Content-Type': 'application/octet-stream'}
)
r.json()

Out[1]: {'predictions': [[3.103691051364876e-05, 0.9999690055847168]],
	     'classes': ['Cat', 'Dog']}
```

Here is an example of how you might post some JSON data in Python:

```
import torch
from torchvision import transforms
from PIL import Image
import requests

img = Image.open('dog.jpeg')
t = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])
tensor = t(img)

headers = 'Content-Type: application/json'
payload = {'tensor': tensor.tolist()}

r = requests.post("http://localhost:5000/classify",
                  json=payload)
r.json()

Out[1]: {'predictions': [[3.103691051364876e-05, 0.9999690055847168]],
	     'classes': ['Cat', 'Dog']}
```

This also supports batches of tensors (sent as JSON 4D lists).

You could also use `curl`, for instance (with some fake data):

```
curl localhost:5000/classify -d '{"tensor": [[[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]}' -H 'Content-Type: application/json'

{"predictions": [[0.7711397409439087, 0.2288602739572525]], "classes": ["Cat", "Dog"]}
```

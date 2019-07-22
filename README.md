# cats-vs-dogs-classifier-pruned-and-served
Example repo showing training and serving of a basic cats vs. dogs convnet classifier

## Model training

You will need `unzip` and `jhead` installed to unzip and prepare the cats vs. dogs dataset.

```
sudo apt-get update && apt-get install -y unzip jhead
```

Then train the model:

```
cd /path/to/repo/train_model

# Make a  virtual environment with the required packages (or just `pip install -r requirements.txt`)
mkvirtualenv -r requirements.txt -p python3.6 catsdogs

# Ensure you are in the correct virtualenv, then run the training script
python train.py
```

## Model serving

After training the model, the final weights will be saved to `final_pruned_model.pth` in the repo root.


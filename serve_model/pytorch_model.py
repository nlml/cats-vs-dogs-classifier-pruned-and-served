import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import torch.nn.functional as F
import time


# Class object storing the Pytorch model with prediction methods
class Model():

    to_pil = transforms.ToPILImage()

    transform_single = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    transform_batch = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.jit.load('final_pruned_model.pth').to(self.device)

    def predict_image_from_path(self, path):
        sample = default_loader(path)
        out, t = self.predict_image_from_pil(sample)
        return '\n'.join([self._html_prettify_list_output(i) for i in out])

    def predict_image_from_pil(self, sample):
        sample = self.transform_single(sample)
        out, t = self._predict_tensor(sample)
        return out, t

    def predict_image_from_tensor(self, tensor):
        tensor = [torch.FloatTensor(i) for i in tensor]
        if len(tensor[0].shape) == 2:  # single image was passed in
            tensor = torch.stack(tensor, 0)
            tensor = self.transform_single(self.to_pil(tensor))
        elif len(tensor[0].shape) == 3:
            if all([i.shape == tensor[0].shape for i in tensor[1:]]):
                tensor = torch.stack(
                    [self.transform_single(self.to_pil(i)) for i in tensor], 0)
            else:  # need all tensors to be same shape
                tensor = torch.stack(
                    [self.transform_batch(self.to_pil(i)) for i in tensor], 0)
        return self._predict_tensor(tensor)

    def _predict_tensor(self, tensor):
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        since = time.time()
        with torch.no_grad():
            out = F.softmax(self.model(tensor.to(self.device)), dim=1).cpu()
        t = (time.time() - since) * 1000
        return out.tolist(), t

    def _html_prettify_list_output(self, out):
        return '\n'.join([
            f'<h1>Prediction     : {"CAT" if out[0] > out[1] else "DOG"}</h1>',
            f'<h2>Cat probability: {out[0] * 100:.2f}%</h2>',
            f'<h2>Dog probability: {out[1] * 100:.2f}%</h2>'
        ])

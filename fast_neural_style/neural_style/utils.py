import torch
from PIL import Image
import random
from torchvision.transforms import InterpolationMode

def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

def random_subset(train_dataset, size_of_subset : int = 100):
    """
    Function for picking a random subset for the 'train_dataset' object
    Attributes
    ----------
        train_dataset:
            A pytorch dataset object
        size_of_subset:
            An integer representing the size of the data subset
    
    Returns:
        A list of size size_of_subset with random valid indices for train_dataset
    """
    N = len(train_dataset.samples) # The total number of indices to choose from
    Subset_list = random.sample(range(0,N),size_of_subset) # Pick size_of_subset random indices in the interval [0;N]
    return Subset_list

# Dictionary for possible activation functions used in the residual block of the transformer network
act_funcs = {
    "ReLU" : torch.nn.ReLU(),
    "RReLU" : torch.nn.RReLU()
}

# Dictionary for possible interpolation methods used in the transformation step when loading the data
interpolationmode = {
    "NEAREST" : InterpolationMode.NEAREST,
    "BILINEAR" : InterpolationMode.BILINEAR
}
import torch
import torch.utils.data
import torchvision.datasets as dset
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from torchvision.io import read_image
# DEFINE CONSTANTS

# Root directory for dataset
root_dir= 'Data/room=[6,6,2.4]'


# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# DEFINE DATASET

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, is_test=False):
        dirs = os.listdir(root_dir)
        dirs = [dir for dir in dirs if dir.startswith('Test') == is_test]
        self.revs = None
        self.targets = None
        for dir in dirs:
            rev_target_dirs = os.listdir(os.path.join(root_dir,dir))
            rev_dir = [d for d in rev_target_dirs if d.startswith('beta')][0]
            target_dir = [d for d in rev_target_dirs if d.startswith('Target')][0]
            rev = os.path.join(root_dir,dir,rev_dir,'data_phase.npy' if is_test else 'Reverberated/data_phase.npy' )
            a = np.load(rev)
            if self.revs is None:
                self.revs = np.copy(a)
            else:
                self.revs = np.vstack((self.revs,a))
            target = os.path.join(root_dir,dir,target_dir,'data_phase.npy')
            b = np.load(target)
            if self.targets is None:
                self.targets = np.copy(b)
            else:
                self.targets = np.vstack((self.targets,b))
                

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        reverberated_image = self.revs[idx, :, :]
        target_image = self.targets[idx, :, :]
        return torch.from_numpy(reverberated_image), torch.from_numpy(target_image)


# LOAD DATA

training_data = CustomImageDataset(root_dir)
test_data = CustomImageDataset(root_dir, is_test=True)


train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot a training image
real_batch = next(iter(train_dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(real_batch[0][0,:,:])
plt.show()
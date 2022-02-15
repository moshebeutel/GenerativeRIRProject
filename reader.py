import torch
import torch.utils.data
import torchvision.datasets as dset
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.utils.data import Dataset
import os
from torchvision.io import read_image
import torchvision.transforms as transforms
# DEFINE CONSTANTS
# Root directory for dataset
root_dir= 'Data/room=[6,6,2.4]'
# Number of workers for dataloader
workers = 2
# Batch size during training
batch_size = 64

# DEFINE DATASET
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, is_test=False):
        self._transform = transforms.Normalize(mean=0.5, std=0.5)

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
                self.revs = np.copy(a[:,:,:126])
            else:
                self.revs = np.vstack((self.revs,a[:,:,:126]))
            self.revs = np.vstack((self.revs,a[:,:,126:]))
            target = os.path.join(root_dir,dir,target_dir,'data_phase.npy')
            b = np.load(target)
            if self.targets is None:
                self.targets = np.copy(b[:,:,:126])
            else:
                self.targets = np.vstack((self.targets,b[:,:,:126]))
            self.targets = np.vstack((self.targets,b[:,:,:126]))

        
    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        reverberated_image = self.revs[idx, :, :].reshape(1,7,-1)
        target_image = self.targets[idx, :, :].reshape(1,7,-1)
        return self._transform(torch.from_numpy(reverberated_image)), self._transform(torch.from_numpy(target_image))

# training_data = CustomImageDataset(root_dir)
# test_data = CustomImageDataset(root_dir, is_test=True)

# train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# # Plot a training image
# from mpl_toolkits.axes_grid1 import ImageGrid

# train_rev, train_target = next(iter(train_dataloader))
# fig = plt.figure(figsize=(5., 10))


# grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                  nrows_ncols=(10, 2),  # creates 2x2 grid of axes
#                  axes_pad=0.1,  # pad between axes in inch.
#                  )

# l = []
# for i in range(10):
# 	l.append(train_rev[i])
# 	l.append(train_target[i])

# for ax, im in zip(grid, l):
#     # Iterating over the grid returns the Axes.
#     ax.imshow(im)

# plt.show()
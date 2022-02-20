from torch import norm
from reader import CustomImageDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pix2pix import display_progress, _weights_init, Generator, PatchGAN

from datetime import datetime as dt
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
# Root directory for dataset
root_dir = 'Data/room=[6,6,2.4]'
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Number of workers for dataloader
workers = 16

# Batch size during training
batch_size = 64

# Number of training epochs
num_epochs = 10
# Number of channels in the training images. For color images this is 3
nc = 1


# Learning rate for optimizers
lr = 0.0002
# Decide which device we want to run on
device = torch.device("cuda:0" if (
    torch.cuda.is_available() and ngpu > 0) else "cpu")

#Dataset and Dataloader 
training_data = CustomImageDataset(root_dir)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=workers)
now=dt.now()
time_str = dt.isoformat(now)
writer = SummaryWriter(f'runs/rir_' + time_str)

# Plot a training image
# train_rev, train_target = next(iter(train_dataloader))
# fig = plt.figure(figsize=(5., 10))

# grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                  nrows_ncols=(10, 1),  # creates 2x2 grid of axes
#                  axes_pad=0.1,  # pad between axes in inch.
#                  )

# l = []
# for i in range(10):
# 	l.append(torch.hstack((train_rev[i].squeeze(),train_target[i].squeeze())))
# 	# l.append(train_target[i])

# for ax, im in zip(grid, l):
#     # Iterating over the grid returns the Axes.
#     ax.imshow(im)

# plt.show()


class Pix2Pix(pl.LightningModule):

    def __init__(self, in_channels, out_channels, learning_rate=0.0002, lambda_recon=200, display_step=25):

        super().__init__()
        self.save_hyperparameters()
        
        self.display_step = display_step
        self.gen = Generator(in_channels, out_channels)
        self.patch_gan = PatchGAN(in_channels + out_channels)

        # intializing weights
        self.gen = self.gen.apply(_weights_init)
        self.patch_gan = self.patch_gan.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    @property
    def generator(self):
        return self.gen

    def _gen_step(self, real_images, conditioned_images):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        fake_images = self.gen(conditioned_images)
        disc_logits = self.patch_gan(fake_images, conditioned_images)
        adversarial_loss = self.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))

        # calculate reconstruction loss
        recon_loss = self.recon_criterion(fake_images, real_images)
        lambda_recon = self.hparams.lambda_recon

        return adversarial_loss + lambda_recon * recon_loss

    def _disc_step(self, real_images, conditioned_images):
        fake_images = self.gen(conditioned_images).detach()
        fake_logits = self.patch_gan(fake_images, conditioned_images)

        real_logits = self.patch_gan(real_images, conditioned_images)

        fake_loss = self.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.adversarial_criterion(real_logits, torch.ones_like(real_logits))
        return (real_loss + fake_loss) / 2

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        self.disc_opt = torch.optim.Adam(self.patch_gan.parameters(), lr=lr)
        return self.disc_opt, self.gen_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        # real, condition = batch
        condition, real = batch

        loss = None
        if optimizer_idx == 0:
            self.errD = self._disc_step(real, condition)
            self.log('PatchGAN Loss', self.errD)
            loss = self.errD
        elif optimizer_idx == 1:
            self.errG = self._gen_step(real, condition)
            self.log('Generator Loss', self.errG)
            loss = self.errG
        
        if (self.current_epoch + 1)%self.display_step==0 and batch_idx==0 and optimizer_idx==1:
            fake = self.gen(condition).detach()
            display_progress(condition[0], fake[0], real[0])
            
            # writer.add_graph(self.gen, condition)
            # writer.add_image('condition-fake-real', torch.hstack((condition[0].squeeze(),fake[0].squeeze(), real[0].squeeze())))
            # writer.flush()
            torch.save({'epoch': self.current_epoch,'model_state_dict': self.gen.state_dict(), \
                'optimizer_state_dict': self.gen_opt.state_dict(),'loss': self.errG,}, \
                     f'Generator_Epoch_{self.current_epoch}_loss_{float(self.errG)}.pt')
            torch.save({'epoch': self.current_epoch,'model_state_dict': self.patch_gan.state_dict(), \
                'optimizer_state_dict': self.disc_opt.state_dict(),'loss': self.errD,}, \
                     f'Discriminator_Epoch_{self.current_epoch}_loss_{float(self.errD)}.pt')
 
        return loss

    
adv_criterion = nn.BCEWithLogitsLoss() 
recon_criterion = nn.L1Loss() 
lambda_recon = 200
display_step = 2


pix2pix = Pix2Pix(nc, nc, learning_rate=lr, lambda_recon=lambda_recon, display_step=display_step)
trainer = pl.Trainer(max_epochs=num_epochs, gpus=ngpu)
trainer.fit(pix2pix, train_dataloader)

test_data = CustomImageDataset(root_dir, is_test=True)
test_loader = DataLoader(test_data, batch_size=batch_size)


# Plot a training image
test_rev, test_target = next(iter(test_loader))
fig = plt.figure(figsize=(5., 10))

num_samples = 15

grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(num_samples, 1),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

pix2pix.gen.eval()
test_generated = pix2pix.gen(test_rev).detach()
l = []
for i in range(num_samples):
	l.append(torch.hstack((test_rev[i].squeeze(),test_target[i].squeeze(), test_generated[i].squeeze())))
	# l.append(train_target[i])

for ax, im in zip(grid, l):
    # Iterating over the grid returns the Axes.
    ax.imshow(im)

plt.show()

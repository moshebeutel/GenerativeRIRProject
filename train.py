from dcgan import Generator, Discriminator, weights_init
from reader import CustomImageDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# Root directory for dataset
root_dir = 'Data/room=[6,6,2.4]'

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 64

# Number of training epochs
num_epochs = 20

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 14

# Size of feature maps in generator
ngf = 16

# Size of feature maps in discriminator
ndf = 16


# Decide which device we want to run on
device = torch.device("cuda:0" if (
    torch.cuda.is_available() and ngpu > 0) else "cpu")


training_data = CustomImageDataset(root_dir)
test_data = CustomImageDataset(root_dir, is_test=True)

train_dataloader = DataLoader(
    training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# # Plot a training image

# train_rev, train_target = next(iter(train_dataloader))
# fig = plt.figure(figsize=(5., 10))


# grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                  nrows_ncols=(5, 1),  # creates 2x2 grid of axes
#                  axes_pad=0.1,  # pad between axes in inch.
#                  )

# l = []
# for i in range(5):
#     l.append(torch.hstack((train_rev[i].squeeze(),train_target[i].squeeze())))
# 	# l.append(train_rev[i])
# 	# l.append(train_target[i])

# for ax, im in zip(grid, l):
#     # Iterating over the grid returns the Axes.
#     ax.imshow(im)

# plt.show()


gen = Generator(ngpu).to(device)
disc = Discriminator(ngpu).to(device)
gen.apply(weights_init)
disc.apply(weights_init)


# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
# optimizerD = optim.Adam(disc.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.SGD(disc.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
optimizerG = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(train_dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Train with all-real batch
        disc.zero_grad()
        # Format batch
        # train_rev =  data[1].reshape(-1,1,7,126).float().to(device)
        train_rev =  torch.concat((data[0],data[1]), 3).float().to(device)
        # real_cpu = data[0].to(device)
        b_size = train_rev.size(0)
        label = torch.full((b_size,), real_label,
                           dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = disc(train_rev).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        # Calculate loss on all-real batchimport matplotlib.pyplot as plt
        D_x = output.mean().item()

        # Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = gen(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = disc(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        gen.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = disc(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(train_dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(train_dataloader)-1)):
            with torch.no_grad():
                fake = gen(fixed_noise).detach().cpu()
            # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            img_list.append(fake)

        iters += 1
    if epoch % 5 == 4:
        torch.save({
            'epoch': epoch,
            'model_state_dict': gen.state_dict(),
            'optimizer_state_dict': optimizerG.state_dict(),
            'loss': errG.item(),
            }, f'Generator_Epoch_{epoch}_loss_{errG.item()}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': disc.state_dict(),
            'optimizer_state_dict': optimizerD.state_dict(),
            'loss': errD.item(),
            }, f'Discriminator_Epoch_{epoch}_loss_{errD.item()}.pt')

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


train_rev, train_target = next(iter(train_dataloader))
fig = plt.figure(figsize=(5., 10))


grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(10, 1),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )
gen.eval()
l = []
for i in range(10):
    with torch.no_grad():
        # noise = torch.randn(1, nz, 1, 1, device=device)
        fake = gen(fixed_noise[i,:,:,:].unsqueeze(0)).detach().cpu().numpy().reshape(7, 126)
    l.append(fake)


for ax, im in zip(grid, l):
    # Iterating over the grid returns the Axes.
    ax.imshow(im)

plt.show()

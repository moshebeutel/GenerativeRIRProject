from unicodedata import name
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Number of channels in the training images. For color images this is 3
nc = 1
# Size of feature maps in generator
ngf = 32
# Size of feature maps in discriminator
ndf = 32


class DownSampleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True, bias=True, name=''):
        """
        Paper details:
        - C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        """
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.name = name
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel, strides, padding, bias=bias, dtype=torch.float64)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels, dtype=torch.float64)

        if activation:
            self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        # print(self.name, 'forward output shape', x.shape)
        return x


class UpSampleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=4,
        strides=2,
        padding=1,
        activation=True,
        batchnorm=True,
        dropout=False,
        name=''
    ):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.name = name

        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel, strides, padding, dtype=torch.float64)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels, dtype=torch.float64)

        if activation:
            self.act = nn.ReLU(True)

        if dropout:
            self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.deconv(x)
        if self.batchnorm:
            x = self.bn(x)

        if self.dropout:
            x = self.drop(x)

        # print(self.name, 'forward output shape', x.shape)
        return x


class Generator(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Paper details:
        - Encoder: C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        - Decoder: CD512-CD1024-CD1024-C1024-C1024-C512 -C256-C128
        """
        super().__init__()

        # encoder/donwsample convs
        self.encoders = [
    
            DownSampleConv(in_channels, ngf, kernel=4, strides=(1, 2), padding=1, batchnorm=False, name='encoder1'),

            DownSampleConv(ngf, ngf * 4,kernel=4, strides= (1, 2), padding=1, name='encoder2'),

            # DownSampleConv(ngf * 4, ngf * 8,kernel=4, strides= (1, 2), padding=1, name='encoder3'),

            # DownSampleConv(ngf * 8, ngf * 16, name='encoer4'),
  
            # DownSampleConv(ngf * 16, ngf * 16, name='encoder5'),
 
            # DownSampleConv(ngf * 16, ngf * 16, name='encoder6'),
 
            # DownSampleConv(ngf * 16, ngf * 16, name='encoder7'),
      
            # DownSampleConv(ngf * 16, ngf * 16,
            #                batchnorm=False, name='encoder8'),
        ]

        # decoder/upsample convs
        self.decoders = [
            # UpSampleConv(nz, ngf * 8, dropout=True,
            #              name='decoder1'),  # bs x 512 x 2 x 2
            # UpSampleConv(ngf * 16, ngf * 8, dropout=True,
            #              name='decoder2'),  # bs x 512 x 4 x 4
            # UpSampleConv(ngf * 16, ngf * 8, dropout=True,
            #              name='decoder3'),  # bs x 512 x 8 x 8

            # UpSampleConv(ngf * 16, ngf * 8, name='decoder4'),

            # UpSampleConv(ngf * 16, ngf * 4, name='decoder5'),

            # UpSampleConv(ngf * 8, ngf * 2,strides=(1,2), padding=1, name='decoder6'),
            UpSampleConv(ngf * 4, ngf, strides=(1,2), padding=1, dropout=True, name='decoder7'),  # bs x 64 x 128 x 128
        ]
        # self.decoder_channels = [512, 512, 512, 512, 256, 128, 64]
        self.final_conv = nn.ConvTranspose2d(
            ngf, out_channels, kernel_size=4, stride=(1,2), padding=(1,0), dtype=torch.float64)
        
        
        self.tanh = nn.Tanh()

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x):
        skips_cons = []
        for encoder in self.encoders:
            x = encoder(x)

            skips_cons.append(x)

        skips_cons = list(reversed(skips_cons[:-1]))
        decoders = self.decoders[:-1]

        for decoder, skip in zip(decoders, skips_cons):
            x = decoder(x)
            # print(x.shape, skip.shape)
            x = torch.cat((x, skip), axis=1)

        x = self.decoders[-1](x)
        # print(x.shape)
        
        x = self.final_conv(x)
        
        # print('decoder8 generator final', 'forward output shape', x.shape)
        return self.tanh(x)
        # return x


class PatchGAN(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.d1 = DownSampleConv(input_channels, ndf,kernel= 4, strides=(1, 3),padding= 1, bias=False, name='patchgan1')
        self.d2 = DownSampleConv(ndf, ndf * 2, kernel=4, strides= (1, 3), padding= 1, bias=False, name='patchgan2')
        self.d3 = DownSampleConv(ndf * 2, ndf * 4,kernel=4, strides= (1, 3), padding=1, bias=False, name='patchgan3')
        self.d4 = DownSampleConv(ndf * 4, ndf * 8,kernel=4, strides= (1, 3), padding=1, bias=False, name='patchgan4')
        self.final = nn.Conv2d(ndf * 8, 1,kernel_size=(3, 4), stride=(1, 3),padding = 1, bias=False, dtype=torch.float64)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.d1(x)
        x1 = self.d2(x0)
        x2 = self.d3(x1)
        x3 = self.d4(x2)
        xn = self.final(x3)
        # print('patchgan5 patchgan final', 'forward output shape', x.shape)

        return xn


def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def display_progress(cond, fake, real, figsize=(10, 5)):
    cond = cond.detach().cpu().permute(1, 2, 0)
    fake = fake.detach().cpu().permute(1, 2, 0)
    real = real.detach().cpu().permute(1, 2, 0)

    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax[0].imshow(cond)
    ax[2].imshow(fake)
    ax[1].imshow(real)
    plt.show()

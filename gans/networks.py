import torch
import torch.nn as nn

from gans.modules import ResnetBlock, ConditionalResidualBlock, TwoInputSequential, ConditionalBatchNorm1d
from settings import GPU_IDs


class ConditionalResnetGenerator(nn.Module):
    """
    Modified version of ResnetGenerator that supports stochastic mappings
    using Conditional instance norm (can support CBN easily)
    """

    def __init__(self, nlatent, input_nc, output_nc, ngf=64, norm_layer=ConditionalBatchNorm1d,
                 use_dropout=False,
                 n_blocks=9, use_bias=True):
        super(ConditionalResnetGenerator, self).__init__()

        model = [
            nn.Linear(input_nc, ngf, bias=use_bias),
            norm_layer(ngf, nlatent),
            nn.ReLU(True),

            nn.Linear(ngf, 2 * ngf, bias=use_bias),
            norm_layer(2 * ngf, nlatent),
            nn.ReLU(True),

            nn.Linear(2 * ngf, 4 * ngf, bias=use_bias),
            norm_layer(4 * ngf, nlatent),
            nn.ReLU(True)
        ]

        for i in range(n_blocks):
            model += [ConditionalResidualBlock(x_dim=4 * ngf, z_dim=nlatent,
                                               norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        model += [

            nn.Linear(4 * ngf, 2 * ngf, bias=use_bias),
            norm_layer(2 * ngf, nlatent),
            nn.ReLU(True),

            nn.Linear(2 * ngf, ngf, bias=use_bias),
            norm_layer(ngf, nlatent),
            nn.ReLU(True),

            nn.Linear(ngf, output_nc, bias=use_bias),
            nn.Softmax(dim=1),
        ]

        self.model = TwoInputSequential(*model)

    def forward(self, input, noise):
        if len(GPU_IDs) > 0:
            x = nn.parallel.data_parallel(self.model, (input, noise), GPU_IDs)
        else:
            x = self.model(input, noise)
        filter = input.bool()
        x = filter * x  # how much did we throw out?
        x = x / torch.sum(x, dim=1).view(x.shape[0], 1)
        return x


class ResnetGenerator(nn.Module):
    """
    ResnetGenerator for deterministic mappings
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm1d, use_dropout=False,
                 n_blocks=9, use_bias=True):
        super(ResnetGenerator, self).__init__()

        model = [
            nn.Linear(input_nc, ngf, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),

            nn.Linear(ngf, 2 * ngf, bias=use_bias),
            norm_layer(2 * ngf),
            nn.ReLU(True),

            nn.Linear(2 * ngf, 4 * ngf, bias=use_bias),
            norm_layer(4 * ngf),
            nn.ReLU(True),
        ]

        for i in range(n_blocks):
            model += [ResnetBlock(4 * ngf, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias)]

        model += [

            nn.Linear(4 * ngf, 2 * ngf, bias=use_bias),
            norm_layer(2 * ngf),
            nn.ReLU(True),

            nn.Linear(2 * ngf, ngf, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),

            nn.Linear(ngf, output_nc, bias=use_bias),
            nn.Softmax(dim=1),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if len(GPU_IDs) > 0:
            x = nn.parallel.data_parallel(self.model, input, GPU_IDs)
        else:
            x = self.model(input)
        filter = input.bool()
        x = filter * x  # how much did we throw out?
        x = x / torch.sum(x, dim=1).view(x.shape[0], 1)
        return x


class Discriminator(nn.Module):
    """
    Discriminator network
    """

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm1d,
                 use_sigmoid=False, use_bias=True, use_norm=True):
        """
        latent2: number of channles in both latent codes (or one of them - depending on the model)
        input_nc: number of channels in input and output (assumes both inputs are concatenated)
        """
        super(Discriminator, self).__init__()

        if use_norm:
            sequence = [
                nn.Linear(input_nc, ndf, bias=use_bias),
                nn.LeakyReLU(0.2, True),

                nn.Linear(ndf, 2 * ndf, bias=use_bias),
                norm_layer(2 * ndf),
                nn.LeakyReLU(0.2, True),

                nn.Linear(2 * ndf, 4 * ndf, bias=use_bias),
                norm_layer(4 * ndf),
                nn.LeakyReLU(0.2, True),

                nn.Linear(4 * ndf, 4 * ndf, bias=use_bias),
                norm_layer(4 * ndf),
                nn.LeakyReLU(0.2, True),

                nn.Linear(4 * ndf, 1),
            ]
        else:
            sequence = [
                nn.Linear(input_nc, ndf, bias=use_bias),
                nn.LeakyReLU(0.2, True),

                nn.Linear(ndf, 2 * ndf, bias=use_bias),
                nn.LeakyReLU(0.2, True),

                nn.Linear(2 * ndf, 4 * ndf, bias=use_bias),
                nn.LeakyReLU(0.2, True),

                nn.Linear(4 * ndf, 4 * ndf, bias=use_bias),
                nn.LeakyReLU(0.2, True),

                nn.Linear(4 * ndf, 1),
            ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(GPU_IDs) > 0:
            return nn.parallel.data_parallel(self.model, input, GPU_IDs)
        else:
            return self.model(input)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, nlatent, input_nc, ndf=64, norm_layer=ConditionalBatchNorm1d,
                 use_sigmoid=False, use_bias=True):
        """
        latent2: number of channles in both latent codes (or one of them - depending on the model)
        input_nc: number of channels in input and output (assumes both inputs are concatenated)
        """
        super(ConditionalDiscriminator, self).__init__()

        sequence = [
            nn.Linear(input_nc, ndf, bias=use_bias),
            nn.LeakyReLU(0.2, True),

            nn.Linear(ndf, 2 * ndf, bias=use_bias),
            norm_layer(2 * ndf, nlatent),
            nn.LeakyReLU(0.2, True),

            nn.Linear(2 * ndf, 4 * ndf, bias=use_bias),
            norm_layer(4 * ndf, nlatent),
            nn.LeakyReLU(0.2, True),

            nn.Linear(4 * ndf, 4 * ndf, bias=use_bias),
            norm_layer(4 * ndf, nlatent),
            nn.LeakyReLU(0.2, True),

            nn.Linear(4 * ndf, 1),
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = TwoInputSequential(*sequence)

    def forward(self, input, noise):
        if len(GPU_IDs) > 0:
            return nn.parallel.data_parallel(self.model, (input, noise), GPU_IDs)
        else:
            return self.model(input, noise)


class DiscriminatorLatent(nn.Module):
    def __init__(self, nlatent, ndf, use_sigmoid=False, use_norm=True):
        super(DiscriminatorLatent, self).__init__()

        if use_norm:
            sequence = [
                nn.Linear(nlatent, ndf),
                nn.BatchNorm1d(ndf),
                nn.LeakyReLU(0.2, True),

                nn.Linear(ndf, ndf),
                nn.BatchNorm1d(ndf),
                nn.LeakyReLU(0.2, True),

                nn.Linear(ndf, ndf),
                nn.BatchNorm1d(ndf),
                nn.LeakyReLU(0.2, True),

                nn.Linear(ndf, 1)
            ]
        else:
            sequence = [
                nn.Linear(nlatent, ndf),
                nn.LeakyReLU(0.2, True),

                nn.Linear(ndf, ndf),
                nn.LeakyReLU(0.2, True),

                nn.Linear(ndf, ndf),
                nn.LeakyReLU(0.2, True),

                nn.Linear(ndf, 1)
            ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):

        if len(GPU_IDs) > 0:
            return nn.parallel.data_parallel(self.model, input, GPU_IDs)
        else:
            return self.model(input)


class LatentEncoder(nn.Module):
    """
    Encoder network for latent variables
    """

    def __init__(self, nlatent, input_nc, nef, norm_layer=nn.BatchNorm1d, use_bias=True):
        super(LatentEncoder, self).__init__()
        # use_bias = False

        sequence = [
            nn.Linear(input_nc, nef, bias=use_bias),
            # nn.Linear(input_nc, nef, bias=True),
            nn.ReLU(True),

            nn.Linear(nef, 2 * nef, bias=use_bias),
            norm_layer(2 * nef),
            nn.ReLU(True),

            nn.Linear(2 * nef, 4 * nef, bias=use_bias),
            norm_layer(4 * nef),
            nn.ReLU(True),

            nn.Linear(4 * nef, 8 * nef, bias=use_bias),
            norm_layer(8 * nef),
            nn.ReLU(True),

            nn.Linear(8 * nef, 8 * nef, bias=use_bias),
            norm_layer(8 * nef),
            nn.ReLU(True),

        ]

        self.conv_modules = nn.Sequential(*sequence)

        # make sure we return mu and logvar for latent code normal distribution
        self.enc_mu = nn.Linear(8 * nef, nlatent, bias=use_bias)
        self.enc_logvar = nn.Linear(8 * nef, nlatent, bias=use_bias)

    def forward(self, input):
        if len(GPU_IDs) > 0:
            conv_out = nn.parallel.data_parallel(self.conv_modules, input, GPU_IDs)
            mu = nn.parallel.data_parallel(self.enc_mu, conv_out, GPU_IDs)
            logvar = nn.parallel.data_parallel(self.enc_logvar, conv_out, GPU_IDs)
        else:
            conv_out = self.conv_modules(input)
            mu = self.enc_mu(conv_out)
            logvar = self.enc_logvar(conv_out)
        return mu, logvar
        # return mu.view(mu.size(0), -1), logvar.view(logvar.size(0), -1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1 and classname != 'ConditionalBatchNorm1d':
        m.weight.data.normal_(1.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)


def print_network(net, out_f=None):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    if out_f is not None:
        out_f.write(net.__repr__() + "\n")
        out_f.write('Total number of parameters: %d\n' % num_params)
        out_f.flush()

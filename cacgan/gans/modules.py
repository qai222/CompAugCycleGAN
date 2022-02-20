import torch.nn as nn


class TwoInputModule(nn.Module):
    """
    Superclass of all Modules that take two inputs
    """

    def forward(self, input1, input2):
        raise NotImplementedError


class MergeModule(TwoInputModule):
    """
    A (sort of) hacky way to create a module that takes two inputs (e.g. x and z)
    and returns one output (say o) defined as follows:
    o = module2.forward(module1.forward(x), z)
    Note that module2 MUST support two inputs as well.
    """

    def __init__(self, module1, module2):
        """
        module1 could be any module (e.g. Sequential of several modules)
        module2 must accept two inputs
        """
        super(MergeModule, self).__init__()
        self.module1 = module1
        self.module2 = module2

    def forward(self, input1, input2):
        output1 = self.module1.forward(input1)
        output2 = self.module2.forward(output1, input2)
        return output2


class TwoInputSequential(nn.Sequential, TwoInputModule):
    """
    A (sort of) hacky way to create a container that takes two inputs (e.g. x and z)
    and applies a sequence of modules (exactly like nn.Sequential) but MergeModule
    is one of its submodules it applies it to both inputs
    """

    def __init__(self, *args):
        super(TwoInputSequential, self).__init__(*args)

    def forward(self, input1, input2):
        """
        overloads forward function in parent class
        """
        for module in list(self._modules.values()):
            if isinstance(module, TwoInputModule):
                input1 = module.forward(input1, input2)
            else:
                input1 = module.forward(input1)
        return input1


class ConditionalBatchNorm1d(TwoInputModule):
    """
    taken from https://github.com/yanggeng1995/GAN-TTS/blob/master/models/generator.py
    """

    def __init__(self, x_dim, z_dim, eps=1e-5):
        super(ConditionalBatchNorm1d, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.eps = eps

        self.batch_norm = nn.BatchNorm1d(self.x_dim, affine=True)

        self.beta = nn.Sequential(
            nn.Linear(z_dim, x_dim, bias=True),
            nn.ReLU(True)
        )
        self.gamma = nn.Sequential(
            nn.Linear(z_dim, x_dim, bias=True),
            nn.ReLU(True)
        )

    def forward(self, inputs, noise):
        outputs = self.batch_norm(inputs)

        beta = self.beta(noise)
        gamma = self.gamma(noise)

        outputs = gamma * outputs + beta

        return outputs


class ConditionalResidualBlock(TwoInputModule):
    """
    A modified resnet block which allows for passing additional noise input
    to be used for conditional instance norm
    """

    def __init__(self, x_dim, z_dim, use_dropout, use_bias, norm_layer=ConditionalBatchNorm1d):
        super(ConditionalResidualBlock, self).__init__()
        self.block = self.build_block(x_dim, z_dim, norm_layer, use_dropout, use_bias)
        self.relu = nn.ReLU(True)

        for idx, module in enumerate(self.block):
            self.add_module(str(idx), module)

    @staticmethod
    def build_block(x_dim, z_dim, norm_layer, use_dropout, use_bias):
        block = []

        block += [
            MergeModule(
                nn.Linear(x_dim, x_dim, bias=use_bias),
                norm_layer(x_dim, z_dim)
            ),
            nn.ReLU(True)
        ]
        if use_dropout:
            block += [nn.Dropout(0.5)]

        block += [
            nn.Linear(x_dim, x_dim, bias=use_bias),
            nn.BatchNorm1d(x_dim, affine=True)  # the last one is not conditional
        ]

        return TwoInputSequential(*block)

    def forward(self, x, noise):
        out = self.block(x, noise)
        out = self.relu(x + out)
        return out


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_dropout, use_bias, norm_layer=nn.BatchNorm1d):
        super(ResnetBlock, self).__init__()
        self.linear_block = self.build_linear_block(dim, norm_layer, use_dropout, use_bias)
        self.relu = nn.ReLU(True)

    @staticmethod
    def build_linear_block(dim, norm_layer, use_dropout, use_bias):
        linear_block = []

        linear_block += [
            nn.Linear(dim, dim, bias=use_bias),
            nn.ReLU(True)
        ]

        if use_dropout:
            linear_block += [nn.Dropout(0.5)]

        linear_block += [
            nn.Linear(dim, dim, bias=use_bias),
            norm_layer(dim)
        ]
        return nn.Sequential(*linear_block)

    def forward(self, x):
        out = self.linear_block(x)

        out = self.relu(x + out)
        # out = self.x + out  # no activation
        return out

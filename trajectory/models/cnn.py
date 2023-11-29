import torch
from torch import nn
from einops.layers.torch import Rearrange

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = dilation*(kernel_size-1)
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, stride=stride)

    def forward(self, x):
        """
        shape of x: [total_seq, num_features, num_timesteps]
        """

        x = self.conv(x)
        last_n = (2*self.padding-self.kernel_size)//self.stride + 1
        if last_n> 0:
            return x[:, :, :-last_n]
        else:
            return x


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
        from https://github.com/jannerm/diffuser/blob/06b8e6a042e6a3312d50ed8048cba14afeab3085/diffuser/models/helpers.py#L46
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, stride, n_groups=8, causal=True):
        super().__init__()
        if causal:
            conv = CausalConv1d(inp_channels, out_channels, kernel_size, dilation=1, stride=stride)
        else:
            conv = nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride)

        self.block = nn.Sequential(
            conv,
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

class CausalDeConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride):
        super(CausalDeConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        # TODO: need to be double checked
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        """
        shape of x: [total_seq, num_features, num_timesteps]
        """
        x = self.conv(x)
        last_n = self.kernel_size-self.stride
        if last_n> 0:
            return x[:, :, :-last_n]
        else:
            return x


class DeConv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
        from https://github.com/jannerm/diffuser/blob/06b8e6a042e6a3312d50ed8048cba14afeab3085/diffuser/models/helpers.py#L46
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, stride, n_groups=8, causal=True):
        super().__init__()
        if causal:
            conv = CausalDeConv1d(inp_channels, out_channels, kernel_size, dilation=1, stride=stride),
        else:
            conv = nn.ConvTranspose1d(inp_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride)

        self.block = nn.Sequential(
            conv,
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)



class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size=3, pooling=1, stride=1, causal=True):
        super().__init__()

        second_kernel_size = kernel_size if stride==1 else stride

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size, 1, causal=causal),
            Conv1dBlock(out_channels, out_channels, second_kernel_size, stride=stride, causal=causal),
        ])

        if out_channels == inp_channels and stride == 1:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = nn.Conv1d(inp_channels, out_channels, kernel_size=1, stride=stride)

        if pooling==1:
            self.pooling = nn.Identity()
        else:
            self.pooling = nn.MaxPool1d(pooling, stride=pooling)

    def forward(self, input_dict):
        '''
            x : [ batch_size x horizon x inp_channels ]
            returns:
            out : [ batch_size x horizon x out_channels ]
        '''
        if isinstance(input_dict, dict):
            x = input_dict['x']
            input_mask = input_dict['input_mask']
        else:
            x = input_dict
            input_mask = None
        if input_mask is not None:
            x = x * input_mask.view(x.shape[0], x.shape[1], 1)
        x = torch.transpose(x, 1, 2)
        out = self.blocks[0](x)
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        out = self.pooling(out)
        return torch.transpose(out, 1, 2)


class ResidualTemporalDeConvBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size=3, pooling=1, stride=1, causal=True):
        super().__init__()
        second_kernel_size = kernel_size if stride==1 else stride

        self.blocks = nn.ModuleList([
            DeConv1dBlock(inp_channels, out_channels, kernel_size, 1, causal=causal),
            DeConv1dBlock(out_channels, out_channels, second_kernel_size, stride=stride, causal=causal),
        ])

        if out_channels == inp_channels and stride==1:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = nn.ConvTranspose1d(inp_channels, out_channels, kernel_size=stride, stride=stride)
        if pooling==1:
            self.pooling = nn.Identity()
        else:
            self.pooling = nn.MaxPool1d(pooling, stride=pooling)

    def forward(self, input_dict):
        '''
            x : [ batch_size x inp_channels x horizon ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        if isinstance(input_dict, dict):
            x = input_dict['x']
            input_mask = input_dict['input_mask']
        else:
            x = input_dict
            input_mask = None

        if input_mask is not None:
            x = x * input_mask.view(x.shape[0], x.shape[1], 1)
        x = torch.transpose(x, 1, 2)
        out = self.blocks[0](x)
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        out = self.pooling(out)
        return torch.transpose(out, 1, 2)

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchsort
except ModuleNotFoundError:
    # TODO: Install torchsort in other servers
    pass

"""
URIE excerpted from https://github.com/taeyoungson/urie/
"""

class Selector(nn.Module):
    def __init__(self, channel, reduction=16, crp_classify=False):
        super(Selector, self).__init__()
        self.spatial_attention = 4
        self.in_channel = channel * (self.spatial_attention ** 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((self.spatial_attention, self.spatial_attention))

        self.fc = nn.Sequential(
            nn.Linear(self.in_channel, self.in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
        )
        self.att_conv1 = nn.Linear(self.in_channel // reduction, self.in_channel)
        self.att_conv2 = nn.Linear(self.in_channel // reduction, self.in_channel)

    def forward(self, x):

        b, c, H, W = x.size()

        y = self.avg_pool(x).view(b, -1)
        y = self.fc(y)

        att1 = self.att_conv1(y).view(b, c, self.spatial_attention, self.spatial_attention)
        att2 = self.att_conv2(y).view(b, c, self.spatial_attention, self.spatial_attention)

        attention = torch.stack((att1, att2))
        attention = nn.Softmax(dim=0)(attention)

        att1 = F.interpolate(attention[0], scale_factor=(H / self.spatial_attention, W / self.spatial_attention), mode="nearest")
        att2 = F.interpolate(attention[1], scale_factor=(H / self.spatial_attention, W / self.spatial_attention), mode="nearest")

        return att1, att2


class SelectiveConv(nn.Module):
    def __init__(self, kernel_size, padding, bias, reduction, in_channels, out_channels, first=False):
        super(SelectiveConv, self).__init__()
        self.first = first
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.selector = Selector(out_channels, reduction=reduction)
        self.IN = nn.InstanceNorm2d(in_channels)
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        if self.first:
            f_input = x
            s_input = x
        else:
            f_input = self.BN(x)
            f_input = self.relu(f_input)

            s_input = self.IN(x)
            s_input = self.relu(s_input)

        out1 = self.conv1(f_input)
        out2 = self.conv2(s_input)

        out = out1 + out2

        att1, att2 = self.selector(out)
        out = torch.mul(out1, att1) + torch.mul(out2, att2)

        return out


class SKDown(nn.Module):
    def __init__(self, kernel_size, padding, bias, reduction, in_channels, out_channels, first=False):
        super(SKDown, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            SelectiveConv(kernel_size, padding, bias, reduction, in_channels, out_channels, first=first)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class SKUp(nn.Module):
    def __init__(self, kernel_size, padding, bias, reduction, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = SelectiveConv(kernel_size, padding, bias, reduction, in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        pass

    def forward(self, x):
        pass


class SKUNet(nn.Module):
    def __init__(self, num_channels, in_kernel_size, mid_kernel_size, bilinear=True):
        super(SKUNet, self).__init__()
        self.bilinear = bilinear

        self.down1 = nn.Conv2d(kernel_size=in_kernel_size, padding=in_kernel_size // 2, in_channels=num_channels, out_channels=32)
        self.down2 = SKDown(mid_kernel_size, mid_kernel_size // 2, False, 16, 32, 64)
        self.down3 = SKDown(mid_kernel_size, mid_kernel_size // 2, False, 16, 64, 64)
        self.up1 = SKUp(mid_kernel_size, mid_kernel_size // 2, False, 16, 128, 32, bilinear)
        self.up2 = SKUp(mid_kernel_size, mid_kernel_size // 2, False, 16, 64, 16, bilinear)
        self.up3 = nn.Conv2d(kernel_size=mid_kernel_size, padding=mid_kernel_size // 2, in_channels=16, out_channels=num_channels)

    def forward(self, x):
        x_origin = x
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.up3(x)

        return torch.add(x, x_origin)


# DiffDiST and helper classes

class LinearExp(nn.Module):
    def __init__(self, n_in, n_out):
        super(LinearExp, self).__init__()
        self.weight = nn.Parameter(torch.zeros(n_in, n_out))
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, x):
        A = torch.exp(self.weight)
        return x @ A + self.bias


class ChannelModulation(nn.Module):
    def __init__(self, n_in):
        super(ChannelModulation, self).__init__()
        self.weight = nn.Parameter(torch.ones(n_in), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(n_in), requires_grad=True)
    
    def forward(self, x):
        return (x.permute(0, 2, 3, 1) * self.weight + self.bias).permute(0, 3, 1, 2)


class TensorMixer(nn.Module):
    def __init__(self, n_in, init_alpha, init_beta):
        super(TensorMixer, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(n_in).fill_(init_alpha), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(n_in).fill_(init_beta), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(n_in), requires_grad=True)

    def forward(self, x1, x2):
        # x1, x2 are assumed to have shape (B x C x H x W)
        return (x1.permute(0, 2, 3, 1) * self.alpha + x2.permute(0, 2, 3, 1) * self.beta + self.bias).permute(0, 3, 1, 2)


class TransConvBlock(nn.Module):
    def __init__(self, n_in, n_hid):
        super(TransConvBlock, self).__init__()
        self.conv2d = nn.Conv2d(n_in, n_hid, kernel_size=3, padding=1)
        self.group_norm = nn.GroupNorm(n_hid, n_hid)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.group_norm(x)
        x = self.relu(x)
        return x


class InputTranformNet(nn.Module):
    def __init__(self, n_in, n_hid=18, n_layers=6):
        super(InputTranformNet, self).__init__()
        self.channel_mod = ChannelModulation(n_in)
        self.tau = nn.Parameter(torch.ones(1), requires_grad=True)
        
        # Make layers
        layers = []
        for idx in range(n_layers):
            if idx == 0:
                layers.append(TransConvBlock(n_in, n_hid))
            elif idx == n_layers - 1:
                layers.append(TransConvBlock(n_hid, n_in))
            else:
                layers.append(TransConvBlock(n_hid, n_hid))

        self.res_transform = nn.Sequential(*layers)

    def forward(self, x):
        res_x = self.res_transform(x)
        x = self.tau * x + (1 - self.tau) * res_x
        x = self.channel_mod(x)

        return x


class DiffDiST(nn.Module):
    def __init__(self, init_alpha=1.0, init_beta=0.0, init_gamma=1.0, num_groups=4):
        # Formula: D = gamma * (argsort(S)) + (1 - gamma) * (monotone(S)), where S = alpha * T + beta * 1 / C
        super(DiffDiST, self).__init__()
        self.num_groups = num_groups
        self.alpha = nn.Parameter(torch.tensor(init_alpha), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(init_beta), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor(init_gamma), requires_grad=True)

    def forward(self, x):
        print(self.alpha.data.item(), self.beta.data.item(), self.gamma.data.item())
        # x is assumed to have shape (B x 4 x H x W)
        inv_count = x[:, [0, 2], ...]  # (B x 2 x H x W)
        time_out = x[:, [1, 3], ...]  # (B x 2 x H x W)
        result = time_out + self.beta * inv_count
        return result


class EnhancedClassifier(nn.Module):
    def __init__(self, classifier: nn.Module, enhancer: nn.Module, return_input=False):
        super(EnhancedClassifier, self).__init__()
        self.classifier = classifier
        self.enhancer = enhancer
        self.return_input = return_input

    def forward(self, x):
        if self.return_input:
            if self.enhancer is None:
                return self.classifier(x), x
            else:
                x = self.enhancer(x)
                return self.classifier(x), x
        else:
            if self.enhancer is None:
                return self.classifier(x)
            else:
                return self.classifier(self.enhancer(x))


class ProjectionClassifier(nn.Module):
    def __init__(self, classifier: nn.Module, projector: nn.Module, return_mode: str):
        super(ProjectionClassifier, self).__init__()
        self.feature_extractor = nn.Sequential(*list(classifier.children())[:-1])
        self.projector = projector
        self.final_classifier = classifier.fc
        self.return_mode = return_mode

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        proj = x + self.projector(x)
        pred = self.final_classifier(proj)

        if self.return_mode == 'both':
            return pred, proj
        elif self.return_mode == 'pred':
            return pred
        elif self.return_mode == 'proj':
            return proj


class Projector(nn.Module):
    def __init__(self, dim, hid_dim):
        super(Projector, self).__init__()
        self.projector = nn.Sequential(nn.Linear(dim, hid_dim, bias=False),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(hid_dim, dim))  # output layer

    def forward(self, x):
        return self.projector(x)

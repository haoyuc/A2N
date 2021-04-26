import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
# import models.archs.arch_util as arch_util
from model import common
import matplotlib.pyplot as plt
import matplotlib


def make_model(args, parent=False):
    return PAN(args)

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out
    
class PAConv(nn.Module):

    def __init__(self, nf, k_size=3):

        super(PAConv, self).__init__()
        self.k1 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.k2 = nn.Conv2d(nf, nf, 1) # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution

    def forward(self, x):
        
        y = self.k1(x)
        y = self.lrelu(y)
        y = self.k2(y)
        y = self.sigmoid(y)

        # global i
        # i += 1
        # print(i)
        # # print(ax)
        # # print(conv1.shape)

        # # map = conv1[0].data.numpy()
        # print(y.std())
        # print(y.mean())
        # feature = torch.mean(y,1)[0,:,:].cpu().data.numpy()
        # print(feature.std())
        # print(feature.mean())
        # # max_ = float(feature.max())
        # # min_ = float(feature.min())
        # # if max_ > abs(min_):
        # #     m = -max_
        # # else:
        # #     m = -min_
        # # feature[0,0] = m       

        # # plt.matshow(feature, interpolation='nearest', cmap='seismic', norm=matplotlib.colors.Normalize(vmin=-0.001, vmax=0.001, clip=False))
        # # plt.matshow(feature, interpolation='nearest', cmap='seismic')
        # plt.matshow(feature, interpolation='nearest')
        # # plt.matshow(feature, interpolation='nearest', cmap='hot')
        # # plt.matshow(feature, interpolation='nearest', norm=matplotlib.colors.Normalize(vmin=-1, vmax=1, clip=False))
        # plt.colorbar()
        # plt.axis('off')
        # name = '/home/ubuntu/SR_Code/RCAN-master/RCAN_TrainCode/img_dy_2/rnan_mx_' + str(i) + '.jpg'
        # plt.savefig(name)


        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out


class DYconv(nn.Module):

    def __init__(self, nf, kernel_size=3, stride=1, reduction=4, K=2, t=30):
        super(DYconv, self).__init__()

        self.t=t
        self.K = K
        self.kernel_size = kernel_size
        self.stride = stride

        # self.ax = Dynamic(nf)
        # self.attention = attention2d(nf, K=K)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(nf, nf // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nf // reduction, self.K, bias=False),
        )
        self.conv1 = PAConv(nf)
        # self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, padding=(3 - 1) // 2, bias=False) 
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, padding=(3 - 1) // 2, bias=False) 

    def forward(self, x):
        a, b, c, d = x.shape

        y = self.avg_pool(x).view(a,b)
        y = self.fc(y)
        ax = F.softmax(y/self.t, dim = 1)
        # print(ax.shape)
        # ax = self.ax(x)
        # print(float(ax[1][1].cpu().data.numpy()))
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        
        
        # global i
        # print(i)
        # print(ax)
        # # print(conv1.shape)
        # print(ax)
        # # map = conv1[0].data.numpy()
        # feature = torch.mean(conv1,1)[0,:,:].cpu().data.numpy()
        # max_ = float(feature.max())
        # min_ = float(feature.min())
        # # if max_ > abs(min_):
        # #     m = -max_
        # # else:
        # #     m = -min_
        # # feature[0,0] = m       

        # # plt.matshow(feature, interpolation='nearest', cmap='seismic', norm=matplotlib.colors.Normalize(vmin=-0.001, vmax=0.001, clip=False))
        # # plt.matshow(feature, interpolation='nearest', cmap='seismic')
        # plt.matshow(feature, interpolation='nearest')
        # # plt.matshow(feature, interpolation='nearest', cmap='hot')
        # # plt.matshow(feature, interpolation='nearest', norm=matplotlib.colors.Normalize(vmin=-1, vmax=1, clip=False))
        # plt.colorbar()
        # plt.axis('off')
        # name = '/home/ubuntu/SR_Code/RCAN-master/RCAN_TrainCode/img_dy/rnan_mx_' + str(i) + '.jpg'
        # plt.savefig(name)


        

        out = conv1 * ax[:,0].view(a,1,1,1) + conv2 * ax[:,1].view(a,1,1,1)
        # out = conv2
        return out


class SCPA(nn.Module):
    
    """SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
        Github: https://github.com/MCG-NKU/SCNet
    """

    def __init__(self, nf, reduction=1, stride=1, dilation=1):
        super(SCPA, self).__init__()
        group_width = nf // reduction
        
        # self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        
        # self.k1 = nn.Sequential(
        #             nn.Conv2d(
        #                 group_width, group_width, kernel_size=3, stride=stride,
        #                 padding=dilation, dilation=dilation,
        #                 bias=False)
        #             )
        
        self.PAConv = DYconv(group_width)
        
        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # global i
        # i += 1
        # print(i)
        residual = x

        # out_a= self.conv1_a(x)
        out_b = self.conv1_b(x)
        # out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        # out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        # out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        # out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out = self.conv3(out_b)
        out += residual

        return out
    
class PAN(nn.Module):
    
    # def __init__(self, in_nc, out_nc, nf, unf, nb, scale=4):
    def __init__(self, args, conv=common.default_conv):
        super(PAN, self).__init__()
        
        in_nc  = 3
        out_nc = 3
        # nf = args.n_feats
        nf  = 40
        unf = 24
        nb  = 16
        scale = args.scale[0]


        # SCPA
        SCPA_block_f = functools.partial(SCPA, nf=nf, reduction=1)
        self.scale = scale
        
        ### first convolution
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        
        ### main blocks
        self.SCPA_trunk = make_layer(SCPA_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, unf, 3, 1, 1, bias=True)
        self.att1 = PA(unf)
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
        
        if self.scale == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att2 = PA(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            
        self.conv_last = nn.Conv2d(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # print('='*30)
        
        # global i
        # i = 0
        
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.SCPA_trunk(fea))
        fea = fea + trunk
        
        if self.scale == 2 or self.scale == 3:
            fea = self.upconv1(F.interpolate(fea, scale_factor=self.scale, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
        elif self.scale == 4:
            fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
            fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att2(fea))
            fea = self.lrelu(self.HRconv2(fea))
        
        out = self.conv_last(fea)
        
        ILR = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        out = out + ILR


        # print('='*30)

        return out
 
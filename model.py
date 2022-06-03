import copy
import torch
import torch.nn as nn
from da_att import *
import numpy as np
from MGR_model import *

####################################################################################################################

def create_conv_kernel(in_channels, out_channels, kernel_size=3, avg=0.0, std=0.1):
    # [out_channels, in_channels, kernel_size, kernel_size]
    kernel_arr = np.random.normal(loc=avg, scale=std, size=(out_channels, in_channels, kernel_size, kernel_size))
    kernel_arr = kernel_arr.astype(np.float32)
    kernel_tensor = torch.from_numpy(kernel_arr)
    kernel_params = nn.Parameter(data=kernel_tensor.contiguous(), requires_grad=True)
    print(kernel_params.type())
    return kernel_params


def create_conv_bias(channels):
    # [channels, ]
    bias_arr = np.zeros(channels, np.float32)
    assert bias_arr.shape[0] % 2 == 1

    bias_arr[bias_arr.shape[0] // 2] = 1.0
    bias_tensor = torch.from_numpy(bias_arr)
    bias_params = nn.Parameter(data=bias_tensor.contiguous(), requires_grad=True)

    return bias_params


def create_mapping_kernel(kernel_size=7):
    # [kernel_size * kernel_size, kernel_size, kernel_size]
    kernel_arr = np.zeros((kernel_size * kernel_size, kernel_size, kernel_size), np.float32)
    for h in range(kernel_arr.shape[1]):
        for w in range(kernel_arr.shape[2]):
            kernel_arr[h * kernel_arr.shape[2] + w, h, w] = 1.0

    # [kernel_size * kernel_size, 1, kernel_size, kernel_size]
    kernel_tensor = torch.from_numpy(np.expand_dims(kernel_arr, axis=1))
    kernel_params = nn.Parameter(data=kernel_tensor.contiguous(), requires_grad=False)
    print(kernel_params.type())

    return kernel_params


class Convblock(nn.Module):
    def __init__(self, inChannals, outChannals):
        super(Convblock, self).__init__()
        self.bn1 = nn.BatchNorm2d(outChannals)
        self.conv2 = nn.Conv2d(inChannals, outChannals, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()

    def forward(self, x):
        out = self.conv2(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class BR(nn.Module):
    def __init__(self, num_class):
        super(BR, self).__init__()

        self.shortcut = nn.Sequential(nn.Conv2d(num_class, num_class, 3, padding=1, bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(num_class, num_class, 3, padding=1, bias=False))

    def forward(self, x):
        return x + self.shortcut(x)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def downsample():
    return nn.MaxPool2d(kernel_size=2, stride=2)

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def deconv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

class ResEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = out + residual
        out = self.relu(out)
        return out

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


###############################################################################################################
class RefUnet(nn.Module):
    def __init__(self, in_ch, inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)

        self.conv1 = nn.Conv2d(inc_ch, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64, 1, 3, padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.upscore2 = nn.Upsample(scale_factor=2, align_corners=True)

    def forward(self, x):
        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        # hx5 = self.cpam(hx5)

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx, hx4), 1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx, hx3), 1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx, hx2), 1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx, hx1), 1))))

        residual = self.conv_d0(d1)
        out1 = x + residual
        out = nn.Sigmoid()(out1)

        return out



def feature_fusion(out1, out2):
    output2 = F.log_softmax(out2, dim=1)
    out1_bg = torch.zeros([out1.shape[0], 1, out1.shape[2], out1.shape[3]]).cuda()
    out1_disc = torch.zeros([out1.shape[0], 1, out1.shape[2], out1.shape[3]]).cuda()
    out2_layer = torch.zeros([out2.shape[0], 1, out2.shape[2], out2.shape[3]]).cuda()
    out1_bg[:, 0, :, :] = out1[:, 0, :, :]
    out1_disc[:, 0, :, :] = out1[:, 2, :, :]
    out2_layer[:, :, :, :] = out2[:, 1:, :, :]
    out = torch.cat([out1_bg, out2_layer, out1_disc], 1)
    return output2, out


def img2df(img, mask):
    img[mask == 0] = 0
    img[mask == 2] = 0
    return img


class base(nn.Module):
    def __init__(self, channels=64, pn_size=3, kernel_size=3, avg=0.0, std=0.1):
        """
        :param channels: the basic channels of feature maps.
        :param pn_size: the size of propagation neighbors.
        :param kernel_size: the size of kernel.
        :param avg: the mean of normal initialization.
        :param std: the standard deviation of normal initialization.
        """
        super(base, self).__init__()
        self.kernel_size = kernel_size

        self.conv1_kernel = create_conv_kernel(in_channels=3, out_channels=channels,
                                               kernel_size=self.kernel_size, avg=avg, std=std)  # ##
        # self.conv2_kernel = create_conv_kernel(in_channels=channels, out_channels=channels,
        #                                        kernel_size=self.kernel_size, avg=avg, std=std)
        # self.conv3_kernel = create_conv_kernel(in_channels=channels, out_channels=channels,
        #                                        kernel_size=self.kernel_size, avg=avg, std=std)
        self.conv4_kernel = create_conv_kernel(in_channels=channels, out_channels=2 * channels,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        # self.conv5_kernel = create_conv_kernel(in_channels=2*channels, out_channels=2*channels,
        #                                        kernel_size=self.kernel_size, avg=avg, std=std)
        # self.conv6_kernel = create_conv_kernel(in_channels=2*channels, out_channels=2*channels,
        #                                        kernel_size=self.kernel_size, avg=avg, std=std)
        self.conv7_kernel = create_conv_kernel(in_channels=2 * channels, out_channels=pn_size * pn_size,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        self.conv7_bias = create_conv_bias(pn_size * pn_size)
        self.bn1 = nn.BatchNorm2d(channels)
        # self.bn2 = nn.BatchNorm2d(channels)
        # self.bn3 = nn.BatchNorm2d(channels)
        self.bn4 = nn.BatchNorm2d(2 * channels)
        # self.bn5 = nn.BatchNorm2d(2*channels)
        # self.bn6 = nn.BatchNorm2d(2*channels)
        self.bn7 = nn.BatchNorm2d(pn_size * pn_size)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, input_src, input_thick, input_thin):
        input_all = torch.cat((input_src, input_thick, input_thin), dim=1)  # [b, 3, h, w] ##
        assert input_all.size()[1] == 3  # ##

        fm_1 = F.conv2d(input_all, self.conv1_kernel, padding=self.kernel_size // 2)
        fm_1 = self.bn1(fm_1)
        fm_1 = self.relu(fm_1)
        # fm_2 = F.conv2d(fm_1, self.conv2_kernel, padding=self.kernel_size//2)
        # fm_2 = self.bn2(fm_2)
        # fm_2 = self.relu(fm_2)
        # fm_3 = F.conv2d(fm_2, self.conv3_kernel, padding=self.kernel_size//2)
        # fm_3 = self.bn3(fm_3)
        # fm_3 = self.relu(fm_3)
        fm_4 = F.conv2d(fm_1, self.conv4_kernel, padding=self.kernel_size // 2)
        fm_4 = self.bn4(fm_4)
        fm_4 = self.relu(fm_4)
        # fm_5 = F.conv2d(fm_4, self.conv5_kernel, padding=self.kernel_size//2)
        # fm_5 = self.bn5(fm_5)
        # fm_5 = self.relu(fm_5)
        # fm_6 = F.conv2d(fm_5, self.conv6_kernel, padding=self.kernel_size//2)
        # fm_6 = self.bn6(fm_6)
        # fm_6 = self.relu(fm_6)
        fm_7 = F.conv2d(fm_4, self.conv7_kernel, self.conv7_bias, padding=self.kernel_size // 2)
        fm_7 = self.bn7(fm_7)
        fm_7 = F.relu(fm_7)

        return F.softmax(fm_7, dim=1)  # [b, pn_size * pn_size, h, w]


class adaptive_aggregation(nn.Module):
    def __init__(self, pn_size=3):
        """
        :param pn_size: the size of propagation neighbors.
        """
        super(adaptive_aggregation, self).__init__()
        self.kernel_size = pn_size
        self.weight = create_mapping_kernel(kernel_size=self.kernel_size)

    def forward(self, input_thick, input_thin, agg_coeff):
        assert input_thick.size()[1] == 1 and input_thin.size()[1] == 1
        input_sal = torch.max(input_thick, input_thin)
        map_sal = F.conv2d(input_sal, self.weight, padding=self.kernel_size // 2)
        # map_sal_inv = 1.0 - map_sal
        assert agg_coeff.size() == map_sal.size()

        prod_sal = torch.sum(map_sal * agg_coeff, dim=1).unsqueeze(1)
        # prod_sal = F.sigmoid(prod_sal)
        # prod_sal_inv = torch.sum(map_sal_inv * agg_coeff, dim=1).unsqueeze(1)

        return prod_sal  # [b, 1, h, w]


class fusion(nn.Module):
    def __init__(self, channels=64, pn_size=3, kernel_size=3, avg=0.0, std=0.1):
        super(fusion, self).__init__()
        self.backbone = base(channels, pn_size, kernel_size, avg, std)
        self.adagg = adaptive_aggregation(pn_size)

    def forward(self, input_src, input_thick, input_thin):  # ##
        agg_coeff = self.backbone(input_src, input_thick, input_thin)  # ##
        prod_sal = self.adagg(input_thick, input_thin, agg_coeff)

        return prod_sal


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class coarse_net(nn.Module):
    def __init__(self):  ##########
        super(coarse_net, self).__init__()
        self.enc_input = ResEncoder(1, 32)
        self.encoder1 = ResEncoder(32, 64)
        self.encoder2 = ResEncoder(64, 128)
        self.encoder3 = ResEncoder(128, 256)
        self.encoder4 = ResEncoder(256, 512)
        self.encoder5 = ResEncoder(512, 1024)
        self.downsample = downsample()
        self.decoder5 = Decoder(1024, 512)
        self.decoder4 = Decoder(512, 256)
        self.decoder3 = Decoder(256, 128)
        self.decoder2 = Decoder(128, 64)
        self.decoder1 = Decoder(64, 32)
        self.deconv5 = deconv(1024 + 1024, 512)
        self.deconv4 = deconv(512, 256)
        self.deconv3 = deconv(256, 128)
        self.deconv2 = deconv(128, 64)
        self.deconv1 = deconv(64, 32)
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        initialize_weights(self)
        # self.pam = CPAM(1024)
        self.cpam = CPAM(1024)
        self.mgr = MGR_Module(1024, 1024)

    def forward(self, inputs):
        enc_input = self.enc_input(inputs)
        down1 = self.downsample(enc_input)

        enc1 = self.encoder1(down1)
        down2 = self.downsample(enc1)

        enc2 = self.encoder2(down2)
        down3 = self.downsample(enc2)

        enc3 = self.encoder3(down3)
        down4 = self.downsample(enc3)

        enc4 = self.encoder4(down4)

        down_pad = False
        right_pad = False
        if enc4.size()[2] % 2 == 1:
            enc4 = F.pad(enc4, (0, 0, 0, 1))
            down_pad = True
        if enc4.size()[3] % 2 == 1:
            enc4 = F.pad(enc4, (0, 1, 0, 0))
            right_pad = True

        down5 = self.downsample(enc4)

        input_feature = self.encoder5(down5)

        # Do Attenttion operations here
        input_feature1 = self.mgr(input_feature)
        input_feature2 = self.cpam(input_feature)
        input_feature = torch.cat([input_feature1, input_feature2], 1)

        up5 = self.deconv5(input_feature)
        up5 = torch.cat((enc4, up5), dim=1)

        if down_pad and (not right_pad):
            up5 = up5[:, :, :-1, :]
        if (not down_pad) and right_pad:
            up5 = up5[:, :, :, :-1]
        if down_pad and right_pad:
            up5 = up5[:, :, :-1, :-1]

        dec5 = self.decoder5(up5)
        # Do decoder operations here
        up4 = self.deconv4(dec5)
        up4 = torch.cat((enc3, up4), dim=1)
        dec4 = self.decoder4(up4)

        up3 = self.deconv3(dec4)
        up3 = torch.cat((enc2, up3), dim=1)
        dec3 = self.decoder3(up3)

        up2 = self.deconv2(dec3)
        up2 = torch.cat((enc1, up2), dim=1)
        dec2 = self.decoder2(up2)

        up1 = self.deconv1(dec2)
        up1 = torch.cat((enc_input, up1), dim=1)
        dec1 = self.decoder1(up1)

        final1 = self.final(dec1)
        final = nn.Sigmoid()(final1)

        return final



class segnet(nn.Module):
    def __init__(self):
        super(segnet, self).__init__()
        self.fu = fusion()
        self.refu = RefUnet(1, 1)
        self.g1 = coarse_net()

    def forward(self, x):  # ##
        x1 = self.g1(x)
        y2 = self.refu(x1)
        y1 = self.fu(x[:, :1, :, :], x1, y2)
        #y2 = self.refu(x1)
        out = torch.max(y1, y2)
        #return x1, y1, y2
        return x1, y1, y2, out




# x = torch.randn(1, 1, 256, 256)
# mgr = segnet()
# _, _, _, a = mgr(x)
#
# print(a.shape)


class coarse_net1(nn.Module):
    def __init__(self):  ##########
        super(coarse_net1, self).__init__()
        self.enc_input = ResEncoder(1, 32)
        self.encoder1 = ResEncoder(32, 64)
        self.encoder2 = ResEncoder(64, 128)
        self.encoder3 = ResEncoder(128, 256)
        self.encoder4 = ResEncoder(256, 512)
        self.encoder5 = ResEncoder(512, 1024)
        self.downsample = downsample()
        self.decoder5 = Decoder(1024, 512)
        self.decoder4 = Decoder(512, 256)
        self.decoder3 = Decoder(256, 128)
        self.decoder2 = Decoder(128, 64)
        self.decoder1 = Decoder(64, 32)
        self.deconv5 = deconv(1024 + 1024, 512)
        self.deconv4 = deconv(512, 256)
        self.deconv3 = deconv(256, 128)
        self.deconv2 = deconv(128, 64)
        self.deconv1 = deconv(64, 32)
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        initialize_weights(self)
        # self.pam = CPAM(1024)
        self.cpam = CPAM(1024)
        self.mgr = MGR_Module(1024, 1024)

    def forward(self, inputs):
        enc_input = self.enc_input(inputs)
        down1 = self.downsample(enc_input)

        enc1 = self.encoder1(down1)
        down2 = self.downsample(enc1)

        enc2 = self.encoder2(down2)
        down3 = self.downsample(enc2)

        enc3 = self.encoder3(down3)
        down4 = self.downsample(enc3)

        enc4 = self.encoder4(down4)

        down_pad = False
        right_pad = False
        if enc4.size()[2] % 2 == 1:
            enc4 = F.pad(enc4, (0, 0, 0, 1))
            down_pad = True
        if enc4.size()[3] % 2 == 1:
            enc4 = F.pad(enc4, (0, 1, 0, 0))
            right_pad = True

        down5 = self.downsample(enc4)

        input_feature = self.encoder5(down5)

        # Do Attenttion operations here
        #input_feature = self.mgr(input_feature)
        #input_feature = self.cpam(input_feature)
        input_feature1 = self.mgr(input_feature)
        input_feature2 = self.cpam(input_feature)
        input_feature = torch.cat([input_feature1, input_feature2], 1)

        up5 = self.deconv5(input_feature)
        up5 = torch.cat((enc4, up5), dim=1)

        if down_pad and (not right_pad):
            up5 = up5[:, :, :-1, :]
        if (not down_pad) and right_pad:
            up5 = up5[:, :, :, :-1]
        if down_pad and right_pad:
            up5 = up5[:, :, :-1, :-1]

        dec5 = self.decoder5(up5)
        # Do decoder operations here
        up4 = self.deconv4(dec5)
        up4 = torch.cat((enc3, up4), dim=1)
        dec4 = self.decoder4(up4)

        up3 = self.deconv3(dec4)
        up3 = torch.cat((enc2, up3), dim=1)
        dec3 = self.decoder3(up3)

        up2 = self.deconv2(dec3)
        up2 = torch.cat((enc1, up2), dim=1)
        dec2 = self.decoder2(up2)

        up1 = self.deconv1(dec2)
        up1 = torch.cat((enc_input, up1), dim=1)
        dec1 = self.decoder1(up1)

        final1 = self.final(dec1)
        final = nn.Sigmoid()(final1)

        return final

x = torch.randn(1, 1, 256, 256)
mgr = coarse_net1()
a = mgr(x)
#
print(a.shape)
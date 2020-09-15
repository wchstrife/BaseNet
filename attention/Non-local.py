import torch
from torch import nn
from torch.nn import functional as F

# embedded_gaussian

class _NonLocalBlockND(nn.Module):

    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:                     # inter_channels是1*1卷积降维后的特征图数量
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:                                  # 根据输入的是几维的数据，判断采用nd卷积
            conv_nd = nn.Conv3d 
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = max_pool_layer

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w), t是3D卷积的t帧视频
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        print("g_x.shape : ", g_x.shape)

        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        print("theta_x.shape : ", theta_x.shape)

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        print("phi_x.shape : ", phi_x.shape)

        f = torch.matmul(theta_x, phi_x)        
        f_div_C = F.softmax(f, dim=-1)
        
        print("f.shape : ", f.shape)
        print("f_div_C.shape : ", f_div_C.shape)

        # if self.store_last_batch_nl_map:
        #     self.nl_map = f_div_C

        y = torch.matmul(f_div_C, g_x)
        print("y.shape : ", y.shape)
        y = y.permute(0, 2, 1).contiguous()
        print("y.shape : ", y.shape)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        print("y.shape : ", y.shape)
        W_y = self.W(y)                     # 1*1卷积升维
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z

    
class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

if __name__ == '__main__':

    # for (sub_sample_, bn_layer_) in [(True, True), (False, False), (True, False), (False, True)]:
    #     img = torch.zeros(2, 3, 20)
    #     net = NONLocalBlock1D(3, sub_sample=sub_sample_, bn_layer=bn_layer_)
    #     out = net(img)
    #     print(out.size())

    #     img = torch.zeros(2, 3, 20, 20)
    #     net = NONLocalBlock2D(3, sub_sample=sub_sample_, bn_layer=bn_layer_)
    #     out = net(img)
    #     print(out.size())

    #     img = torch.randn(2, 3, 8, 20, 20)
    #     net = NONLocalBlock3D(3, sub_sample=sub_sample_, bn_layer=bn_layer_)
    #     out = net(img)
    #     print(out.size())

    img = torch.zeros(2, 3, 20, 20)
    net = NONLocalBlock2D(3, sub_sample=True, bn_layer=True)
    out = net(img)
    print(out.size())
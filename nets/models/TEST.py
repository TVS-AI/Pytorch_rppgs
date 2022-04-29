import torch
import torch.nn as nn
from nets.modules.vit_pytorch.max_vit import MaxViT,MaxVit_GAN_layer,MaxViT_layer,CrissCrossAttention,FeedForward,MBConv
from einops import rearrange
from einops.layers.torch import Rearrange
import math
import torch.nn.functional as F

class TEST(nn.Module):
    def __init__(self):
        super(TEST, self).__init__()

        length = 32
        height,width = (128,128)

        self.main_seq_stem = nn.Sequential(
            Rearrange('b c l h w -> (b l) c h w'),
            Conv2d_cd(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            Conv2d_cd(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            # nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            # nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            Rearrange('(b l) c h w -> b c l (h w)', l=length)
        )
        self.main_seq_max_1 = nn.Sequential(
            MaxViT_layer(layer_depth=2, layer_dim_in=3, layer_dim=32,
                         kernel=(1,32),dilation=(1,32),padding=0,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,flag=False)
        )
        self.main_seq_max_2 = nn.Sequential(
            MaxViT_layer(layer_depth=1, layer_dim_in=32, layer_dim=64,
                         kernel=3, dilation=1, padding=1,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,flag=True)
        )

        self.ptt_seq_stem = nn.Sequential(
            Rearrange('b c l h w -> (b h) c l w'),
            Conv2d_cd(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            Conv2d_cd(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            # nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            # nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            Rearrange('(b h) c l w -> b c h (l w)', h=height)
        )
        self.ptt_seq_max_1 = nn.Sequential(
            MaxViT_layer(layer_depth=2, layer_dim_in=3, layer_dim=32,
                         kernel=(1,8),dilation=(1,32),padding=0,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,flag=True))
        self.ptt_seq_max_2 = nn.Sequential(
            MaxViT_layer(layer_depth=1, layer_dim_in=32, layer_dim=64,
                         kernel=3, dilation=1, padding=1,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,flag=True),
            # Rearrange('b c h e -> b c (h e)'),
            # nn.AdaptiveAvgPool2d((64, 128))
        )

        self.bvp_seq_stem = nn.Sequential(
            Rearrange('b c l h w -> (b w) c l h'),
            Conv2d_cd(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            Conv2d_cd(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            # nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            # nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            Rearrange('(b w) c l h -> b c w (l h)', w=width)
        )
        self.bvp_seq_max_1 = nn.Sequential(
            MaxViT_layer(layer_depth=2, layer_dim_in=3, layer_dim=32,
                         kernel=(1,8),dilation=(1,32),padding=0,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,flag=True))
        self.bvp_seq_max_2 = nn.Sequential(
            MaxViT_layer(layer_depth=1, layer_dim_in=32, layer_dim=64,
                         kernel=3, dilation=1, padding=1,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,flag=True),
            # Rearrange('b c w e -> b c (w e)'),
            # nn.AdaptiveAvgPool2d((64,128))
        )
        self.max_vit = MaxViT_layer(layer_depth=2, layer_dim_in=32, layer_dim=32,
                         kernel=3, dilation=1, padding=1,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,flag=False)
        self.max_vit_2 = MaxViT_layer(layer_depth=1, layer_dim_in=128, layer_dim=256,
                                    kernel=3, dilation=1, padding=1,
                                    mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=2, dim_head=32,
                                    dropout=0.1,flag=True )
        # self.max_vit_gan_layer_1 = MaxVit_GAN_layer(layer_depth=1, layer_dim_in=64, layer_dim=32,
        #                                             mbconv_expansion_rate=4,
        #                                             mbconv_shrinkage_rate=0.25, w=2, dim_head=32, dropout=0.1,
        #                                             scale_factor=(2,1))
        # self.max_vit_gan_layer_2 = MaxVit_GAN_layer(layer_depth=1, layer_dim_in=32, layer_dim=32,
        #                                             mbconv_expansion_rate=4,
        #                                             mbconv_shrinkage_rate=0.25, w=2, dim_head=32, dropout=0.1,
        #                                             scale_factor=(2, 1))
        # self.max_vit_gan_layer_3 = MaxVit_GAN_layer(layer_depth=1, layer_dim_in=32, layer_dim=32,
        #                                             mbconv_expansion_rate=4,
        #                                             mbconv_shrinkage_rate=0.25, w=2, dim_head=32, dropout=0.1,
        #                                             scale_factor=(2, 1))
        self.adaptation = nn.AdaptiveAvgPool2d((4,2))
        # self.conv2d = nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1)
        self.up_1 = UpBlock(128,64)
        self.up_2 = UpBlock(64, 32)
        self.up_3 = UpBlock(32, 3)
        self.up_4 = UpBlock(3, 1)

        self.sa_main = SpatialAttention()
        self.sa_bvp = SpatialAttention()
        self.sa_ptt = SpatialAttention()

        self.adaptive = nn.AdaptiveAvgPool2d((32,16))
        self.be_conv1d = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding="same")
        self.out_conv1d = nn.Conv1d(in_channels=32,out_channels=1,kernel_size=1)

        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(32,64)



    def forward(self,x):
        main = self.main_seq_stem(x)
        main = self.main_seq_max_1(main)
        #ver1
        # main = self.sa_main(main)
        #ver2
        main_att = self.sa_main(main)
        main = main_att*main + main
        main = self.adaptive(main)
        main = rearrange(main,'b c l (w h) -> b c l w h',w = 4, h = 4)
        # main = self.main_seq_max_2(main)

        bvp = self.bvp_seq_stem(x)
        bvp = self.bvp_seq_max_1(bvp)
        #ver1
        bvp = self.sa_bvp(bvp)
        #ver2
        # bvp_att = self.sa_bvp(bvp)
        # bvp = bvp_att*bvp + bvp
        bvp = rearrange(bvp, 'b c w (l h) -> b c l w h', l=4, h=4)


        ptt = self.ptt_seq_stem(x)
        ptt = self.ptt_seq_max_1(ptt)
        #ver1
        ptt = self.sa_bvp(ptt)
        #ver2
        # ptt_att = self.sa_bvp(ptt)
        # ptt = ptt_att*ptt + ptt
        ptt = rearrange(ptt, 'b c h (l w) -> b c l w h', l=4, w=4)

        att = ptt@bvp
        main = main * F.interpolate(att,scale_factor=(8,1,1)) + main


        # out = []
        # batch, channel, length, e = main.shape
        # for i in range(length):
        #     out.append(torch.unsqueeze(torch.cat([main[:, :, i, :], ptt, bvp], dim=2), dim=2))
        # out = torch.cat(out,dim=2)
        main = rearrange(main,'b c l w h -> b c l (w h)')
        out = self.max_vit(main)
        # out = self.adaptation(out)
        # out = self.max_vit_2(out)
        # out = self.max_vit_gan_layer_1(out)
        # out = self.max_vit_gan_layer_2(out)
        # out = self.max_vit_gan_layer_3(out)
        # out = self.adaptation(out)
        # out = self.conv2d(out)
        # out = torch.squeeze(out)
        # out = self.up_1(out)
        # out = self.up_2(out)
        # out = self.up_3(out)
        # out = self.up_4(out)
        out = torch.squeeze(out)
        out = torch.mean(out,dim = -1)
        out_att = self.be_conv1d(out)
        out = (1 + self.sigmoid(out_att)) * out
        out = self.out_conv1d(out)
        out = torch.squeeze(out)
        out = self.linear(out)
        return out

class UpBlock(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(UpBlock, self).__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_dim,out_channels=out_dim,kernel_size=(1,2),stride=(1,2)),
            nn.Conv2d(in_channels=out_dim,out_channels=out_dim,kernel_size=3,stride=(2,1),padding=(1,1)),
            nn.BatchNorm2d(out_dim),
            nn.Conv2d(in_channels=out_dim,out_channels=out_dim,kernel_size=3,stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(out_dim),
            nn.SELU(inplace=True)
        )
    def forward(self,x):
        return self.seq(x)


class TEST_1(nn.Module):
    def __init__(self):
        super(TEST_1, self).__init__()
        self.dim = [3, 32, 64, 128, 192, 256, 320 ]
                   #0  1   2   3    4    5

        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=2,padding=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3,padding=1)
        )
        i = 0
        self.max_vit_layer_1 = MaxViT_layer(layer_depth=2, layer_dim_in=self.dim[i], layer_dim=self.dim[i+1], mbconv_expansion_rate=4,
                                            mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1)
        i +=1
        self.max_vit_layer_2 = MaxViT_layer(layer_depth=2, layer_dim_in=self.dim[i], layer_dim=self.dim[i+1], mbconv_expansion_rate=4,
                                            mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1)
        i += 1
        self.max_vit_layer_3 = MaxViT_layer(layer_depth=5, layer_dim_in=self.dim[i], layer_dim=self.dim[i+1], mbconv_expansion_rate=4,
                                            mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1)
        i += 1
        self.max_vit_layer_4 = MaxViT_layer(layer_depth=2, layer_dim_in=self.dim[i], layer_dim=self.dim[i+1], mbconv_expansion_rate=4,
                                            mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1)
        i += 1
        self.max_vit_layer_5 = MaxViT_layer(layer_depth=2, layer_dim_in=self.dim[i], layer_dim=self.dim[i+1], mbconv_expansion_rate=4,
                                            mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1)
        i += 1
        self.max_vit_layer_6 = MaxViT_layer(layer_depth=2, layer_dim_in=self.dim[i], layer_dim=self.dim[i+1], mbconv_expansion_rate=4,
                                            mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1)
        self.max_vit_gan_layer_1 = MaxVit_GAN_layer(layer_depth=2,layer_dim_in=self.dim[i+1], layer_dim=self.dim[i],mbconv_expansion_rate=4,
                                                    mbconv_shrinkage_rate=0.25,w=4,dim_head=32,dropout=0.1)
        i -=1
        self.max_vit_gan_layer_2 = MaxVit_GAN_layer(layer_depth=2,layer_dim_in=self.dim[i+1], layer_dim=self.dim[i],mbconv_expansion_rate=4,
                                                    mbconv_shrinkage_rate=0.25,w=4,dim_head=32,dropout=0.1)
        i -= 1
        self.max_vit_gan_layer_3 = MaxVit_GAN_layer(layer_depth=2, layer_dim_in=self.dim[i + 1], layer_dim=self.dim[i],
                                                    mbconv_expansion_rate=4,
                                                    mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1)
        i -= 1
        self.max_vit_gan_layer_4 = MaxVit_GAN_layer(layer_depth=2, layer_dim_in=self.dim[i + 1], layer_dim=self.dim[i],
                                                    mbconv_expansion_rate=4,
                                                    mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1)
        self.lat = nn.Conv1d(in_channels=self.dim[i+1],out_channels=1,kernel_size=1)
        # self.last0 = nn.Sequential(
        #     nn.Conv2d(in_channels=)
        # )

    def forward(self,x):
        batch, channel, length, height, width = x.shape
        x = rearrange(x,'b c l h w -> (b l) c h w')
        x = self.conv_stem(x)
        x = self.max_vit_layer_1(x)
        x = self.max_vit_layer_2(x)
        x = self.max_vit_layer_3(x)
        # x = self.max_vit_layer_4(x)
        x = rearrange(x,'(b l) c h w -> b c l (h w)',l=32)
        x = self.max_vit_layer_4(x)
        x = self.max_vit_layer_5(x)
        x = self.max_vit_layer_6(x)
        x = self.max_vit_gan_layer_1(x)
        x = self.max_vit_gan_layer_2(x)
        x = self.max_vit_gan_layer_3(x)
        # x = self.max_vit_gan_layer_4(x)
        x = torch.mean(x,dim=-1)
        x = self.lat(x)
        x = torch.squeeze(x)

        # at = self.Criss_1(x)
        # at = self.Criss_2(at)

        # x = at + x
        # x = rearrange(x, 'b d x y -> b x y d')
        # fe = self.feed_Forward(x)
        # fe = x + fe

        # x = self.max_vit_gan_layer_1(x)
        return x

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta *  out_diff


class SpatialAttention(nn.Module):
    def __init__(self, kernel=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)
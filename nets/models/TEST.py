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
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            Rearrange('(b l) c h w -> b c l (h w)', l=length)
        )
        self.main_seq_max_1 = nn.Sequential(
            MaxViT_layer(layer_depth=2, layer_dim_in=3, layer_dim=32,
                         kernel=(1,32),dilation=(1,32),padding=0,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,flag=False)
        )

        self.ptt_seq_stem = nn.Sequential(
            Rearrange('b c l h w -> (b h) c l w'),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            Rearrange('(b h) c l w -> b c h (l w)', h=height)
        )
        self.ptt_seq_max_1 = nn.Sequential(
            MaxViT_layer(layer_depth=2, layer_dim_in=3, layer_dim=32,
                         kernel=(1,8),dilation=(1,32),padding=0,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,flag=True))

        self.bvp_seq_stem = nn.Sequential(
            Rearrange('b c l h w -> (b w) c l h'),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1),
            Rearrange('(b w) c l h -> b c w (l h)', w=width)
        )
        self.bvp_seq_max_1 = nn.Sequential(
            MaxViT_layer(layer_depth=2, layer_dim_in=3, layer_dim=32,
                         kernel=(1,8),dilation=(1,32),padding=0,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,flag=True))
        self.max_vit = MaxViT_layer(layer_depth=2, layer_dim_in=1, layer_dim=32,
                         kernel=3, dilation=1, padding=1,
                         mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, w=4, dim_head=32, dropout=0.1,flag=False)
        self.adaptation = nn.AdaptiveAvgPool2d((4,2))

        self.sa_main = SpatialAttention()
        self.sa_bvp = SpatialAttention()
        self.sa_ptt = SpatialAttention()

        self.adaptive = nn.AdaptiveAvgPool2d((32,16))
        self.be_conv1d = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding="same")
        self.out_conv1d = nn.Conv1d(in_channels=32,out_channels=1,kernel_size=1)

        self.sigmoid = nn.Sigmoid()



    def forward(self,x):
        main = self.main_seq_stem(x)
        main = self.main_seq_max_1(main)
        #ver1
        main = self.sa_main(main)
        #ver2
        # main_att = self.sa_main(main)
        # main = main_att*main + main
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

        main = rearrange(main,'b c l w h -> b c l (w h)')
        out = self.max_vit(main)

        out = torch.squeeze(out)
        out = torch.mean(out,dim = -1)
        out_att = self.be_conv1d(out)
        out = (1 + self.sigmoid(out_att)) * out
        out = self.out_conv1d(out)
        out = torch.squeeze(out)
        # out = self.linear(out)
        return out


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
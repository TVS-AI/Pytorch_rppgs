import numpy as np
import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
from self_attention_cv import AxialAttentionBlock,MultiHeadSelfAttention,TransformerEncoder
from self_attention_cv.common import expand_to_batch
from self_attention_cv.pos_embeddings import AbsPosEmb1D, RelPosEmb2D
from nets.models.gcn_utils import KNN_dist, View_selector, LocalGCN, NonLocalMP
from nets.models import PhysNet
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
import math
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import math
import pdb
class Seq_GCN_TT(nn.Module):
    def __init__(self, patches = 16,dim = 768,batch = 32, length = 32):
        super().__init__()
        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim // 4, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(dim // 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim // 2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

    def forward(self,x):
        batch, channel, length ,width, height = x.shape
        # x = rearrange(x,'b c l w h -> (b l) c w h',b = batch, c = channel, l = length)
        x = self.Stem0(x)
        x = self.Stem1(x)
        x = self.Stem2(x)
        return x
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
class ReAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
class LeFF(nn.Module):

    def __init__(self, dim=192, scale=4, depth_kernel=3):
        super().__init__()

        scale_dim = dim * scale
        self.up_proj = nn.Sequential(nn.Linear(dim, scale_dim),
                                     Rearrange('b n c -> b c n'),
                                     nn.BatchNorm1d(scale_dim),
                                     nn.GELU(),
                                     Rearrange('b c (h w) -> b c h w', h=14, w=14)
                                     )

        self.depth_conv = nn.Sequential(
            nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=1, groups=scale_dim, bias=False),
            nn.BatchNorm2d(scale_dim),
            nn.GELU(),
            Rearrange('b c h w -> b (h w) c', h=14, w=14)
            )

        self.down_proj = nn.Sequential(nn.Linear(scale_dim, dim),
                                       Rearrange('b n c -> b c n'),
                                       nn.BatchNorm1d(dim),
                                       nn.GELU(),
                                       Rearrange('b c n -> b n c')
                                       )

    def forward(self, x):
        x = self.up_proj(x)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x
class LCAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        q = q[:, :, -1, :].unsqueeze(2)  # Only Lth element use as query

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=32):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
class channel_Encoding_Block(nn.Module):
    def __init__(self,token=32,dim_head=32):
        super(channel_Encoding_Block, self).__init__()
        #x-axis
        self.emb_x_r = AbsPosEmb1D(tokens=token,dim_head=32)
        self.muulti_x_r = MultiHeadSelfAttention(dim=32)

        self.emb_x_g = AbsPosEmb1D(tokens=token,dim_head=32)
        self.muulti_x_g = MultiHeadSelfAttention(dim=32)

        self.emb_x_b = AbsPosEmb1D(tokens=token,dim_head=32)
        self.muulti_x_b = MultiHeadSelfAttention(dim=32)

        self.norm1_x = nn.LayerNorm(32)
        #y-axis
        self.emb_y_r = AbsPosEmb1D(tokens=token,dim_head=32)
        self.muulti_y_r = MultiHeadSelfAttention(dim=32)

        self.emb_y_g = AbsPosEmb1D(tokens=token,dim_head=32)
        self.muulti_y_g = MultiHeadSelfAttention(dim=32)

        self.emb_y_b = AbsPosEmb1D(tokens=token,dim_head=32)
        self.muulti_y_b = MultiHeadSelfAttention(dim=32)

        self.norm1_y = nn.LayerNorm(32)
        #FNN
        self.norm1 = nn.LayerNorm(32)
        self.norm2 = nn.LayerNorm(32)
        self.norm3 = nn.LayerNorm(32)

        self.drop_out = nn.Dropout2d(0.5)

        self.linear_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*6, 2048),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(2048,1024),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 32)
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        [batch, channel, length, width, height] = x.shape
        ori = x
        tst = torch.permute(x, [0, 1, 4, 2, 3])  # tst1
        x_tmp = []
        for i in range(width):
            inner_tmp = []

            for j in range(3):
                in_ = torch.unsqueeze(tst[:, j, i, :, :], dim=1)
                if j == 0:
                    emb_y = self.emb_y_r(in_)
                    emb_y = torch.squeeze(emb_y)
                    tmp = self.muulti_y_r(emb_y)
                elif j == 1:
                    emb_y = self.emb_y_g(in_)
                    emb_y = torch.squeeze(emb_y)
                    tmp = self.muulti_y_g(emb_y)
                else:
                    emb_y = self.emb_y_b(in_)
                    emb_y = torch.squeeze(emb_y)
                    tmp = self.muulti_y_b(emb_y)


                tmp = emb_y + self.drop_out(tmp)
                tmp = self.norm1_y(tmp)

                tmp = torch.unsqueeze(tmp, dim=1)
                # tmp = self.conv1x1_2(tmp)
                if j == 0:
                    inner_tmp = tmp
                else:
                    inner_tmp = torch.cat([inner_tmp, tmp], dim=1)
            if i == 0:
                x_tmp = torch.unsqueeze(inner_tmp, dim=2)
            else:
                x_tmp = torch.cat([x_tmp, torch.unsqueeze(inner_tmp, dim=2)], dim=2)

        # x_tmp = x_tmp + self.norm1_y(tst)
        # [0, 1, 4, 2, 3]
        x_tmp = torch.permute(x_tmp, [0, 1, 3, 4, 2])  # tst2
        # x_tmp = torch.permute(x_tmp, [0, 1, 3, 2, 4])

        x = torch.sigmoid(x_tmp)*x

        x = torch.permute(x, [0, 1, 3, 4, 2])  # x1
        y = []
        for i in range(width):
            inner_tmp = []

            for j in range(3):
                in_ = torch.unsqueeze(x[:, j, i, :, :], dim=1)

                if j == 0:
                    emb_x = self.emb_x_r(in_)
                    emb_x = torch.squeeze(emb_y)
                    tmp = self.muulti_x_r(emb_y)
                elif j == 1:
                    emb_x = self.emb_x_g(in_)
                    emb_x = torch.squeeze(emb_y)
                    tmp = self.muulti_x_g(emb_y)
                else:
                    emb_x = self.emb_x_b(in_)
                    emb_x = torch.squeeze(emb_y)
                    tmp = self.muulti_x_b(emb_y)

                tmp = emb_x + self.drop_out(tmp)
                tmp = self.norm1_x(tmp)

                tmp = torch.unsqueeze(tmp, dim=1)
                # tmp = self.conv1x1_2(tmp)
                if j == 0:
                    inner_tmp = tmp
                else:
                    inner_tmp = torch.cat([inner_tmp, tmp], dim=1)
            if i == 0:
                y = torch.unsqueeze(inner_tmp, dim=2)
            else:
                y = torch.cat([y, torch.unsqueeze(inner_tmp, dim=2)], dim=2)

        # y = y + self.norm1(x)
        x = torch.permute(x,[0,1,4,2,3])
        y = torch.permute(y, [0, 1, 4, 2, 3])  # x2
        # x_tmp = torch.permute(x_tmp,[0,1,3,2,4])
        # y = torch.relu(y)
        # x_tmp = torch.relu(x_tmp)
        y = torch.sigmoid(y) * x
        # y = ori*torch.relu(x_tmp+y)

        # # y = self.linear_net(y)
        # y = y + self.drop_out(y)
        # att = self.norm2(y)
        # y = self.linear_net(att)
        # y = att + self.drop_out(y)
        # y = self.norm3(y)

        return y
class Seq_GCN_1(nn.Module):
    def __init__(self, roi = 32, divide = 4, length = 32):
        super().__init__()
        self.Encoding_block = channel_Encoding_Block()



        # self.adaptivepool = nn.AdaptiveMaxPool3d([32,1,1])
        # self.adaptivepool_mean = nn.AdaptiveAvgPool2d(1)

        self.adaptivepool = nn.AdaptiveMaxPool3d([32, 1, 1])
        self.adaptivepool_mean = nn.AdaptiveAvgPool2d(1)

        self.linear_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*8*8 , 1024),
            nn.Dropout(0.5),
            nn.ELU(inplace=True),
            # nn.Linear(2048, 1024),
            # nn.Dropout(0.5),
            # nn.ELU(inplace=True),
            nn.Linear(1024, 32)
        )

        self.bpm_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*8*8, 512),
            nn.BatchNorm1d(512),
            # nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            # nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),

        )
        self.sigmoid = nn.Sigmoid()
        self.drop_out = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(32)

        #VIT test
        self.token_dim_w = 3 * (32 **2)
        self.dim_w = 3*32
        self.project_patches_w = nn.Linear(self.token_dim_w,self.dim_w)

        self.transformer_w = TransformerEncoder(3*32, blocks=6, heads=4,
                                              dim_head=4,
                                              dim_linear_block=3*32,
                                              dropout=0.2)

        self.cls_token_w = nn.Parameter(torch.randn(1, 1, 3*32))
        self.pos_emb1D_w = nn.Parameter(torch.randn(32 + 1, 3*32))
        self.emb_dropout_w = nn.Dropout(0.2)

        #VIT test
        self.token_dim_h = 3 * (32 **2)
        self.dim_h = 3*32
        self.project_patches_h = nn.Linear(self.token_dim_h,self.dim_h)

        self.transformer_h = TransformerEncoder(3*32, blocks=6, heads=4,
                                              dim_head=4,
                                              dim_linear_block=3*32,
                                              dropout=0.2)

        self.cls_token_h = nn.Parameter(torch.randn(1, 1, 3*32))
        self.pos_emb1D_h = nn.Parameter(torch.randn(32 + 1, 3*32))
        self.emb_dropout_h = nn.Dropout(0.2)

        #VIT test
        # self.token_dim = 3 * (32 **2)
        # self.dim = 3
        # self.project_patches = nn.Linear(self.token_dim,self.dim)
        #
        # self.transformer = TransformerEncoder(3, blocks=6, heads=4,
        #                                       dim_head=4,
        #                                       dim_linear_block=3,
        #                                       dropout=0.2)
        #
        # self.cls_token = nn.Parameter(torch.randn(1, 1, 3))
        # self.pos_emb1D = nn.Parameter(torch.randn(32 + 1, 3))
        # self.emb_dropout = nn.Dropout(0.2)

        self.dropout = nn.Dropout2d(0.5)

        self.vit_l = ViT(img_dim=32,in_channels=32,patch_dim=1,classification=None)
        self.vit_w = ViT(img_dim=32, in_channels=32, patch_dim=4,dim_linear_block=32, classification=None)
        self.vit_h = ViT(img_dim=32, in_channels=32, patch_dim=4,dim_linear_block=32, classification=None)

        # self.sequential = nn.Sequential(
        #     nn.Linear(32*256,1024),
        #     nn.Linear(1024,256),
        #     nn.Linear
        # )
        self.normlayer = nn.LayerNorm(32)
        self.conv3d = nn.Conv3d(in_channels=3,out_channels=1,kernel_size=(1,5,5))
        self.conv2d = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        self.gelu = nn.GELU()

        self.project_patches_feature = nn.Linear(24576,64)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 64))
        self.pos_emb1D = nn.Parameter(torch.randn(64 + 1, 64))
        self.mlp_head = nn.Linear(64, 10)
        self.transformer = TransformerEncoder(64, blocks=6, heads=4,
                                                  dim_head=64,
                                                  dim_linear_block=1024,
                                                  dropout=0)
        self.conv2d_1 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding='same')
        self.conv2d_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')


    def forward(self,x):
        # x = F.normalize(x,dim=2)
        img = []
        batch, channel, length, h, w = x.shape
        for i in range(batch):
            tmp = rearrange(x[i],'c length w h ->length c w h')
            if i == 0:
                img = torch.unsqueeze(self.conv2d(tmp),dim=0)
            else:
                img = torch.cat([img,torch.unsqueeze(self.conv2d(tmp),dim=0)],dim=0)

        img = torch.squeeze(img)
        batch, length,h,w = img.shape

        l_axis = img
        h_axis = rearrange(img,'batch length w h -> batch h w length')
        w_axis = rearrange(img,'batch length w h -> batch w length h')

        l_feature = self.vit_l(l_axis)
        h_feature = self.vit_h(h_axis)
        w_feature = self.vit_w(w_axis)

        l_feature =rearrange(l_feature,'batch (w h) (dim length)  -> batch dim length (w h)',w=32,h=32,length=32)
        h_feature = rearrange(h_feature,'batch (w l) (dim h) -> batch dim l (w h)',w=8,l=8,h=32)
        w_feature = rearrange(w_feature,'batch (l h) (dim w) -> batch dim l (w h)',l=8,h=8,w=32)

        features = []
        for i in range(length):
            l_tmp =  l_feature[:,:,i,:]
            h_tmp =  h_feature[:,:,i//4,:]
            w_tmp =  w_feature[:,:,i//4,:]
            tmp = torch.cat([l_tmp,h_tmp,w_tmp],dim=2)
            if i == 0:
                features = torch.unsqueeze(tmp,dim=2)
            else:
                features = torch.cat([features,torch.unsqueeze(tmp,dim=2)],dim=2)
        features = rearrange(features,'batch dim length feature -> batch length (dim feature)',length=32)
        features = self.project_patches_feature(features)

        features = torch.cat((expand_to_batch(self.cls_token, desired_size=features.shape[0]), features), dim=1)
        features = features + self.pos_emb1D[:32 + 1, :]
        y = self.transformer(features,None)
        y = y[:,1:,:]
        y = rearrange(y,'batch len (x y)  -> batch len x y',x=8,y=8)
        att = y
        y = self.conv2d_1(y)
        y = torch.relu(y)
        y = self.conv2d_2(y)
        y = y + att





        #x = self.conv2d(x)
        # img_patches_w = rearrange(x,'b c length w h -> b w (length h c)')
        # batch_size, tokens, _ = img_patches_w.shape
        # img_patches_w = self.project_patches_w(img_patches_w)
        # img_patches_w = torch.cat((expand_to_batch(self.cls_token_w, desired_size=batch_size), img_patches_w), dim=1)
        #
        # # add pos. embeddings. + dropout
        # # indexing with the current batch's token length to support variable sequences
        # img_patches_w = img_patches_w + self.pos_emb1D_w[:tokens + 1, :]
        # patch_embeddings_w = self.emb_dropout_w(img_patches_w)
        # w = self.transformer_w(patch_embeddings_w)
        # w = w[:, 1:, :]
        # w = rearrange(w,'b w (length h c) -> b c length w h', w=32, length = 32, c = 3)
        #
        #
        #
        # img_patches_h = rearrange(x,'b c length w h -> b h (length w c)')
        # batch_size, tokens, _ = img_patches_h.shape
        # img_patches_h = self.project_patches_h(img_patches_h)
        # img_patches_h = torch.cat((expand_to_batch(self.cls_token_h, desired_size=batch_size), img_patches_h), dim=1)
        #
        # # add pos. embeddings. + dropout
        # # indexing with the current batch's token length to support variable sequences
        # img_patches_h = img_patches_h + self.pos_emb1D_h[:tokens + 1, :]
        # patch_embeddings_h = self.emb_dropout_h(img_patches_h)
        # h = self.transformer_h(patch_embeddings_h)
        # h = h[:, 1:, :]
        # h = rearrange(h, 'b h (length w c) -> b c length w h', h=32, length=1, c=3)
        #
        # for i in range(32):
        #     h_axis_feature = h[:,:,i,:,:]
        #     w_axis_feature = w[:,:,i,:,:]
        #     img = x[:,:,i,:,:]
        #     img_feature = self.vit(img)
        #     a=[]
        # att = x*self.softmax(w*h)


        # img_patches = rearrange(x,'b c length w h -> b length (h w c)')
        # batch_size, tokens, _ = img_patches.shape
        # img_patches = self.project_patches(img_patches)
        # img_patches = torch.cat((expand_to_batch(self.cls_token, desired_size=batch_size), img_patches), dim=1)
        #
        # # add pos. embeddings. + dropout
        # # indexing with the current batch's token length to support variable sequences
        # img_patches = img_patches + self.pos_emb1D[:tokens + 1, :]
        # patch_embeddings = self.emb_dropout_h(img_patches)
        # i = self.transformer(patch_embeddings)
        # i = i[:, 1:, :]
        # i = rearrange(i, 'b length (h w c) -> b (c length w h)', h=1, length=32, c=3)
        #
        # i = self.l_sequential(i)
        # i = rearrange(i,'b (c length w h) -> b c length w h', h=1, length = 32,c = 3)

        # att = torch.matmul(w,h)
        # att = torch.squeeze(att)
        # att = self.softmax2d(att)
        # att = torch.unsqueeze(att,dim=2)
        # att = i * att
        # att = self.normlayer(x * att)



        #
        # x = self.conv3d(att)
        # x = torch.squeeze(x)
        # x = self.conv2d(x)
        # x = self.gelu(x)


        # y = rearrange(y,'b w (length h) -> b (length w h)',w=32,h=32)
        # att = self.Encoding_block(x)

        # max_y = self.adaptivepool(att)
        # mean_y = self.adaptivepool_mean(att)
        #
        #
        # max_y = torch.squeeze(max_y)
        # mean_y = torch.squeeze(mean_y)
        # y = torch.cat([mean_y, max_y], axis=1)

        bpm = self.bpm_net(y)
        bpm = 200 * self.sigmoid(bpm)

        # y = torch.cat([mean_y,max_y],axis=1)1
        y = self.linear_net(y)
        # y = y + self.drop_out(y)
        # y = self.norm(y)
        # bpm = []
        return y,bpm,att#p.view(-1,self.length-2),p.view(-1,self.length-2)#y.view(-1,self.length-6) # 32 length
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
class Seq_GCN_test(nn.Module):
    def __init__(self):
        super(Seq_GCN, self).__init__()
        self.net = ViViT(image_size=128,patch_size=16,num_classes=32,num_frames=32,dim=192,depth=4,heads=3,pool='cls',in_channels=3,dim_head=64,dropout=0., emb_dropout=0.,scale_dim=4)
        # self.net = ViViTBackbone(32, 128, 128, 8, 16, 16, 32, 512, 3, 10, 3, model=3)
        # self.image_size = 128
        # self.patch_size_l = 32
        # self.patch_size_w = 4
        # self.patch_size_h = 4
        # self.frames = 32
        # self.image_channels = 3
        # self.dim = 192
        #
        # #image feature
        # num_l_axis_patches = (self.image_size//self.patch_size_l) *(self.image_size//self.patch_size_l)
        # patch_dim_l_axis = self.image_channels * self.patch_size_l ** 2
        # self.to_patch_embedding_l_axis = nn.Sequential(
        #     Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size_l,p2=self.patch_size_l),
        #     nn.Linear(patch_dim_l_axis,self.dim)
        # )
        #
        # #ptt feature
        # num_w_axis_patches = (self.image_size//self.patch_size_l)*(self.frames//self.patch_size_w)
        # patch_dim_w_axis = self.image_channels * (self.patch_size_l * self.patch_size_w)
        # self.to_patch_embedding_w_axis = nn.Sequential(
        #     Rearrange('b (t p2) c h (w p1) -> b h (t w) (p1 p2 c)',p1=self.patch_size_l,p2=self.patch_size_w),
        #     nn.Linear(patch_dim_w_axis,self.dim)
        # )
        # #bvp feature
        # num_h_axis_patches = (self.image_size//self.patch_size_l)*(self.frames//self.patch_size_h)
        # patch_dim_h_axis = self.image_channels * (self.patch_size_l * self.patch_size_h)
        # self.to_patch_embedding_w_axis = nn.Sequential(
        #     Rearrange('b (t p2) c (h p1) w -> b w (t h) (p1 p2 c)',p1=self.patch_size_l,p2=self.patch_size_w),
        #     nn.Linear(patch_dim_w_axis,self.dim)
        # )
        #
        #
        #
        # # self.pos_embedding = nn.Parameter(torch.randn(1, self.frames, num_l_axis_patches+1, self.dim))
        #
        # # self.l_token = nn.Parameter(torch.randn(1,1,self.dim))
        # # self.l_transformer = Transformer(self.dim, 4, 3, 64, self.dim*4,)
        # #
        # # self.w_token = nn.Parameter(torch.randn(1, 1, self.dim))
        # # self.w_transformer = Transformer(self.dim, 4, 3, 64, self.dim * 4, )
        # #
        # # self.h_token = nn.Parameter(torch.randn(1, 1, self.dim))
        # # self.h_transformer = Transformer(self.dim, 4, 3, 64, self.dim * 4, )
        # #
        # # ##ViViT
        # #
        # num_patches = (128 // 32) ** 2
        # patch_dim = 3 * 32 ** 2
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=32, p2=32),
        #     nn.Linear(patch_dim, 192),
        # )
        #
        # self.pos_embedding = nn.Parameter(torch.randn(1, 32, num_patches + 1, 192))
        # self.pos_embedding_decoder = nn.Parameter(torch.randn(1, 32, num_patches + 1, 192))
        # self.space_token = nn.Parameter(torch.randn(1, 1, 192))
        # self.space_transformer = Transformer(192, 4, 3, 64, 192 * 4, 0)
        #
        # self.temporal_token = nn.Parameter(torch.randn(1, 1, 192))
        # self.temporal_transformer = Transformer(192, 4, 3, 64, 192 * 4, 0)
        #
        # self.dropout = nn.Dropout(0)
        # self.pool = 'cls'
        #
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(192),
        #     nn.Linear(192, 32)
        # )

        # self.adaptived_3d = nn.AdaptiveAvgPool1d(1)
        # self.conv2d_1x1 = nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1)
        # self.conv1d = nn.Conv1d(in_channels=32,out_channels=32,kernel_size=1)
        #
        # self.Encoding_block = channel_Encoding_Block()

        # self.embed = nn.Embedding(128,self.dim)

    def forward(self,x):
        batch, channel, length, h, w = x.shape
        x = rearrange(x,'b c l h w -> b l c h w')
        x = self.net(x)
        # x = self.net(x).to('cuda:0')
        # x = self.to_patch_embedding(x)
        # b, t, n, _ = x.shape
        #
        # cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
        # x = torch.cat((cls_space_tokens, x), dim=2)
        # x += self.pos_embedding[:, :, :(n + 1)] # pos_embed
        # x = self.dropout(x)
        #
        # x = rearrange(x, 'b t n d -> (b t) n d')
        # x = self.space_transformer(x)
        # x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)
        #
        # cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        # x = torch.cat((cls_temporal_tokens, x), dim=1)
        #
        # x = self.temporal_transformer(x)
        # # x = rearrange(x, 'b l (c w h) -> b l c w h',c = 3, w =8,h=8)
        # # x = self.adaptived_3d(x)
        # # x = self.conv1d(x[:,1:,:])
        # # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        # # x = rearrange(x,'b c n -> b c n 1')
        # a = []
        # x += self.embed(x)


        # x = self.conv2d_1x1(x)
        # x = self.mlp_head(x)

        return x,

        #
        # l_patch = self.patch_size_l(x)
        # w_patch = self.patch_size_w(x)
        # h_patch = self.patch_size_h(x)

        a = []
        y,bpm,att = []
        return y, bpm, att
class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim=192, depth=4, heads=3, pool='cls',
                 in_channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0., scale_dim=4, ):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.deconv = nn.ConvTranspose2d(in_channels=1, out_channels=32, kernel_size=1)

        self.conv2d = nn.AdaptiveAvgPool2d((32,1))
        self.conv1x1 = nn.Conv2d(32,1,[1,1],stride=1,padding=0)
        # self.conv3d = nn.Conv3d(32,1,[1,1],stride=1,padding=0)
        # torch.nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)


        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)
        x = torch.unsqueeze(x,dim=1)
        # x = rearrange(x, 'b (x y) (8 8 3) -> b 1 (x y) (8 8 3)')
        x = self.deconv(x)
        x = self.conv2d(x)
        x = self.conv1x1(x)
        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return x.view(-1, 32)#self.mlp_head(x)
class ViViTBackbone(nn.Module):
    """ Model-3 backbone of ViViT """
    def __init__(self, t, h, w, patch_t, patch_h, patch_w, num_classes, dim, depth, heads, mlp_dim, dim_head=3,
                 channels=3, mode='tubelet', device='cuda', emb_dropout=0., dropout=0., model=3):
        super().__init__()

        assert t % patch_t == 0 and h % patch_h == 0 and w % patch_w == 0, "Video dimensions should be divisible by " \
                                                                           "tubelet size "

        self.T = t
        self.H = h
        self.W = w
        self.channels = channels
        self.t = patch_t
        self.h = patch_h
        self.w = patch_w
        self.mode = mode
        self.device = device

        self.nt = self.T // self.t
        self.nh = self.H // self.h
        self.nw = self.W // self.w

        tubelet_dim = self.t * self.h * self.w * channels

        self.to_tubelet_embedding = nn.Sequential(
            Rearrange('b c (t pt) (h ph) (w pw) -> b t (h w) (pt ph pw c)', pt=self.t, ph=self.h, pw=self.w),
            nn.Linear(tubelet_dim, dim)
        )

        # repeat same spatial position encoding temporally
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.nh * self.nw, dim)).repeat(1, self.nt, 1, 1).to('cuda:0')

        self.dropout = nn.Dropout(emb_dropout)

        if model == 3:
            self.transformer = FSATransformerEncoder(dim, depth, heads, dim_head, mlp_dim,
                                                     self.nt, self.nh, self.nw, dropout)
        elif model == 4:
            assert heads % 2 == 0, "Number of heads should be even"
            self.transformer = FDATransformerEncoder(dim, depth, heads, dim_head, mlp_dim,
                                                     self.nt, self.nh, self.nw, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        """ x is a video: (b, C, T, H, W) """

        tokens = self.to_tubelet_embedding(x)

        tokens += self.pos_embedding
        tokens = self.dropout(tokens)

        x = self.transformer(tokens)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.mlp_head(x)
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
class FSAttention(nn.Module):
    """Factorized Self-Attention"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
class FDAttention(nn.Module):
    """Factorized Dot-product Attention"""

    def __init__(self, dim, nt, nh, nw, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.nt = nt
        self.nh = nh
        self.nw = nw

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, d, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        qs, qt = q.chunk(2, dim=1)
        ks, kt = k.chunk(2, dim=1)
        vs, vt = v.chunk(2, dim=1)

        # Attention over spatial dimension
        qs = qs.view(b, h // 2, self.nt, self.nh * self.nw, -1)
        ks, vs = ks.view(b, h // 2, self.nt, self.nh * self.nw, -1), vs.view(b, h // 2, self.nt, self.nh * self.nw, -1)
        spatial_dots = einsum('b h t i d, b h t j d -> b h t i j', qs, ks) * self.scale
        sp_attn = self.attend(spatial_dots)
        spatial_out = einsum('b h t i j, b h t j d -> b h t i d', sp_attn, vs)

        # Attention over temporal dimension
        qt = qt.view(b, h // 2, self.nh * self.nw, self.nt, -1)
        kt, vt = kt.view(b, h // 2, self.nh * self.nw, self.nt, -1), vt.view(b, h // 2, self.nh * self.nw, self.nt, -1)
        temporal_dots = einsum('b h s i d, b h s j d -> b h s i j', qt, kt) * self.scale
        temporal_attn = self.attend(temporal_dots)
        temporal_out = einsum('b h s i j, b h s j d -> b h s i d', temporal_attn, vt)

        # return self.to_out(out)
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
class FSATransformerEncoder(nn.Module):
    """Factorized Self-Attention Transformer Encoder"""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, nt, nh, nw, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.nt = nt
        self.nh = nh
        self.nw = nw

        for _ in range(depth):
            self.layers.append(nn.ModuleList(
                [PreNorm(dim, FSAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                 PreNorm(dim, FSAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                 ]))

    def forward(self, x):

        b = x.shape[0]
        x = torch.flatten(x, start_dim=0, end_dim=1)  # extract spatial tokens from x

        for sp_attn, temp_attn, ff in self.layers:
            sp_attn_x = sp_attn(x) + x  # Spatial attention

            # Reshape tensors for temporal attention
            sp_attn_x = sp_attn_x.chunk(b, dim=0)
            sp_attn_x = [temp[None] for temp in sp_attn_x]
            sp_attn_x = torch.cat(sp_attn_x, dim=0).transpose(1, 2)
            sp_attn_x = torch.flatten(sp_attn_x, start_dim=0, end_dim=1)

            temp_attn_x = temp_attn(sp_attn_x) + sp_attn_x  # Temporal attention

            x = ff(temp_attn_x) + temp_attn_x  # MLP

            # Again reshape tensor for spatial attention
            x = x.chunk(b, dim=0)
            x = [temp[None] for temp in x]
            x = torch.cat(x, dim=0).transpose(1, 2)
            x = torch.flatten(x, start_dim=0, end_dim=1)

        # Reshape vector to [b, nt*nh*nw, dim]
        x = x.chunk(b, dim=0)
        x = [temp[None] for temp in x]
        x = torch.cat(x, dim=0)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        return x
class FDATransformerEncoder(nn.Module):
    """Factorized Dot-product Attention Transformer Encoder"""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, nt, nh, nw, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.nt = nt
        self.nh = nh
        self.nw = nw

        for _ in range(depth):
            self.layers.append(
                PreNorm(dim, FDAttention(dim, nt, nh, nw, heads=heads, dim_head=dim_head, dropout=dropout)))

    def forward(self, x):
        for attn in self.layers:
            x = attn(x) + x

        return x
class Seq_GCN_SS(nn.Module):
    def __init__(self):
        super(Seq_GCN_SS,self).__init__()
        self.conv1 = ConvBlock(3,64)
        self.conv2 = ConvBlock(64,128)
        self.conv3 = ConvBlock(128,256)
        self.conv4 = ConvBlock(256,512)
        self.up1 = UpBlock(512,256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up4 = UpBlock(64, 1)

    def forward(self,x):
        # x = x.repeat(4,1,25,1)
        batch, n, l, c = x.shape
        x = rearrange(x,'batch n l c -> batch c n l')

        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.up1(y)
        y = self.up2(y)
        y = self.up3(y)
        y = self.up4(y)

        return y
class ConvBlock(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim,out_channels=out_dim,kernel_size=(3,3),stride=(2,2)),
            nn.BatchNorm2d(out_dim),
            nn.Conv2d(in_channels=out_dim,out_channels=out_dim,kernel_size=(3,3),stride=(1,1)),
            nn.BatchNorm2d(out_dim),
            nn.SELU(inplace=True)
        )
        # (25 - 3)/2 +1

    def forward(self,x):
        return self.conv(x)
class UpBlock(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(UpBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_dim,out_channels=out_dim,kernel_size=(1,2),stride=(1,2)),
            nn.Conv2d(in_channels=out_dim,out_channels=out_dim,kernel_size=3,stride=(2,1)),
            nn.BatchNorm2d(out_dim),
            nn.Conv2d(in_channels=out_dim,out_channels=out_dim,kernel_size=3,stride=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)
def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)
'''
Temporal Center-difference based Convolutional layer (3D version)
theta: control the percentage of original convolution and centeral-difference convolution
'''
class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
                    2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal
#physformer
# stem_3DCNN + ST-ViT with local Depthwise Spatio-Temporal MLP
class Physformer(nn.Module):

    def __init__(
            self,
            name: Optional[str] = None,
            pretrained: bool = False,
            patches: int = 16,
            dim: int = 64,
            ff_dim: int = 3072,
            num_heads: int = 4,
            num_layers: int = 12,
            attention_dropout_rate: float = 0.0,
            dropout_rate: float = 0.2,
            representation_size: Optional[int] = None,
            load_repr_layer: bool = False,
            classifier: str = 'token',
            # positional_embedding: str = '1d',
            in_channels: int = 3,
            frame: int = 32,
            theta: float = 0.2,
            image_size: int = 128,
    ):
        super().__init__()

        self.image_size = image_size
        self.frame = frame
        self.dim = dim

        # Image and patch sizes
        t, h, w = (32,128,128)#as_tuple(image_size)  # tube sizes
        ft, fh, fw = (4,4,4)#as_tuple(patches)  # patch sizes, ft = 4 ==> 160/4=40
        gt, gh, gw = t // ft, h // fh, w // fw  # number of patches
        seq_len = gh * gw * gt

        # Patch embedding    [4x16x16]conv
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))

        # Transformer
        self.transformer1 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers // 3, dim=dim, num_heads=num_heads,
                                                         ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        # Transformer
        self.transformer2 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers // 3, dim=dim, num_heads=num_heads,
                                                         ff_dim=ff_dim, dropout=dropout_rate, theta=theta)
        # Transformer
        self.transformer3 = Transformer_ST_TDC_gra_sharp(num_layers=num_layers // 3, dim=dim, num_heads=num_heads,
                                                         ff_dim=ff_dim, dropout=dropout_rate, theta=theta)

        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim // 4, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(dim // 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim // 2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        # self.normLast = nn.LayerNorm(dim, eps=1e-6)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(dim),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(dim, dim // 2, [3, 1, 1], stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(dim // 2),
            nn.ELU(),
        )

        self.ConvBlockLast = nn.Conv1d(dim // 2, 1, 1, stride=1, padding=0)

        # Initialize weights
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)

        self.apply(_init)

    def forward(self, x, gra_sharp=2.0):

        b, c, t, fh, fw = x.shape

        x = self.Stem0(x)
        x = self.Stem1(x)
        x = self.Stem2(x)  # [B, 64, 160, 64, 64]

        x = self.patch_embedding(x)  # [B, 64, 40, 4, 4]
        x = rearrange(x,('b c t w h -> b (t w h) c'), b = b, c = 64, t = 8,w = 4, h = 4)
        #x.flatten(2).transpose(1, 2)  # [B, 40*4*4, 64]

        Trans_features, Score1 = self.transformer1(x, gra_sharp)  # [B, 4*4*40, 64]
        Trans_features2, Score2 = self.transformer2(Trans_features, gra_sharp)  # [B, 4*4*40, 64]
        Trans_features3, Score3 = self.transformer3(Trans_features2, gra_sharp)  # [B, 4*4*40, 64]

        # Trans_features3 = self.normLast(Trans_features3)

        # upsampling heads
        # features_last = Trans_features3.transpose(1, 2).view(b, self.dim, 40, 4, 4) # [B, 64, 40, 4, 4]
        features_last = Trans_features3.transpose(1, 2).view(b, self.dim, t // 4, 4, 4)  # [B, 64, 40, 4, 4]

        features_last = self.upsample(features_last)  # x [B, 64, 7*7, 80]
        features_last = self.upsample2(features_last)  # x [B, 32, 7*7, 160]

        features_last = torch.mean(features_last, 3)  # x [B, 32, 160, 4]
        features_last = torch.mean(features_last, 3)  # x [B, 32, 160]
        rPPG = self.ConvBlockLast(features_last)  # x [B, 1, 160]

        # pdb.set_trace()

        rPPG = rPPG.squeeze(1)

        return rPPG#, Score1, Score2, Score3
'''
Temporal Center-difference based Convolutional layer (3D version)
theta: control the percentage of original convolution and centeral-difference convolution
'''
class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
                    2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal
def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)
def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)
class MultiHeadedSelfAttention_TDC_gra_sharp(nn.Module):
    """Multi-Headed Dot Product Attention with depth-wise Conv3d"""

    def __init__(self, dim, num_heads, dropout, theta):
        super().__init__()

        self.proj_q = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
            # nn.ELU(),
        )
        self.proj_k = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
            # nn.ELU(),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
            # nn.BatchNorm3d(dim),
            # nn.ELU(),
        )

        # self.proj_q = nn.Linear(dim, dim)
        # self.proj_k = nn.Linear(dim, dim)
        # self.proj_v = nn.Linear(dim, dim)

        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None  # for visualization

    def forward(self, x, gra_sharp):  # [B, 4*4*40, 128]
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        [B, P, C] = x.shape
        x = x.transpose(1, 2).view(B, C, P // 16, 4, 4)  # [B, dim, 40, 4, 4]
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q = q.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        k = k.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        v = v.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]

        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / gra_sharp

        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h, scores
class PositionWiseFeedForward_ST(nn.Module):
    """FeedForward Neural Networks for each position"""

    def __init__(self, dim, ff_dim):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Conv3d(dim, ff_dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )

        self.STConv = nn.Sequential(
            nn.Conv3d(ff_dim, ff_dim, 3, stride=1, padding=1, groups=ff_dim, bias=False),
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )

        self.fc2 = nn.Sequential(
            nn.Conv3d(ff_dim, dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(dim),
        )

    def forward(self, x):  # [B, 4*4*40, 128]
        [B, P, C] = x.shape
        # x = x.transpose(1, 2).view(B, C, 40, 4, 4)      # [B, dim, 40, 4, 4]
        x = x.transpose(1, 2).view(B, C, P // 16, 4, 4)  # [B, dim, 40, 4, 4]
        x = self.fc1(x)  # x [B, ff_dim, 40, 4, 4]
        x = self.STConv(x)  # x [B, ff_dim, 40, 4, 4]
        x = self.fc2(x)  # x [B, dim, 40, 4, 4]
        x = x.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]

        return x

        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        # return self.fc2(F.gelu(self.fc1(x)))
class Block_ST_TDC_gra_sharp(nn.Module):
    """Transformer Block"""

    def __init__(self, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.attn = MultiHeadedSelfAttention_TDC_gra_sharp(dim, num_heads, dropout, theta)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward_ST(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gra_sharp):
        Atten, Score = self.attn(self.norm1(x), gra_sharp)
        h = self.drop(self.proj(Atten))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x, Score
class Transformer_ST_TDC_gra_sharp(nn.Module):
    """Transformer with Self-Attentive Blocks"""

    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block_ST_TDC_gra_sharp(dim, num_heads, ff_dim, dropout, theta) for _ in range(num_layers)])

    def forward(self, x, gra_sharp):
        for block in self.blocks:
            x, Score = block(x, gra_sharp)
        return x, Score
class Seq_GCN(nn.Module):
    def __init__(self):
        super(Seq_GCN, self).__init__()
        self.conv = nn.Conv2d(in_channels=3,out_channels=20,kernel_size=3)
        self.vit_1 = ViT(img_dim=(128,32))
    def forward(self,x):
        batch, length, channel, height, width = x.shape
        x = Rearrange(x, 'b l c h w -> (b h) c l w')
        y = self.vit_1(x)


        return y


############## ver 0415
class Seq_GCN(nn.Module):
    def __init__(self):
        super(Seq_GCN, self).__init__()

        self.dim = [3,32,64,128]


        self.main_plane = MainPlan()
        self.bvp_plane = BvpPlan()
        self.ptt_plane = PttPlan()

        self.main_vit = ViT(img_dim=(8,264),in_channels=self.dim[2],patch_dim=(2,132),dim=4*32,blocks=1,classification=False,dropout=0.1)

        self.up_1 = UpBlock(64,32)
        self.up_2 = UpBlock(32, 16)
        self.up_3 = UpBlock(16, 3)
        self.up_4 = UpBlock(3, 1)

        self.batch_norm = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.5)
        self.dropout_2 = nn.Dropout2d(0.5638209250254641)



    def forward(self,x):

        main_plane = self.main_plane(x)
        ptt_plane = self.ptt_plane(x)
        bvp_plane = self.bvp_plane(x)

        out = []
        batch, channel, length, e = main_plane.shape
        for i in range(length):
            out.append(torch.unsqueeze(torch.cat([main_plane[:,:,i,:],ptt_plane,bvp_plane],dim=2),dim=2))

        out = torch.cat(out,dim=2)
        out = self.main_vit(out)
        out = rearrange(out, 'b xy (p c) -> b c xy p',c =self.dim[2],xy=8)

        out = self.up_1(out)
        out = self.up_2(out)
        out = self.up_3(out)
        out = self.up_4(out)
        out = torch.squeeze(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(ConvBlock, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=in_dim,out_channels=out_dim,kernel_size=(3,3),stride=(2,2),padding=(1,1)),
            nn.BatchNorm2d(out_dim),
            nn.Conv2d(in_channels=out_dim,out_channels=out_dim,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
        # (25 - 3)/2 +1

    def forward(self,x):
        return self.seq(x)

class ConvBlock_main(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(ConvBlock_main, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=in_dim,out_channels=out_dim,kernel_size=(3,3),stride=(2,2),padding=(1,1)),
            nn.BatchNorm2d(out_dim),
            nn.Conv2d(in_channels=out_dim,out_channels=out_dim,kernel_size=(3,3),stride=(2,2),padding=(1,1)),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        # (25 - 3)/2 +1

    def forward(self,x):
        return self.seq(x)


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

class MainPlan(nn.Module):
    def __init__(self):
        super(MainPlan, self).__init__()
        self.dim = [3,32,64,128]
        self.involve_main_conv2d = ConvBlock_main(in_dim=self.dim[0], out_dim=self.dim[1])
        # nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(2,2),stride=2)
        self.main_conv2d = nn.Sequential(
            nn.Conv2d(in_channels=self.dim[1], out_channels=self.dim[1], kernel_size=(1, 32), stride=(1, 1), dilation=(1, 32)),
            nn.BatchNorm2d(self.dim[1]),
            nn.Conv2d(in_channels=self.dim[1], out_channels=self.dim[1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(self.dim[1]),
            nn.ReLU(inplace=True)
        )
        # self.main_vit = V(image_size=32,patch_size=4,nu)
        self.main_vit = ViT(img_dim=(32,32),in_channels=self.dim[1],patch_dim=(4,4),blocks=2,classification=False)
        self.main_conv_block = ConvBlock(self.dim[1],self.dim[2])
    def forward(self,x):
        batch, channel, length, height, width = x.shape
        main_plane = rearrange(x, 'b c l h w -> (b l) c h w')  # 128/3/128/128

        main_plane = self.involve_main_conv2d(main_plane)  # 128/3/32/32
        main_plane = rearrange(main_plane, '(b l) c h w -> b c l (h w)', l=length)  # 4/3/32/1024
        main_plane = self.main_conv2d(main_plane)  # 4/3/16/16
        main_plane = self.main_vit(main_plane) # 32/32/16/16
        main_plane = rearrange(main_plane, 'b xy (patch c) -> b c patch xy', c = self.dim[1])

        main_plane = self.main_conv_block(main_plane)  # 4/64/8/8 # b c l (h w)
        return main_plane
class PttPlan(nn.Module):
    def __init__(self):
        super(PttPlan, self).__init__()
        self.dim = [3, 32, 64, 128]
        self.involve_ptt_conv2d = ConvBlock_main(in_dim=self.dim[0], out_dim=self.dim[
            1])  # nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(2, 2), stride=2)
        self.ptt_conv2d = nn.Sequential(
            nn.Conv2d(in_channels=self.dim[1], out_channels=self.dim[1], kernel_size=(1, 8), stride=(1, 1),
                      dilation=(1, 32)),
            nn.BatchNorm2d(self.dim[1]),
            nn.Conv2d(in_channels=self.dim[1], out_channels=self.dim[1], kernel_size=(3, 3), stride=(2, 2),
                      padding=(1, 1)),
            nn.BatchNorm2d(self.dim[1]),
            nn.ReLU(inplace=True)
        )
       # self.ptt_vit = ViT(img_dim=(64, 16), in_channels=self.dim[1], patch_dim=(8, 2), blocks=2,dim_head=64, classification=False)
        self.ptt_conv_block = ConvBlock(self.dim[1], self.dim[2])
        self.dropout_2 = nn.Dropout2d(0.5638209250254641)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((64, 128))
    def forward(self,x):
        batch, channel, length, height, width = x.shape
        ptt_plane = rearrange(x, 'b c l h w -> (b h) c l w')     # 512/3/32/128
        ptt_plane = self.involve_ptt_conv2d(ptt_plane)          # 512/3/8/32
        ptt_plane = rearrange(ptt_plane, '(b h) c l w -> b c h (l w)',h=height)  # 4/3/128/256
        ptt_plane = self.ptt_conv2d(ptt_plane) # 4/32/128/32
        # ptt_plane = self.ptt_conv2d_2(ptt_plane) # 4/32/64/16
        # ptt_plane = self.ptt_vit(ptt_plane)
        # ptt_plane = rearrange(ptt_plane, 'b xy (patch c) -> b c patch xy', c=self.dim[1])
        ptt_plane = self.ptt_conv_block(ptt_plane) #4/64/32/8 # b c h (l w)

        ptt_plane = rearrange(ptt_plane, 'b c h e -> b c (h e)')
        ptt_plane = self.dropout_2(ptt_plane)
        ptt_plane = self.adaptive_pool(ptt_plane)
        return ptt_plane
class BvpPlan(nn.Module):
    def __init__(self):
        super(BvpPlan, self).__init__()
        self.dim = [3,32,64,128]
        self.involve_bvp_conv2d = ConvBlock_main(in_dim=self.dim[0], out_dim=self.dim[
            1])  # nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(2, 2), stride=2)
        self.bvp_conv2d = nn.Sequential(
            nn.Conv2d(in_channels=self.dim[1], out_channels=self.dim[1], kernel_size=(1, 8), stride=(1, 1),
                      dilation=(1, 32)),
            nn.BatchNorm2d(self.dim[1]),
            nn.Conv2d(in_channels=self.dim[1], out_channels=self.dim[1], kernel_size=(3, 3), stride=(2, 2),
                      padding=(1, 1)),
            nn.BatchNorm2d(self.dim[1]),
            nn.ReLU(inplace=True)
        )
        # self.bvp_vit = ViT(img_dim=(64, 16), in_channels=self.dim[1], patch_dim=(8, 2), blocks=2,dim_head=64, classification=False)
        self.bvp_conv_block = ConvBlock(self.dim[1], self.dim[2])
        self.dropout = nn.Dropout2d(0.5638209250254641)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((64, 128))
    def forward(self,x):
        batch, channel, length, height, width = x.shape
        bvp_plane = rearrange(x, 'b c l h w -> (b w) c l h')  # 512/3/32/128

        bvp_plane = self.involve_bvp_conv2d(bvp_plane)  # 512/3/8/32
        bvp_plane = rearrange(bvp_plane, '(b w) c l h -> b c w (l h)', w=width)  # 4/3/128/256
        bvp_plane = self.bvp_conv2d(bvp_plane)  # 4/32/64/16
        # bvp_plane = self.bvp_vit(bvp_plane)
        # bvp_plane = rearrange(bvp_plane, 'b xy (patch c) -> b c patch xy', c=self.dim[1])
        bvp_plane = self.bvp_conv_block(bvp_plane)  # 4/64/32/8 # b c w (l h)
        bvp_plane = rearrange(bvp_plane, 'b c w e -> b c (w e)')
        bvp_plane = self.dropout(bvp_plane)
        bvp_plane = self.adaptive_pool(bvp_plane)
        return bvp_plane


class TransformerEncoder(nn.Module):
    def __init__(self, dim, blocks=6, heads=8, dim_head=None, dim_linear_block=1024, dropout=0, prenorm=False):
        super().__init__()
        self.block_list = [TransformerBlock(dim, heads, dim_head,
                                            dim_linear_block, dropout, prenorm=prenorm) for _ in range(blocks)]
        self.layers = nn.ModuleList(self.block_list)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
class TransformerBlock(nn.Module):
    """
    Vanilla transformer block from the original paper "Attention is all you need"
    Detailed analysis: https://theaisummer.com/transformer/
    """

    def __init__(self, dim, heads=8, dim_head=None,
                 dim_linear_block=1024, dropout=0.1, activation=nn.GELU,
                 mhsa=None, prenorm=False):
        """
        Args:
            dim: token's vector length
            heads: number of heads
            dim_head: if none dim/heads is used
            dim_linear_block: the inner projection dim
            dropout: probability of droppping values
            mhsa: if provided you can change the vanilla self-attention block
            prenorm: if the layer norm will be applied before the mhsa or after
        """
        super().__init__()
        self.mhsa = mhsa if mhsa is not None else MultiHeadSelfAttention(dim=dim, heads=heads, dim_head=dim_head)
        self.prenorm = prenorm
        self.drop = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)

        self.linear = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            activation(),  # nn.ReLU or nn.GELU
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        if self.prenorm:
            y = self.drop(self.mhsa(self.norm_1(x), mask)) + x
            out = self.linear(self.norm_2(y)) + y
        else:
            y = self.norm_1(self.drop(self.mhsa(x, mask)) + x)
            out = self.norm_2(self.linear(y) + y)
        return out
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None):
        """
        Implementation of multi-head attention layer of the original transformer model.
        einsum and einops.rearrange is used whenever possible
        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head<dim.
            However, it may not necessary be (dim/heads)
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        # self.to_qv = nn.Linear(dim, _dim * 2, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5

    def forward(self, x, mask=None):
        assert x.dim() == 3
        qvk = self.to_qvk(x)  # [batch, tokens, dim*3*heads ]

        # decomposition to q,v,k and cast to tuple
        # the resulted shape before casting to tuple will be: [3, batch, heads, tokens, dim_head]
        q,k, v = tuple(rearrange(qvk, 'b t (d k h ) -> k b h t d ', k=3, h=self.heads))
        # k = rearrange(mask, 'b t (d h ) -> b h t d ',h=self.heads)
        out = compute_mhsa(q, k, v, mask=None, scale_factor=self.scale_factor)

        # re-compose: merge heads with dim_head
        out = rearrange(out, "b h t d -> b t (h d)")
        # Apply final linear transformation layer
        return self.W_0(out)
def compute_mhsa(q, k, v, scale_factor=1, mask=None):
    # resulted shape will be: [batch, heads, tokens, tokens]
    scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) * scale_factor

    if mask is not None:
        assert mask.shape == scaled_dot_prod.shape[2:]
        scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

    attention = torch.softmax(scaled_dot_prod, dim=-1)
    # calc result per head
    return torch.einsum('... i j , ... j d -> ... i d', attention, v)
class ViT(nn.Module):
    def __init__(self, *,
                 img_dim=(256,256),
                 in_channels=3,
                 patch_dim=(16,16),
                 num_classes=10,
                 dim=None,
                 blocks=6,
                 heads=4,
                 dim_linear_block=1024,
                 dim_head=None,
                 dropout=0, transformer=None, classification=True):
        """
        Minimal re-implementation of ViT
        Args:
            img_dim: the spatial image size
            in_channels: number of img channels
            patch_dim: desired patch dim
            num_classes: classification task classes
            dim: the linear layer's dim to project the patches for MHSA
            blocks: number of transformer blocks
            heads: number of heads
            dim_linear_block: inner dim of the transformer linear block
            dim_head: dim head in case you want to define it. defaults to dim/heads
            dropout: for pos emb and transformer
            transformer: in case you want to provide another transformer implementation
            classification: creates an extra CLS token that we will index in the final classification layer
        """
        super().__init__()
        assert img_dim[0] % patch_dim[0] == 0, f'patch size h {patch_dim[0]} not divisible by img dim {img_dim[0]}'
        assert img_dim[1] % patch_dim[1] == 0, f'patch size w {patch_dim[1]} not divisible by img dim {img_dim[1]}'
        # self.p = patch_dim
        self.p_h = patch_dim[0]
        self.p_w = patch_dim[1]
        self.classification = classification
        # tokens = number of patches
        # tokens = (img_dim // patch_dim) ** 2
        tokens = (img_dim[0]//patch_dim[0]) *(img_dim[1]//patch_dim[1])
        #self.token_dim = in_channels * ( patch_dim ** 2)
        self.token_dim = in_channels * (patch_dim[0] * patch_dim[1])
        # self.token_dim = self.p_h*self.p_w*in_channels
        if dim is None:
            self.dim = self.token_dim
        else:
            self.dim = dim
        self.dim_head = (int(self.dim / heads)) if dim_head is None else dim_head

        # Projection and pos embeddings
        self.project_patches = nn.Linear(self.token_dim, self.dim)

        self.emb_dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.pos_emb1D = nn.Parameter(torch.randn(tokens + 1, self.dim))
        self.mlp_head = nn.Linear(self.dim, num_classes)

        if transformer is None:
            self.transformer = TransformerEncoder(self.dim, blocks=blocks, heads=heads,
                                                  dim_head=self.dim_head,
                                                  dim_linear_block=dim_linear_block,
                                                  dropout=dropout)
        else:
            self.transformer = transformer

    def forward(self, img, mask=None):
        # Create patches
        # from [batch, channels, h, w] to [batch, tokens , N], N=p*p*c , tokens = h/p *w/p
        img_patches = rearrange(img,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.p_h, patch_y=self.p_w)

        batch_size, tokens, _ = img_patches.shape

        # project patches with linear layer + add pos emb
        img_patches = self.project_patches(img_patches)

        img_patches = torch.cat((expand_to_batch(self.cls_token, desired_size=batch_size), img_patches), dim=1)

        # add pos. embeddings. + dropout
        # indexing with the current batch's token length to support variable sequences
        img_patches = img_patches + self.pos_emb1D[:tokens + 1, :]
        patch_embeddings = self.emb_dropout(img_patches)

        # feed patch_embeddings and output of transformer. shape: [batch, tokens, dim]
        y = self.transformer(patch_embeddings, mask)

        # we index only the cls token for classification. nlp tricks :P
        return self.mlp_head(y[:, 0, :]) if self.classification else y[:, 1:, :]



### 0417


def conv_3x3_bn(inp,oup,image_size,downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp,oup,3,stride,1,bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )

class PreNorm(nn.Module):
    def __init__(self,dim,fn,norm):
        super(PreNorm, self).__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self,x,**kwargs):
        return self.fn(self.norm(x), **kwargs)

class SE(nn.Module):
    def __init__(self,inp,oup,expansion=0.25):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup,int(inp*expansion),bias=False),
            nn.GELU(),
            nn.Linear(int(inp*expansion),oup, bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,_,_ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x*y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)


class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih * self.iw, w=self.ih * self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x

class UpBlock(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(UpBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_dim,out_channels=out_dim,kernel_size=(1,2),stride=(1,2)),
            nn.Conv2d(in_channels=out_dim,out_channels=out_dim,kernel_size=3,stride=(2,1),padding=1),
            nn.BatchNorm2d(out_dim),
            nn.Conv2d(in_channels=out_dim,out_channels=out_dim,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_dim),
            nn.SELU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)

class Seq_GCN(nn.Module):
    def __init__(self):
        super(Seq_GCN, self).__init__()
        il,ih,iw = (32,32,32)
        block = { 'C' :MBConv, 'T':Transformer}
        num_blocks = [2, 2, 2, 2, 2, 2, 2]  # L
        channels = [3, 3, 32, 64, 128, 256, 768]  # D
        block_types = ['C', 'C','C', 'T', 'T']
                      # 0    1   2    3    4    5    6
        in_channels = 3
        self.max_pool = nn.MaxPool2d(kernel_size=4)
        #img
        self.s0 = self._make_layer(
            conv_3x3_bn,in_channels,channels[0],num_blocks[0],(ih//2,iw//2)
        )
        #img
        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4)
        )
        # stmap
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], ((ih//4 * iw // 4)//2,il//2)
        )
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], ((ih//4 * iw // 4)//4,il//4)
        )

        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], ((ih//4 * iw // 4)//8,il//8)
        )
        #stmap
        self.s5 = self._make_layer(
            block[block_types[4]], channels[4], channels[5], num_blocks[5],((ih//4 * iw // 4)//16,il//16)
        )

        self.s8 = UpBlock(256,128)
        self.s9 = UpBlock(128, 64)
        self.s10 = UpBlock(64, 32)
        self.s11 = UpBlock(32, 1)
        # self.drop = nn.Dropout(0.5)
        # self.conv = nn.Conv2d(in_channels=channels[0],out_channels=channels[0],kernel_size=(1,32))
        self.mlp = nn.Sequential(
            nn.Linear(32,128),
            nn.Linear(128,32)
        )

    def forward(self,x):
        batch, channel, length, height, width = x.shape
        x = rearrange(x, 'b c l h w -> (b l) c h w')
        x = self.max_pool(x)
        x = self.s0(x)
        x = self.s1(x)
        x = rearrange(x, '(b l) c h w -> b c (h w) l', l=length)
        # x = self.drop(x)
        x = self.mlp(x)
        # x = self.conv(x)
        x = self.s2(x)
        x = self.s3(x)
        # x = self.drop(x)
        x = self.s4(x)
        x = self.s5(x)

        x = self.s8(x)
        x = self.s9(x)
        x = self.s10(x)
        x = self.s11(x)


        return torch.squeeze(x)

    def _make_layer(self,block,inp,oup,depth,image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp,oup,image_size,downsample=True))
            else:
                layers.append(block(oup,oup,image_size))
        return nn.Sequential(*layers)

from dataset.dataset_loader import dataset_loader
from torch.utils.data import DataLoader, random_split
import wandb
import os
from self_attention_cv.common import expand_to_batch
import torch
from torch import nn
from einops import rearrange,repeat
import numpy as np
from loss import loss_fn
from optim import optimizer
from torch.autograd import Variable

hyperparameter_defaults = dict(
    dropout = 0.5,
    batch_size = 100,
    learning_rate = 0.001,
    epochs = 2,
    )

wandb.init(project="SeqNet",entity="daeyeolkim",config=hyperparameter_defaults)
config = wandb.config



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
        self.dropout = nn.Dropout2d(config.dropout)
        self.dropout_2 = nn.Dropout2d(0.2)



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
        self.dropout_2 = nn.Dropout2d(config.dropout)
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
        self.dropout = nn.Dropout2d(config.dropout)
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




def main():
    train_dataset = dataset_loader(model_name="PhysNet",option="train")
    test_dataset = dataset_loader(model_name="PhysNet", option="test")
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=config.batch_size,
                             shuffle=True)
    model = Seq_GCN()
    model.cuda()
    wandb.watch(model)
    criterion = loss_fn("neg_pearson")
    optim = optimizer(model.parameters(),config.learning_rate,config.optimizer)


    iter = 0
    for epoch in range(config.epochs):
        for i, (images, labels) in enumerate(train_loader):

            images = Variable(images)
            labels = Variable(labels)

            # Clear gradients w.r.t. parameters
            optim.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optim.step()

            iter += 1

            if iter % 100 == 0:
                # Calculate Accuracy
                correct = 0.0
                correct_arr = [0.0] * 10
                total = 0.0
                total_arr = [0.0] * 10

                # # Iterate through test dataset
                for images, labels in test_loader:
                    images = Variable(images)

                    # Forward pass only to get logits/output
                    outputs = model(images)

                    loss = criterion(outputs, labels)

                    # Get predictions from the maximum value
                    # _, predicted = torch.max(outputs.data, 1)

                    # Total number of labels
                    total += 1
                    correct += loss.item()
                    # print(str(labels.size(0)))
                    # print(str(total))
                #
                # accuracy = correct / total
                loss = correct/total
                # metrics{'total':total}
                # metrics = {'accuracy': accuracy, 'loss': loss}
                metrics = {'loss': loss}
                # for label in range(10):
                #     metrics['Accuracy ' + label_names[label]] = correct_arr[label] / total_arr[label]

                wandb.log(metrics)

                # Print Loss
                # print('Iteration: {0} Loss: {1:.2f} Accuracy: {2:.2f}'.format(iter, loss, accuracy))
                print('Iteration: {0} Loss: {1:.2f}'.format(iter, loss))
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))

if __name__ == '__main__':
   main()
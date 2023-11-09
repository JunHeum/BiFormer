import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_

from cupy_module import bilateralcorrelation

from .twins_encoder import twins_svt_small, twins_svt_large
from .BilateralAttention import BCALayer

class DeconvSequential(nn.Sequential):
    def forward(self, input, output_size):
        for module in self:
            if isinstance(module, nn.ConvTranspose2d):
                input = module(input, output_size)
            else:
                input = module(input)
        return input

class resblock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(resblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,padding=1)
        self.leakyRELU = nn.LeakyReLU(0.1)
    def forward(self,x):
        shortcut = x
        x = self.leakyRELU(self.conv1(x))
        x = self.conv2(x)
        x += shortcut
        return self.leakyRELU(x)


class BiFormer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, depths=4, sblock_depths=2,
                 window_size=8, B_window_size=0, mlp_ratio=2., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, small=False,
                 use_checkpoint=False, pretrained=False,
                 **kwargs):
        super(BiFormer, self).__init__()
        self.window_size = window_size
        self.B_window_size = B_window_size if B_window_size!= 0 else window_size - 1
        self.corr_range = self.B_window_size // 2
        self.depths = depths # the number of bilateral attention modules
        self.s_depths = sblock_depths # the number of self attention layer for each bilateral attention module
        
        assert sblock_depths % 2 == 0
        
        ###################################################################################################
        ################################### 1, twins feature extraction ###################################
        if small:
            self.encoder = twins_svt_small(pretrained)    
        else:
            self.encoder = twins_svt_large(pretrained)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.pos_drop = nn.Dropout(p=drop_rate)
            
        # build normalization layers
        self.norm_layer = norm_layer(self.embed_dim)
        
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths * sblock_depths)]  # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2 * sblock_depths)] if depths == 1 else dpr


        #####################################################################################################
        ################################### 3, motion feature processing ####################################
        self.decoder_t = nn.ModuleList()
        
        for i_layer in range(depths):
            self.decoder_t.append(BCALayer(dim_in=256, dim_out=None,
                                        depth=sblock_depths,
                                        num_heads=num_heads,
                                        window_size=window_size,
                                        B_window_size=B_window_size,
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=dpr[i_layer*sblock_depths:(i_layer+1)*sblock_depths],
                                        norm_layer=norm_layer,
                                        downsample=None,
                                        upsample=None,
                                        with_anchor=False if i_layer==0 else True,
                                        use_checkpoint=use_checkpoint,
                                        pretrained_window_size=0))
                    
        
        #####################################################################################################
        ################################### 4, motion and mask estimation ###################################
        
        self.corr = bilateralcorrelation.apply
        
        # build convolution layers for estimating bilateral motions and upsampling masks. 
        self.conv_last = nn.Sequential(
            nn.Conv2d(256+self.B_window_size**2, 64, 3, 1, 1),
            resblock(64,64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        self.predict_flow = DeconvSequential(nn.ConvTranspose2d(64,64, 3, 2, 1),
                                             nn.Conv2d(64, 2, 3, 1, 1))
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal(m.weight.data, mode='fan_in')
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
        yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)

        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.to(x.device)

        vgrid = torch.autograd.Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)

        return output * mask
    
    def upsample_flow(self, flow):
        flow = F.interpolate(flow, scale_factor=4.0, mode='bilinear') * 4.0
        return flow
    
    def forward(self, x1, x2):
        N = x1.size(0)
        
        x = self.encoder(torch.cat([x1,x2], dim=0))
        x_size = H,W = (x.size(-2),x.size(-1))
        x1, x2 = x.split([N,N], dim=0)

        bi_corr = self.corr(F.normalize(x1,dim=1),F.normalize(x2,dim=1), self.corr_range)
        
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x2 = rearrange(x2, 'b c h w -> b (h w) c')
        
        for idx in range(self.depths):
            xt = None if idx == 0 else xt
            
            xt,H,W, _,_,_ = self.decoder_t[idx](x1,x2,xt,x_size)
            
        xt = self.norm_layer(xt)
        xt = xt.transpose(1, 2).view(N, -1, H,W) # N C H W
        xt = self.conv_last(torch.cat([xt,bi_corr], dim=1))
                
        flow = self.predict_flow(xt,(H*2,W*2))
        
        flow_fw = self.upsample_flow(flow)
        return flow_fw
        

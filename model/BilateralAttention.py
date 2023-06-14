
from builtins import eval
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath
from einops import rearrange
from .SwinTransformer import SwinTransformerBlockV2

from cupy_module import bi_attn, bi_attn_anchor, apply_attn, apply_attn_inv

def position_encoding(window_size, num_heads, version='v2'):
    assert version == 'v1' or version == 'v2'
    
    wh, ww = window_size

    if version == 'v1':
        return nn.Parameter(torch.zeros(num_heads,(wh*ww)))
    
    elif version =='v2':
        coords_h = torch.arange(wh, dtype=torch.float32) * 2
        coords_w = torch.arange(ww, dtype=torch.float32) * 2
        coords_table = torch.stack(torch.meshgrid([coords_h,coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)
        
        coords_table[:,:,:,0] = coords_table[:,:,:,0] / (wh-1) - 1
        coords_table[:,:,:,1] = coords_table[:,:,:,1] / (ww-1) - 1
        
        coords_table *= 8
        coords_table = torch.sign(coords_table) * torch.log2(torch.abs(coords_table) + 1.0) / np.log2(8)
        return coords_table
    else:
        raise ValueError("No [%s] version is available in position encoding")


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MBCA(nn.Module):
    r""" Multi-head bilateral cross attention without anchor(MBCA-A).
    """
    def __init__(self, window_size,num_heads):
        super().__init__()
        self.window_size = window_size if isinstance(window_size, tuple) else (window_size, window_size)
        assert self.window_size[0] == self.window_size[1], 'Height and Width of window_size must be same in cuda version of MBCA(MBCA_cu)'
        assert self.window_size[0] % 2 == 1, 'window_size must be odd number'
        
        self.num_heads = num_heads
        self.logit_scale =  nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True) # from Swin_v2
        
        self.attn = bi_attn.apply
        self.apply_attn = apply_attn.apply
        
        coords_table = position_encoding(self.window_size, num_heads, version='v2')
        self.register_buffer("coords_table", coords_table)
        
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))
        
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q,k,v):
        r"""
        Args:
            q: (b*nh,hc,h,w)
            k: (b*nh,hc,h,w)
            v: (b*nh,hc,h,w)
        """
        nh = self.num_heads
        wh, ww = self.window_size
        md = wh // 2
        
        _, hc, h, w = q.shape
        
        q,k = map(lambda t: F.normalize(t, dim=1), (q,k))
                
        bi_attn = self.attn(q, k, md)
        bi_attn = bi_attn.view(-1,nh,wh*ww,h,w) # B, nh, wh*ww, h, w
        
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(q.new_tensor(1. / 0.01))).exp()
        bi_attn = bi_attn * logit_scale.view(1,nh,1,1,1)

        relative_position_bias = self.cpb_mlp(self.coords_table).view(wh*ww, nh).permute(1,0).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)

        bi_attn = bi_attn + relative_position_bias.view(1,nh,wh*ww,1,1) # B, nh, wh*ww, h, w

        bi_attn = self.softmax(bi_attn).view(-1,wh*ww,h,w) # B, nh, wh*ww, h, w (softmax) --> B*nh, wh*ww, h, w
        x = self.apply_attn(bi_attn, v)  # B*nh, hc, h, w
        
        ## merge heads
        x = x.view(-1,nh*hc,h,w) # B, c, h, w
        return x


class MBCA_Anchor(nn.Module):
    r""" Multi-head bilateral cross attention with anchor(MBCA+A).
    """
    def __init__(self, window_size,num_heads):
        super().__init__()
        self.window_size = window_size if isinstance(window_size, tuple) else (window_size, window_size)
        assert self.window_size[0] == self.window_size[1], 'Height and Width of window_size must be same in cuda version of MBCA(MBCA_cu)'
        assert self.window_size[0] % 2 == 1, 'window_size must be odd number'
        
        self.num_heads = num_heads
        self.logit_scale =  nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True) # from Swin_v2
        
        self.attn = bi_attn_anchor.apply
        self.apply_attn = apply_attn.apply
        self.apply_attn_inv = apply_attn_inv.apply
        
        coords_table = position_encoding(self.window_size, num_heads, version='v2')
        self.register_buffer("coords_table", coords_table)
        
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))
        
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q,k0,k1,v0,v1):
        r"""
        Args:
            q: (b*nh,hc,h,w)
            k: (b*nh,hc,h,w)
            v: (b*nh,hc,h,w)
        """
        nh = self.num_heads
        wh, ww = self.window_size
        md = wh // 2
        
        _, hc, h, w = q.shape
        
        q,k0,k1 = map(lambda t: F.normalize(t, dim=1), (q,k0,k1))
        
        Sym_attn = self.attn(q,k0,k1,md) # B*nh, wh*ww, h, w
        
        Sym_attn = Sym_attn.view(-1,nh,wh*ww,h,w) # B, nh, wh*ww, h, w
        
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(q.new_tensor(1. / 0.01))).exp()
        Sym_attn = Sym_attn * logit_scale.view(1,nh,1,1,1)

        relative_position_bias = self.cpb_mlp(self.coords_table).view(wh*ww, nh).permute(1,0).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)

        Sym_attn = Sym_attn + relative_position_bias.view(1,nh,wh*ww,1,1) # B, nh, wh*ww, h, w

        Sym_attn = self.softmax(Sym_attn).view(-1,wh*ww,h,w) # B, nh, wh*ww, h, w (softmax) --> B*nh, wh*ww, h, w
        
        x0 = self.apply_attn_inv(Sym_attn, v0) # B*nh, hc, h, w
        x1 = self.apply_attn(Sym_attn, v1) # B*nh, hc, h, w
        
        ## merge heads
        x0, x1 = map(lambda t: t.view(-1,nh*hc,h,w), (x0,x1))
        return x0, x1
    


class BCAblock(nn.Module):
    """ 
        Bilateral cross attention without anchor(BCA-A) block    
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, mlp_ratio=4.,
                       drop_path=0., drop=0., proj_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        
        super().__init__()
        assert window_size % 2 == 1

        self.dim = dim
        
        self.window_size =  window_size
        self.num_heads = num_heads
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)

        self.biattn = MBCA(self.window_size,num_heads)
        self.biattn_inv = MBCA(self.window_size,num_heads) 
        self.proj = nn.Linear(dim*2, dim)

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_last")

        self.drop_path = DropPath(drop_path) if drop_path >0. else nn.Identity()

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self,x0, x1, xt, x_size):
        '''
        Args:
            x0: input features at time 0 with shape of (B,h*w,C)
            x1: input features at time 1 with shape of (B,h*w,C)
            xt: intermediate features at time t with shape of (B,h*w,C)
        '''
                
        B_,N,C = x0.shape
        h, w = x_size
        
        shortcut = x0.new_zeros(B_,N,C, requires_grad=True) if xt is None else xt 
        
        qkv0 = self.qkv(x0).reshape(B_,h,w,3,self.num_heads, C//self.num_heads)
        qkv1 = self.qkv(x1).reshape(B_,h,w,3,self.num_heads, C//self.num_heads)
        q0,k0,v0 = rearrange(qkv0,'b h w qkv nh hc -> qkv (b nh) hc h w')
        q1,k1,v1 = rearrange(qkv1,'b h w qkv nh hc -> qkv (b nh) hc h w')
        q0,k0,v0, q1,k1,v1 = map(lambda t: t.contiguous(), (q0,k0,v0, q1,k1,v1))
        
        xb = self.biattn(q0,k1,v1) # B, C, h, w
        xf = self.biattn_inv(q1,k0,v0) # B, C, h, w
        
        x = rearrange(torch.cat((xb,xf), dim=1), 'b c h w -> b (h w) c')  # B, h*w, 2C
        x = self.proj(x) # B, h*w, C
        x = self.norm1(x)

        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x # B, h*w, C


class BCAblock_Anchor(nn.Module):
    '''
        Bilateral cross attention with anchor(BCA+A) block
    '''
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, mlp_ratio=4.,
                       drop_path=0., drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_ffn=False):
        
        super().__init__()
        assert window_size % 2 == 1

        self.no_ffn = no_ffn

        self.dim = dim
        
        self.window_size = window_size
        self.num_heads = num_heads
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        # self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, stride=1)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.biattn = MBCA_Anchor(self.window_size,num_heads)
        self.proj = nn.Linear(dim*2, dim)

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_last")

        self.drop_path = DropPath(drop_path) if drop_path >0. else nn.Identity()

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self,x0, x1, xt, x_size):
        '''
        Args:
            x0: input features at time 0 with shape of (B,h*w,C)
            x1: input features at time 1 with shape of (B,h*w,C)
            xt: intermediate features at time t with shape of (B,h*w,C)
        '''
                
        B_,N,C = x0.shape
        h, w = x_size
        
        shortcut =  xt 
        
        qt = self.q(xt).reshape(B_,h,w,self.num_heads, C//self.num_heads)
        kv0 = self.kv(x0).reshape(B_,h,w,2,self.num_heads, C//self.num_heads)
        kv1 = self.kv(x1).reshape(B_,h,w,2,self.num_heads, C//self.num_heads)
        
        qt = rearrange(qt, 'b h w nh hc -> (b nh) hc h w')
        k0,v0 = rearrange(kv0,'b h w kv nh hc -> kv (b nh) hc h w')
        k1,v1 = rearrange(kv1,'b h w kv nh hc -> kv (b nh) hc h w')
        qt,k0,v0,k1,v1 = map(lambda t: t.contiguous(), (qt,k0,v0,k1,v1))

        xb, xf = self.biattn(qt,k0,k1,v0,v1) # B, C, h, w
        
        x = rearrange(torch.cat((xb,xf), dim=1), 'b c h w -> b (h w) c')  # B, h*w, 2C
        x = self.proj(x) # B, h*w, C
        x = self.norm1(x)

        x = shortcut + self.drop_path(x)

        if not self.no_ffn:
            x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x # B, h*w, C


class BCALayer(nn.Module):
    def __init__(self, dim_in, dim_out, depth, num_heads, window_size, B_window_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, upsample=None, with_anchor=False,
                 use_checkpoint=False, pretrained_window_size=0):

        super().__init__()
        self.dim = dim_in
        self.dim_out = dim_out
        self.depth = depth
        self.use_checkpoint = use_checkpoint       
        
        bi_window_size = window_size - 1 if B_window_size == 0 else B_window_size
        
        SwinTransformerBlock = SwinTransformerBlockV2

        
        # build BCA-A or BCA+A block
        Bblock = 'BCAblock_Anchor' if with_anchor else 'BCAblock'
        self.Bblock= eval(Bblock)(dim=self.dim, window_size=bi_window_size, num_heads=num_heads,
                                    qkv_bias=qkv_bias, mlp_ratio=mlp_ratio, drop=drop,
                                    drop_path = drop_path[0], norm_layer=norm_layer)
        
        # build swin blocks
        self.Sblocks = nn.ModuleList()
        for i in range(0, depth):
            self.Sblocks.append(SwinTransformerBlock(dim=self.dim,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size))
        
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim_in=dim_in, dim_out=dim_out, norm_layer=norm_layer)
        else:
            self.downsample = None

        # patch split layer
        if upsample is not None:
            self.upsample = upsample(dim_in=dim_in, dim_out=dim_out, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x0, x1, xt, x_size):
        H, W = x_size
        x = xt
        x = self.Bblock(x0,x1,x,x_size)
        for blk in self.Sblocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        
        if self.downsample is not None:
            x_down = self.downsample(x, x_size)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        elif self.upsample is not None:
            x_up = self.upsample(x, x_size)
            Wh, Ww = H*2, W*2
            return x, H, W, x_up, Wh, Ww
        else:
            return x, H, W, x, H, W
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from cupy_module import bilateralcorrelation_nn as bicorr_nn


def L2normalize(x, d=1):
    eps = 1e-6
    norm = x ** 2
    norm = norm.sum(dim=d, keepdim=True) + eps
    norm = norm ** (0.5)
    return (x/norm)


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride=stride,
                  padding=((kernel_size - 1) * dilation) //2 if padding==1 else padding, dilation = dilation, bias= True),
        nn.LeakyReLU(0.1))


class resblock(nn.Module):
    def __init__(self, in_channels,out_channels,stride):
        super(resblock, self).__init__()
        self.conv = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=3, stride=stride),
            conv(out_channels, out_channels, kernel_size=3, stride=1))
        self.leakyRELU = nn.LeakyReLU(0.1)
    def forward(self,x):
        y = self.conv(x)
        return self.leakyRELU(x+y)


class ContextNetwork(nn.Module):
    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, dilation=1),
            conv(128, 128, 3, 1, dilation=2),
            conv(128, 128, 3, 1, dilation=4),
            conv(128, 96, 3, 1, dilation=8),
            conv(96, 64, 3, 1, dilation=16),
            conv(64, 32, 3, 1, dilation=1),
            nn.Conv2d(32,2,3,1,1)
        )

    def forward(self, x):
        return self.convs(x)


class Upsampler(nn.Module):
    def __init__(self, search_range=2, cost_levels=3):
        super(Upsampler, self).__init__()
        self.md = search_range
        self.cost_levels= cost_levels
        assert cost_levels < 5, 'cost levels is not supported for more than 4'
        
        block_enc_ch = [32,64,128,256]
        
        self.encoder = nn.Sequential(
            conv(3,32, kernel_size=3, stride=2),
            conv(32,32, kernel_size=3, stride=1),
            conv(32,32, kernel_size=3, stride=1))
        
        self.decoder=nn.Sequential(
            conv(64 + 32 + 2,32,3,1,1,1),
            resblock(32,32,1),
            resblock(32,32,1),
            resblock(32,32,1),
            conv(32,64,3,1,1,1))
        
        self.block_enc = nn.ModuleList()
        self.motion_dec = nn.ModuleList()
        
        for i in range(self.cost_levels):
            self.block_enc.append(conv(32,block_enc_ch[i],2**i,2**i))
            self.motion_dec.append(conv((self.md*2+1)**2 + 2, 32, 3,1,1,1))
            
        self.motion_dec_agg = conv(32*self.cost_levels, 32, 1,1,1,1)
        
        # self.predict_flow = nn.ConvTranspose2d(64, 2, 4,2,1) 
        self.deconv = nn.ConvTranspose2d(64, 32, 4,2,1)
        self.predict_flow = nn.Conv2d(32, 2, kernel_size = 3, stride=1, padding = 1, dilation = 1, bias= True)
        
        self.context = ContextNetwork(32+2)
        
        # self.bilateral_corr = BilateralCorrelation_Multi(md=1, num_levels=1)
        self.bilateral_corr = bicorr_nn.apply # Cupy module    
        self.leakyRELU = nn.LeakyReLU(0.1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical    
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
    
    def compute_motion_feature(self, x1, x2, flow_fw, time):
        out = []
        
        for l, (block_embed, motion_embed) in enumerate(zip(self.block_enc, self.motion_dec)):
            # sr = 1 / (2**l)
            x1_ = block_embed(x1)
            x2_ = block_embed(x2)
            
            x1_ = L2normalize(x1_)
            x2_ = L2normalize(x2_)
            
            # compute the blockwise bilateral cost volumes
            corr = self.bilateral_corr(x1_,x2_, flow_fw, time, self.md)
            corr = self.leakyRELU(corr)
            
            out.append(motion_embed(torch.cat([corr,flow_fw],dim=1)))
        
        return self.motion_dec_agg(torch.cat(out, dim=1))

    def forward(self, im1, im2, flow_fw, time=0.5):
        _,_, H_,W_= flow_fw.shape
        assert im1.size(-1) == W_ and im1.size(-2) == H_
        
        if type(time) == float:
            t_tmp = time
            time = im1.new_zeros(im1.size(0), 1)
            time += t_tmp
            
        fmap1 = self.encoder(im1)
        fmap2 = self.encoder(im2)   
        
        flow_fw_down = F.interpolate(flow_fw, scale_factor=0.5, mode='bilinear') * 0.5
        
        fmap1 = self.warp(fmap1,flow_fw_down*(-1))
        fmap2 = self.warp(fmap2,flow_fw_down)
        fmap1 = self.leakyRELU(fmap1)
        fmap2 = self.leakyRELU(fmap2)

        corr_feat = self.compute_motion_feature(fmap1, fmap2, flow_fw_down, time)

        dec_feat = self.decoder(torch.cat([fmap1,fmap2, corr_feat, flow_fw_down], dim=1))
        motion_feat = self.deconv(dec_feat)
        flow_res = self.predict_flow(motion_feat)
        
        flow_fw = flow_fw + flow_res
        
        flow_fine = self.context(torch.cat([motion_feat,flow_fw],dim=1))
        
        flow_fw = flow_fw + flow_fine
        
        return flow_fw
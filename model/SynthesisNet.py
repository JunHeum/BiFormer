import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride=stride,
                  padding = padding, dilation = dilation, bias= True),
        nn.LeakyReLU(0.1))


def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y


class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[32, 64, 96], fpn_out=32):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                    for ft_size in feature_channels[1:]])
        self.smooth_conv =  nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)] 
                                    * (len(feature_channels)-1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels)*fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1,inplace=True)
        )

    def forward(self, features):        
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i-1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1]) #P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x


class SynNet(nn.Module):
    def __init__(self):
        super(SynNet, self).__init__()

        self.synthesis_encoder = nn.ModuleList([
            nn.Sequential(
                conv(3, 32,3,1,1),
                conv(32,32,3,1,1)),
            nn.Sequential(
                conv(32,64,3,2,1),
                conv(64,64,3,1,1)),
            nn.Sequential(
                conv(64,96,3,2,1),
                conv(96,96,3,1,1)),
        ])
        
        self.synthesis_decoder = nn.ModuleList([
            nn.Sequential(
                conv(64+6+64,32,3,1,1),
                conv(32,32,3,1,1)),
            nn.Sequential(
                conv(64*2+6+96,64,3,2,1),
                conv(64,64,3,1,1)),
            nn.Sequential(
                conv(96*2+6,96,3,2,1),
                conv(96,96,3,1,1)),
        ])
        
        self.fuse = FPN_fuse([32,64,96],32)
        
        self.predict_frame = nn.Conv2d(32,3,3,1,1)
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

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

    def forward(self,im1,im2, flow_bw, flow_fw):
        _,_, H,W = flow_bw.shape
        x1_pyr = []
        x2_pyr = []
        x_out = []
        
        x1 = im1
        x2 = im2
        
        for i in range(len(self.synthesis_encoder)):
            x1 = self.synthesis_encoder[i](x1)
            x2 = self.synthesis_encoder[i](x2)
            x1_pyr.append(torch.cat([F.interpolate(im1,(x1.size(-2),x1.size(-1)),mode='bilinear'), x1],dim=1))
            x2_pyr.append(torch.cat([F.interpolate(im2,(x2.size(-2),x2.size(-1)),mode='bilinear'), x2],dim=1))
                
        for i in reversed(range(len(self.synthesis_decoder))):
            x1,x2 = x1_pyr[i], x2_pyr[i]
            flow_bw_down = F.interpolate(flow_bw, (x1.size(-2),x1.size(-1)), mode='bilinear')
            flow_fw_down = F.interpolate(flow_fw, (x1.size(-2),x1.size(-1)), mode='bilinear')
            flow_bw_down[:, 0, :, :] *= x1.size(-1) / float(W)
            flow_bw_down[:, 1, :, :] *= x1.size(-2) / float(H)
            flow_fw_down[:, 0, :, :] *= x1.size(-1) / float(W)
            flow_fw_down[:, 1, :, :] *= x1.size(-2) / float(H)
            
            x1 = self.warp(x1, flow_bw_down)
            x2 = self.warp(x2, flow_fw_down)
                        
            if len(x_out) == 0:
                x_out.append(self.synthesis_decoder[i](torch.cat([x1,x2],dim=1)))
            else:
                x_out.append(self.synthesis_decoder[i](torch.cat([x1,x2,F.interpolate(x_out[-1],(x1.size(-2),x1.size(-1)),mode='bilinear')],dim=1)))
            
        x_out = self.fuse(list(reversed(x_out))) # input the x_out with reversed order
        
        I_t = self.predict_frame(x_out)
        
        return I_t

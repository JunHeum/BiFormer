import numpy as np
import torch
import torch.nn.functional as F
from itertools import chain
from torchvision.utils import save_image
from skimage.io import imread

from utils import load_yml2args, text_color
from model import *

torch.manual_seed(4321)
np.random.seed(4321)

torch.set_grad_enabled(False) 
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

COARSE_SIZE = (256,448)
txt_color = text_color()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--first',      type=str,   required=True)
parser.add_argument('--second',     type=str,   required=True)
parser.add_argument('--output',     type=str,   required=True)
parser.add_argument('--target',     type=str,   required=False)
parser.add_argument('--configs',    type=str,   default='configs/BiFormer_paper.yaml')
parser.add_argument('--ckpt_path',  type=str,   default='checkpoints/BiFormer_weights.pth')
args = parser.parse_args()

cfgs = load_yml2args(args.configs)

BiFormer = BiFormer(**cfgs.transformer_cfgs)
Upsampler = Upsampler(**cfgs.Upsampler_cfgs)
SynNet = SynNet()

checkpoint = torch.load(args.ckpt_path,map_location='cpu')
BiFormer.load_state_dict(checkpoint['BiFormer_state_dict'], strict=True)
Upsampler.load_state_dict(checkpoint['Upsampler_state_dict'], strict=True)
SynNet.load_state_dict(checkpoint['SynNet_state_dict'], strict=True)

BiFormer = BiFormer.cuda()
Upsampler = Upsampler.cuda()
SynNet = SynNet.cuda()
              
for param in chain(BiFormer.parameters(), Upsampler.parameters(), SynNet.parameters()):
    param.requires_grad = False


print(f'[{txt_color.yellow("BiFormer")}] Start to interpolate an intermediate frame')
print(f'First input frame: {txt_color.green(args.first)}')
print(f'Second input frame: {txt_color.green(args.second)}')
      
frame1 = torch.from_numpy(imread(args.first)).permute(2,0,1).float().unsqueeze(0) / 255.0
frame3 = torch.from_numpy(imread(args.second)).permute(2,0,1).float().unsqueeze(0) / 255.0

with torch.no_grad():
    frame1 = frame1.cuda()
    frame3 = frame3.cuda()

    _, _, H_ori, W_ori = frame1.shape
    assert (H_ori >= 256 * 8) and (W_ori >= 448 * 8), 'Only 4K resolution available'
    img1_prev = F.interpolate(frame1, COARSE_SIZE, mode='bilinear')
    img3_prev = F.interpolate(frame3, COARSE_SIZE, mode='bilinear')

    flow_fw = BiFormer(img1_prev, img3_prev)
    
    for iter in reversed(range(1,3)):
        H_ = H_ori // (2**iter)
        W_ = W_ori // (2**iter)
        img1_prev = F.interpolate(frame1, (H_,W_), mode='bilinear')
        img3_prev = F.interpolate(frame3, (H_,W_), mode='bilinear')
        
        _,_,H_c,W_c = flow_fw.shape
        flow_fw = F.interpolate(flow_fw, (H_, W_), mode='bilinear')
        flow_fw[:,0,:,:] *= W_ / float(W_c)
        flow_fw[:,1,:,:] *= H_ / float(H_c)
        
        flow_fw = Upsampler(img1_prev, img3_prev, flow_fw)

    _,_,H_c,W_c = flow_fw.shape
    flow_fw = F.interpolate(flow_fw, (H_ori, W_ori), mode='bilinear')
    flow_fw[:,0,:,:] *= W_ori / float(W_c)
    flow_fw[:,1,:,:] *= H_ori / float(H_c)
    
    # Based on linear motion assumption
    flow_bw = flow_fw * (-1)
    
    output = SynNet(frame1, frame3, flow_bw, flow_fw)

    save_image(output.clone(), args.output, value_range=(0,1))
    print(f'Output frame: {txt_color.green(args.output)} interpolated!')

    
    ## This is not the correct PSNR computation due to datatype ##
    if args.target is not None:
        from math import log10
        
        target = torch.from_numpy(imread(args.target)).permute(2,0,1).float().unsqueeze(0) / 255.0
        target = target.cuda()
        
        output = output.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
        output = output.float() / 255.0
        
        mse = F.mse_loss(output, target)
        psnr = 10 * log10(1 / mse.item())
        print(f'PSNR: {psnr:.06f}dB')
    
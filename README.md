# BiFormer (CVPR2023)

Junheum Park,
Jintae Kim, 
and Chang-Su Kim

Official code for **"BiFormer: Learning Bilateral Motion Estimation via Bilateral Transformer for 4K Video Frame Interpolation"**[[paper]](https://arxiv.org/abs/2304.02225)

### Requirements
- PyTorch 1.12.1
- CUDA 11.6
- CuDNN 8.3.2
- python 3.9

### Installation
Create conda environment:
```bash
    $ conda create -n BiFormer python=3.9 anaconda
    $ conda activate BiFormer
    $ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
    $ pip install opencv-python einops timm cupy-cuda116 easydict
```
Download repository:
```bash
    $ git clone https://github.com/JunHeum/BiFormer.git
```
Download [pre-trained model](https://drive.google.com/file/d/13qDB6aZ1sraa4aBsvXgnhcAhQxr7aPzm/view?usp=sharing) parameters:
```bash
    $ unzip BiFormer_Weights.zip
```
### Quick Usage
Generate an intermediate frame on your pair of frames:
```bash
    $ python run.py --first images/im1.png --second images/im3.png --output images/im2.png
```
### Citation
Please cite the following paper if you feel this repository useful.
```bibtex
    @inproceedings{park2023BiFormer,
        author    = {Park, Junheum and Kim, Jintae and Kim, Chang-Su}, 
        title     = {BiFormer: Learning Bilateral Motion Estimation via Bilateral Transformer for 4K Video Frame Interpolation}, 
        booktitle = {Computer Vision and Pattern Recognition},
        year      = {2023}
    }
```
### License
See [Apache License](https://github.com/JunHeum/BiFormer/blob/master/LICENSE)

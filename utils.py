import sys
import numpy as np

from colorama import Style, Fore


def load_yml2args(fn):
    import yaml as yml
    from easydict import EasyDict
    with open(fn,"r") as f:
        config =  yml.safe_load(f)
    return EasyDict(config)


def load_flow(path):
    with open(path, 'rb') as f:
        magic = float(np.fromfile(f, np.float32, count = 1)[0])
        if magic == 202021.25:
            w, h = np.fromfile(f, np.int32, count = 1)[0], np.fromfile(f, np.int32, count = 1)[0]
            data = np.fromfile(f, np.float32, count = h*w*2)
            data.resize((h, w, 2))
            return data
        return None


def writeFlow(filename, uv, v=None):
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    TAG_CHAR = np.array([202021.25], np.float32)
    
    nBands = 2

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def makeColorwheel():
    #  color encoding scheme



	#   http://members.shaw.ca/quadibloc/other/colint.htm
	#   adapted from the color circle idea described at

	RY = 15
	YG = 6
	GC = 4
	CB = 11
	BM = 13
	MR = 6

	ncols = RY + YG + GC + CB + BM + MR

	colorwheel = np.zeros([ncols, 3]) # r g b

	col = 0
	#RY
	colorwheel[0:RY, 0] = 255
	colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
	col += RY

	#YG
	colorwheel[col:YG+col, 0]= 255 - np.floor(255*np.arange(0, YG, 1)/YG)
	colorwheel[col:YG+col, 1] = 255;
	col += YG;

	#GC
	colorwheel[col:GC+col, 1]= 255
	colorwheel[col:GC+col, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
	col += GC;

	#CB
	colorwheel[col:CB+col, 1]= 255 - np.floor(255*np.arange(0, CB, 1)/CB)
	colorwheel[col:CB+col, 2] = 255
	col += CB;

	#BM
	colorwheel[col:BM+col, 2]= 255
	colorwheel[col:BM+col, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
	col += BM;

	#MR
	colorwheel[col:MR+col, 2]= 255 - np.floor(255*np.arange(0, MR, 1)/MR)
	colorwheel[col:MR+col, 0] = 255
	return 	colorwheel

def computeColor(u, v):

	colorwheel = makeColorwheel();
	nan_u = np.isnan(u)
	nan_v = np.isnan(v)
	nan_u = np.where(nan_u)
	nan_v = np.where(nan_v)

	u[nan_u] = 0
	u[nan_v] = 0
	v[nan_u] = 0
	v[nan_v] = 0

	ncols = colorwheel.shape[0]
	radius = np.sqrt(u**2 + v**2)
	a = np.arctan2(-v, -u) / np.pi
	fk = (a+1) /2 * (ncols-1) # -1~1 maped to 1~ncols
	k0 = fk.astype(np.uint8)	 # 1, 2, ..., ncols
	k1 = k0+1
	k1[k1 == ncols] = 0
	f = fk - k0

	img = np.empty([k1.shape[0], k1.shape[1],3])
	ncolors = colorwheel.shape[1]
	for i in range(ncolors):
		tmp = colorwheel[:,i]
		col0 = tmp[k0]/255
		col1 = tmp[k1]/255
		col = (1-f)*col0 + f*col1
		idx = radius <= 1
		col[idx] = 1 - radius[idx]*(1-col[idx]) # increase saturation with radius
		col[~idx] *= 0.75 # out of range
		img[:,:,2-i] = np.floor(255*col).astype(np.uint8)

	return img.astype(np.uint8)


def vis_flow(flow):
	eps = sys.float_info.epsilon
	UNKNOWN_FLOW_THRESH = 1e9
	UNKNOWN_FLOW = 1e10

	u = flow[:,:,0]
	v = flow[:,:,1]

	maxu = -999
	maxv = -999

	minu = 999
	minv = 999

	maxrad = -1
	#fix unknown flow
	greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
	greater_v = np.where(v > UNKNOWN_FLOW_THRESH)
	u[greater_u] = 0
	u[greater_v] = 0
	v[greater_u] = 0
	v[greater_v] = 0

	maxu = max([maxu, np.amax(u)])
	minu = min([minu, np.amin(u)])

	maxv = max([maxv, np.amax(v)])
	minv = min([minv, np.amin(v)])
	rad = np.sqrt(np.multiply(u,u)+np.multiply(v,v))
	maxrad = max([maxrad, np.amax(rad)])
	# print('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n' % (maxrad, minu, maxu, minv, maxv))

	u = u/(maxrad+eps)
	v = v/(maxrad+eps)
	img = computeColor(u, v)
	return img[:,:,[2,1,0]]

class text_color():
	def __init__(self):
		pass
	def green(self, str):
		return Fore.GREEN + Style.BRIGHT + f"{str}" + Style.RESET_ALL

	def red(self, str):
		return Fore.RED + Style.BRIGHT + f"{str}" + Style.RESET_ALL

	def cyan(self, str):
		return Fore.CYAN + Style.BRIGHT + f"{str}" + Style.RESET_ALL

	def yellow(self, str):
		return Fore.YELLOW + Style.BRIGHT + f"{str}" + Style.RESET_ALL

	def blue(self, str):
		return Fore.BLUE + Style.BRIGHT + f"{str}" + Style.RESET_ALL

	def magenta(self, str):
		return Fore.MAGENTA + Style.BRIGHT + f"{str}" + Style.RESET_ALL

	def text_color_help(self):
		print("These text colors are available\n" + self.green("[green]\n")
												  + self.red("[red]\n")
												  + self.cyan("[cyan]\n")
												  + self.yellow("[yellow]\n")
												  + self.blue("[blue]\n")
												  + self.magenta("[magenta]"))
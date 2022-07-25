import torch  
import torch.nn.functional as F 
import numpy as np
import math
from PIL import Image
import cv2
import os


# M: code from https://github.com/pranjaldatta/SSIM-PyTorch/blob/master/SSIM_notebook.ipynb

def gaussian(window_size, sigma):
    """
    Generates a list of Tensor values drawn from a gaussian distribution with standard
    diviation = sigma and sum of all elements = 1.

    Length of list = window_size
    """    
    gauss =  torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):

    # Generate an 1D tensor containing values sampled from a gaussian distribution
    _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)
    
    # Converting to 2D  
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
     
    window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

    return window

# M: SSIM score in [0, 1] where 0 indicates very different and 1 very similar images
def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):

    L = val_range # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

    pad = window_size // 2
    
    try:
        _, channels, height, width = img1.size()
    except:
        channels, height, width = img1.size()

    # if window is not provided, init one
    if window is None: 
        real_size = min(window_size, height, width) # window should be atleast 11x11 
        window = create_window(real_size, channel=channels).to(img1.device)
    
    # calculating the mu parameter (locally) for both images using a gaussian filter 
    # calculates the luminosity params
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2 
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 =  F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    # Some constants for stability 
    C1 = (0.01 ) ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = (0.03 ) ** 2 

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1  
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1 
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean() 
    else: 
        ret = ssim_score.mean(1).mean(1).mean(1)
    
    if full:
        return ret, contrast_metric
    
    return ret

# helper function to load images
load_images = lambda x: np.asarray(Image.open(x).resize((800, 800))) #(480, 640) (504, 378)

# Helper functions to convert to Tensors
tensorify = lambda x: torch.Tensor(x.transpose((2, 0, 1))).unsqueeze(0).float().div(255.0)


def configSSIMParser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument("--img1", type=str, default='', 
                        help='First image to compare.')
    parser.add_argument("--img2", type=str, default='', 
                        help='Second imgae to compare.')

    return parser


def SSIMmain(pathImg1, pathImg2):
    # The true reference Image 
    img1 = load_images(pathImg1)

    # The False image
    img2 = load_images(pathImg2)

    _img1 = tensorify(img1)
    _img2 = tensorify(img2)
    true_vs_false = ssim(_img1, _img2, val_range=255)
    print("True vs False Image SSIM Score:", true_vs_false)

def SSIMAll():
    imgG = load_images('compareSSIM/hotdog/r_2.png') # 3, 3 hotdog
    imgG = imgG[:,:,0:3]
    _imgG = tensorify(imgG)

    dir_path = r'C:/github/DSNeRF/compareSSIM/hotdog_Basic'
    imgs = []
    for path in os.listdir(dir_path):
    # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            imgs.append('compareSSIM/hotdog_Basic/'+path)
    
    print("Read in imgs, starting caluclation")

    bestSSIM = 0
    bestname = ''
    for im in imgs:
        img = load_images(im)
        _img = tensorify(img)
        true_vs_false = ssim(_imgG, _img, val_range=255)

        if true_vs_false > bestSSIM:
            bestSSIM = true_vs_false
            bestname = im

    print("Best SSIM: ", bestSSIM, " at: ", bestname)


#SSIMAll()

if __name__=='__main__':

    parser = configSSIMParser()
    args = parser.parse_args()

    SSIMmain(args.img1, args.img2)
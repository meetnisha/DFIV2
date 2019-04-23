import torch
from torch import nn, optim
from torchvision import datasets
import torchvision
import matplotlib.pyplot as plt
import sys
import numpy as np
from math import log10
import random
import os
sys.path.append("./src")
sys.path.append("./models")
from models.UNet_model import UNetGenerator, UNetDiscriminator
from models.GAN_model import GANGenerator, GANDiscriminator
#import msssim
from tqdm import tqdm_notebook
#from pytorch_msssim import msssim, ssim
#import pytorch_ssim
from pytorch_msssim import msssim, ssim
#from skimage.measure import compare_ssim, compare_mse, compare_psnr, compare_nrmse
def imagesave(img,path="./experiments/"):
    #print('Path Exists', os.path.exists(path))
    #if not os.path.exists(path):
        #os.makedirs(path)
    output_image = img.numpy().transpose((1, 2, 0))
    npimg = np.interp(output_image, (-1.0, 1.0), (0, 255.0)).astype(np.uint8)
    #format H,W,C
    print(path)
    plt.figure(figsize=(20,10))
    plt.imshow(npimg)
    #if os.path.exists(path):
    plt.savefig(path)


def evalGAN(dataloader,load_path,sampleImagesName = None,unet=True):
    """
    :param sampleImagesName: name of the
    :param dataloader: dataloader of eval dataset
    :param load_path: path to model
    :return:
    """
    assert(os.path.exists(load_path),"model dict does not exist")
    
    if unet:
        generator = UNetGenerator()
    else:
        generator = GANGenerator(conv_layers_size=5)
    generator.load_state_dict(torch.load(load_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    
    generator.eval()
    avg_psnr = 0
    ssim_val, msssim_val, psnr = 0, 0, 0
    index_for_sample = random.randint(0,len(dataloader))
    # print(index_for_sample)
    with torch.no_grad():
        with tqdm_notebook(total=len(dataloader)) as pbar:
            for index, sample in enumerate(dataloader):
                # print(index)
                #inframes  (N,C,H,W,2), outframes (N,C,H,W)
                left, right, outframes = sample['left'].to(device),sample['right'].to(device),sample['out'].to(device)
                inframes = (left, right)
                
                generated_data = generator(inframes)
                #if not os.path.exists('/.outframes/val/'):
                    #os.makedirs('/.outframes/val/')
                #imagesave(generated_data.data.cpu(), '/.outframes/val/1.jpg')
                #turn the generated image 
                
                #G_eval = msssim.MultiScaleSSIM(generated_data,outframes)
                #G_eval = msssim(generated_data,outframes)
                #G_eval = pytorch_ssim.ssim(generated_data,outframes).data[0]
                #G_eval = ssim(generated_data,outframes)
                msssim_val = msssim(generated_data,outframes) 
                ssim_val = ssim(generated_data,outframes)
                # print(gen_file, ssim_val) 
                #psnr_val = compare_psnr(generated_data,outframes, multichannel=True)
                G_eval = nn.functional.mse_loss(generated_data,outframes)
                psnr = 10 * log10(1/G_eval.item())

                #avg_psnr += psnr
                avg_psnr += psnr

                # print(index)
                # G_loss = train_GS(discriminator,G_optimizer,outframes,generated_data,criterion,dtype,epoch)
                if index == index_for_sample and sampleImagesName is not None:
                    # N = generated_data.shape[0]
                    n_imgs = generated_data.data.cpu()
                    
                    imagesave(torchvision.utils.make_grid(n_imgs),path=sampleImagesName+"_generated.png")
                    imagesave(torchvision.utils.make_grid(outframes.data.cpu()),path=sampleImagesName+"_real.png")
                    # print("mean red:{}, mean green:{},mean blue:{} ".format(n_imgs[:,0,:,:].mean(),
                    #                                                         n_imgs[:,1,:,:].mean(),
                    #                                                         n_imgs[:,2,:,:].mean()))
                pbar.update(1)
        print('SSIM on val:{:.4f}'.format(ssim_val))
        print('MSSSIM on val:{:.4f}'.format(msssim_val))
        print('Avg. PNSR on val:{:.4f} dB'.format(avg_psnr))
    return ssim_val, msssim_val, avg_psnr/len(dataloader)
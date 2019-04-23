import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets
import torchvision
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import time
sys.path.append("./src")
sys.path.append("./models")
import random
from models.GAN_model import GANGenerator, GANDiscriminator
from models.UNet_model import UNetGenerator, UNetDiscriminator
from src.eval import evalGAN
from tqdm import tqdm_notebook

def imshow(img):
    print('Image device and mean')
    print(img.device)
    print(img.mean())
    output_image = img.numpy().transpose((1,2,0))
    npimg = np.interp(output_image,(-1.0,1.0),(0,255.0)).astype(np.uint8)
    print('Mean of image: {}'.format(npimg.mean()))
    #format H,W,C
    plt.imshow(npimg)
    plt.show()

def save_image(img, path):
    img = img * 0.5 + 0.5
    torchvision.utils.save_image(img, path)

def init_weights(m):
    for param in m.parameters():
        # print(param.shape)
        if len(param.shape) >= 2:
            # print("before: {}".format(param.data[0]))
            torch.nn.init.xavier_uniform_(param)
            # print("after: {}".format(param.data[0]))


def trainGAN(epochs, dataloader, save_path, save_every=None,valloader=None, supervised=True, unet=True,gan = True):
    """
    :param epochs: # of epochs to run for
    :param datasetloader: dataloader of dataset we want to train on
    :param save_path: path to where to save model
    :return: saved models
    """
    #assert not os.path.exists(save_path), 'Experiment folder already exists!'
    if save_every is None:
        save_every = epochs
    height, width = dataloader.dataset.getsize()
    print('Video (h,w): ({}, {})'.format(height,width))
    if unet:
        generator = UNetGenerator()
        if gan:
            discriminator = UNetDiscriminator(height=height, width=width, hidden_size=300)
    else:
        generator = GANGenerator(conv_layers_size=5)
        if gan:
            discriminator = GANDiscriminator(height=height, width=width, hidden_size=300)
    print('Created models')
    
#     print('Model Architecture Generator: ')
#     for name, param in generator.named_parameters():
#         if param.requires_grad:
#             print(name)
#     print('Model Architecture Discriminator: ')
#     for name, param in discriminator.named_parameters():
#         if param.requires_grad:
#             print(name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() >= 1:
        if gan:
            discriminator = nn.DataParallel(discriminator)
        generator = nn.DataParallel(generator)
        for gpu in range(torch.cuda.device_count()):
            print('GPU: {}, number: {}'.format(torch.cuda.get_device_name(gpu),gpu))

    generator.to(device)
    if gan:
        discriminator.to(device)
    # elif torch.cuda.is_available() and torch.cuda.device_count() == 1:
    #     discriminator = discriminator.cuda()
    #     generator = generator.cuda()
    #     device = torch.cuda.FloatTensor

    if gan:
        discriminator.apply(init_weights)
    generator.apply(init_weights)
    print('Initialized weights')
    if gan:
        D_optimizer = optim.Adam(discriminator.parameters())
    G_optimizer = optim.Adam(generator.parameters())

    criterion = nn.BCELoss()
    print('Set up models')
    loss_file = []
    for epoch in range(1, epochs + 1):
        start_time = time.time()
#         index_for_sample = random.randint(0, len(dataloader))
        index_for_sample = len(dataloader) - 1
#         print('Index for sample: {}'.format(index_for_sample))
        with tqdm_notebook(total=len(dataloader)) as pbar:
            for index, sample in enumerate(dataloader):
                #print(sample)
                #inframes  (N,C,H,W,2), real (N,C,H,W)
                left, right, real = sample['left'], sample['right'], sample['out'].to(device)
                #imshow(sample)
                #imshow(left)
                #imshow(right)
                #imshow(real)
                inframes = (left.to(device), right.to(device))
#                 #train discriminator
                gen = generator(inframes).detach()
                d_loss, real_pred, generated_pred = train_D(discriminator,
                                                               D_optimizer,
                                                               real,
                                                               gen,
                                                               criterion,device)

#                 #train generator
                gen = generator(inframes)
                if supervised:
                     g_loss, G0_loss, S_loss = train_GS(discriminator,G_optimizer,real,gen,criterion,device,epoch)
                else:
                     g_loss = train_G(discriminator, G_optimizer, gen, criterion,device)
                gen = generator(inframes)
                #if not os.path.exists('/.outframes/train/'):
                    #os.makedirs('/.outframes/train/')
                #save_image(gen.data.cpu(), '/.outframes/train/1.jpg')
                if gan:
                    g_loss, dg_loss, ds_loss = loss_G(discriminator, generator, G_optimizer, gen, real, supervised)
                    d_loss, dr, dg = loss_D(discriminator, D_optimizer, real, gen)
                else: 
                    g_loss = loss_S(G_optimizer,gen,real)
                    
                if index == len(dataloader) - 1:
                    N = gen.shape[0]
                    gen_cpu = gen.data.cpu()
                    real_cpu = real.data.cpu()
                    gen_grid = torchvision.utils.make_grid(gen_cpu)
                    real_grid = torchvision.utils.make_grid(real_cpu)
                    directory = '{}/{}'.format(save_path, epoch)
                    #print(epoch, os.path.exists(directory))
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    if os.path.exists(directory):
                        save_image(gen_grid, '{}/gengrid.jpg'.format(directory))
                        save_image(real_grid, '{}/realgrid.jpg'.format(directory))
                    print('Generated images')
                    imshow(gen_grid)
                    print('Real images')
                    imshow(real_grid)
                    loss_file.append("epoch {} out of {}".format(epoch, epochs))
                    if gan:
                        loss_file.append("discriminator loss:{}, generator loss:{}".format(d_loss, g_loss))
                    else: 
                        loss_file.append("generator loss:{}".format(g_loss))
                    if supervised and gan:
                        loss_file.append("generator GAN loss:{}, supervised loss:{}".format(dg_loss, ds_loss))
                    if gan:
                        loss_file.append("discriminator mean real prediction:{}, mean fake prediction:{}\n".format(dr, dg))      
                pbar.update(1)
        loss_file.append('runtime: {}'.format(time.time() - start_time))
        
        if epoch % save_every == 0 and save_path is not None:
            torch.save(generator.module.state_dict(), '{}/{}_Generator'.format(save_path, epoch))
            if gan:
                torch.save(discriminator.module.state_dict(), '{}/{}_Discriminator'.format(save_path, epoch))
            if valloader is not None: 
                directory = '{}/{}'.format(save_path, epoch)
                path = '{}/{}_Generator'.format(save_path, epoch)
                valImageName = "{}/val".format(directory)
                ssim_val, msssim_val, avg_psnr = evalGAN(valloader,load_path = path ,sampleImagesName=valImageName, unet=unet)
                loss_file.append('SSIM on val:{:.4f}'.format(ssim_val))
                loss_file.append('MSSSIM on val:{:.4f}'.format(msssim_val))
                loss_file.append('Avg. PNSR on val:{:.4f} dB'.format(avg_psnr))
        for l in loss_file:
            print(l)
        with open('{}/stats.txt'.format(save_path),'a') as f:
            for l in loss_file:
                f.write('{}\n'.format(l))
        loss_file = []

    if epochs % save_every != 0 and save_path is not None:
        torch.save(generator.module.state_dict(), '{}/{}_Generator'.format(save_path, epochs))
        if gan:
            torch.save(discriminator.module.state_dict(), '{}/{}_Discriminator'.format(save_path, epochs))
    if gan:
        return generator, discriminator
    else:
        return generator

def loss_S(g_optim,gen,real):
    loss = torch.nn.functional.smooth_l1_loss(gen,real)
    g_optim.zero_grad()
    loss.backward()
    g_optim.step()
    return loss

def loss_G(d, g, g_optim, gen, real, supervised, lmd=0.0001):
    loss = -torch.mean(F.logsigmoid(d(gen)))
    gen_loss, sup_loss = loss, None
    if supervised:
        sup_loss = torch.nn.functional.smooth_l1_loss(gen, real)
        loss = sup_loss + lmd * gen_loss
    g_optim.zero_grad()
    loss.backward()
    g_optim.step()
#     print('Generator loss')
#     print(loss, gen_loss, sup_loss)
    return loss, gen_loss, sup_loss

def loss_D(d, d_optim, real, gen):
    dr = d(real)
    dg = d(gen.detach())
    loss = -torch.mean(F.logsigmoid(dr)) - torch.mean(F.logsigmoid(-dg))
    d_optim.zero_grad()
    loss.backward()
    d_optim.step()
#     print('D loss, dr, dg')
#     print(loss, torch.sigmoid(dr).mean(), torch.sigmoid(dg).mean())
    return loss, torch.sigmoid(dr).mean(), torch.sigmoid(dg).mean()

def train_D(discriminator,optimizer,real_data,gen,criterion,device):
    """
    :param discriminator: discriminator model
    :param optimizer: optimizer object
    :param real_data: data from dataset (N,C,H,W)
    :param gen: generated frames (N,C,H,W)
    :param criterion: criterion for loss calc
    :return:real_loss+gen_loss,real_output,generated_output
    """
    optimizer.zero_grad()

    real_output = discriminator(real_data)
    N = real_output.shape[0]

    real_loss = criterion(real_output, torch.ones(N,1).to(device))
    real_loss.backward()

    generated_output = discriminator(gen)
    N = generated_output.shape[0]


    gen_loss = criterion(generated_output, torch.zeros(N,1).to(device))
    gen_loss.backward()

    optimizer.step()

    return real_loss+gen_loss, real_output, generated_output

def train_G(discriminator,optimizer,gen,criterion,device):
    """
    :param discriminator: generator model
    :param optimizer: optimizer object
    :param gen: generated frames (N,C,H,W)
    :param criterion: criterion for loss calc
    :return: generated loss
    """
    optimizer.zero_grad()

    output = discriminator(gen)
    N = output.shape[0]

    generated_loss = criterion(output,torch.ones(N,1)).to(device)
    generated_loss.backward()

    optimizer.step()

    return generated_loss

def train_GS(discriminator,optimizer,real_data,gen,criterion,device,epoch,lmd=0.1):
    """
    :param discriminator: generator model
    :param optimizer: optimizer object
    :param real_data: for supervised loss (N,C,H,W)
    :param gen: generated frames (N,C,H,W)
    :param criterion: criterion for loss calc
    :param epoch: adds to decay term of supervised loss
    :return: generated loss
    """
    optimizer.zero_grad()
    output = discriminator(gen)

    N = output.shape[0]
    generated_loss = criterion(output, torch.ones(N,1).to(device))
    supervised_loss = torch.nn.functional.smooth_l1_loss(gen, real_data)
    # print("generated_loss: {}".format(generated_loss))
    total_loss = lmd * generated_loss + supervised_loss
    total_loss.backward()

    optimizer.step()
    return total_loss, generated_loss, supervised_loss

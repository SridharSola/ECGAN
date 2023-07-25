#imports
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
import argparse
import matplotlib.pyplot as plt
import random
import os
import matplotlib.pyplot as plt
import glob
import random
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

from Raf_dataset import RAafDataset
from ECGAN import cgan_inpaint_in
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #To use GPU
torch.manual_seed(random.randint(1,1000))


def main():
    

    #Dataset
    train_set = RafDataset()
    test_set = RafDataset(partition = 'test')

    #Parameters
    adv_criterion = nn.BCEWithLogitsLoss()
    #adv_criterion = nn.MSELoss()
    recon_criterion = nn.L1Loss()
    lambda_recon = 100
    lambda_Dwhole = 0.3
    lambda_Dmask = 0.7
    lambda_adv_whole = 0.3
    lambda_adv_mask = 0.7

    n_epochs=2
    input_dim = 6#Correct
    output_dim = 3
    disc_dim = 9
    display_step = 400    # 10 times
    batch_size = 3
    lr = 0.0003
    target_shape = 256

    #Change
    cur_step = 20400
    device = 'cuda'

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    model_path = '/content/drive/MyDrive/Face_Inpainting_1/Models/Inpaint_CUNet_36400.pth'         #Change#
    gen, gen_opt, disc_whole, disc_whole_opt, disc_mask, disc_mask_opt = cgan_inpaint_in(True, model_path)
    gen.to(device)
    disc_whole.to(device)
    disc_mask.to(device)


    if trn = True:
        #TRAIN
        train(train_set,cur_step, gen, gen_opt, disc_whole, disc_mask, disc_whole_opt, disc_mask_opt, save_model=False)
    else:
        #Test
        #Only saves unmasked images and does not calculate metrics to avoid crashing
        gener(test_set, gen)
    
    




def train(train_set,cur_step, gen, gen_opt, disc_whole, disc_mask, disc_whole_opt, disc_mask_opt, batch_size = 3, save_model=False):
    mean_generator_loss = 0
    mean_disc_whole_loss = 0
    mean_disc_mask_loss = 0
    fake_features_list = []
    real_features_list = []
    dataloader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
    ##change## (0, 2000, 4800, 7600, 13200, 17200, 20400)
    gen.train()
    disc_whole.train()
    disc_mask.train()

    for epoch in range(1,n_epochs+1):
        for gt,mask,binary, label in tqdm(dataloader):
            gt = gt.to(device)
            mask = mask.to(device)
            binary = binary.to(device)
            label = label.to(device)
            #print(mask.shape, binary.shape, label.shape)

            with torch.no_grad():
                input_imgs = torch.cat((mask,binary),1)
                fake = gen(input_imgs, label)



            if cur_step%5==0:
                disc_whole_opt.zero_grad()
                disc_whole_loss = discwhole_loss_func(disc_whole,gt,mask,binary,label, fake,adv_criterion,lambda_Dwhole)
                disc_whole_loss.backward(retain_graph=True)
                disc_whole_opt.step()
                mean_disc_whole_loss += disc_whole_loss.item()/4

                if cur_step>=17200: #3516*6: #Add disc_mask (paper implementation after 50% epochs. Assume 50k for us => after 20k-25k iters)
                    disc_mask_opt.zero_grad()
                    disc_mask_loss = discmask_loss_func(disc_mask,gt,fake,mask,binary, label, adv_criterion,lambda_Dmask)
                    disc_mask_loss.backward(retain_graph=True)
                    disc_mask_opt.step()
                    mean_disc_mask_loss += disc_mask_loss.item()/4


            gen_opt.zero_grad()
            gen_loss,fake,l1_loss,ssim_loss,perceptual_loss = generator_loss(cur_step,gen,disc_whole,disc_mask,
                                                               gt,mask,binary, label,
                                                               adv_criterion,recon_criterion,
                                                               lambda_recon,lambda_adv_whole,lambda_adv_mask)
            gen_loss.backward()
            gen_opt.step()
            mean_generator_loss += gen_loss.item()/20


            if cur_step%40 == 0:#Changed from 20
                real_features = inception_model(gt.to(device)).detach().to('cpu')    #FID
                real_features_list.append(real_features)
                fake_features = inception_model(fake.to(device)).detach().to('cpu')
                fake_features_list.append(fake_features)
                fake_features_all = torch.cat(fake_features_list)            #FID
                real_features_all = torch.cat(real_features_list)
                mu_fake = fake_features_all.mean(dim=0)
                mu_real = real_features_all.mean(dim=0)
                sigma_fake = get_covariance(fake_features_all)
                sigma_real = get_covariance(real_features_all)
                FID = frechet_distance(mu_real,mu_fake,sigma_real,sigma_fake).item()

                fid_file = open('/content/drive/MyDrive/Face_Inpainting_1/FID/FID1.txt','a')       ##change##
                fid_file.write(str(cur_step)+"\n")
                fid_file.write(str(round(FID,4))+"\n"+"\n")
                fid_file.close()
                fake_features_list.clear()
                real_features_list.clear()

                loss_file = open('/content/drive/MyDrive/Face_Inpainting_1/Loss/Loss.txt','a')     ##change##
                loss_file.write(str(cur_step)+"\n")
                loss_file.write(str(round(mean_generator_loss,4))+"    "+str(round(mean_disc_whole_loss,4))+"    "+str(round(mean_disc_mask_loss,4))+"    ")
                loss_file.write(str(round(l1_loss.item(),4))+"    "+str(round(1-ssim_loss.item(),4))+"    "+str(round(perceptual_loss.item(),4)))
                loss_file.write("\n"+"\n")
                loss_file.close()
                mean_generator_loss = 0
                mean_disc_whole_loss = 0
                mean_disc_mask_loss = 0

            if cur_step%display_step == 0:
                print(f"Epoch {epoch}: Step {cur_step}")
                show_tensor_images(gt,size=(3,target_shape,target_shape))
                show_tensor_images(mask,size=(3,target_shape,target_shape))
                show_tensor_images(fake,size=(3,target_shape,target_shape))

                print("    ", get_class(label[0].item()),"    ", get_class(label[1].item()),"    ", get_class(label[2].item()))



            if save_model and cur_step%display_step == 0: #saving every 1/4th epoch to prevent loss of training in case of error/crash
                print('Saving Model')
                torch.save({'gen':gen.state_dict(),
                          'gen_opt':gen_opt.state_dict(),
                          'disc_whole':disc_whole.state_dict(),
                          'disc_whole_opt':disc_whole_opt.state_dict(),
                          'disc_mask':disc_mask.state_dict(),
                          'disc_mask_opt':disc_mask_opt.state_dict()},
                          f"/content/drive/MyDrive/Face_Inpainting_1/Models/Inpaint_CUNet_{cur_step}.pth" )
            cur_step += 1


def gener(test_set, gen, batch_size = 3):
    device = 'cuda'
    dataloader = DataLoader(test_set,batch_size=3,shuffle=False)
    cur_step = 0              ##change## (0, 2000, 4800, 7600, 13200, 17200)
    #gen.eval()
    #disc_whole.eval()
    #disc_mask.eval()
    #emb  = nn.Embedding(7, 7)
    #emb.to(device)
    gen.to(device)
    i = 0
    fake_features_list = []
    real_features_list = []
    ploss = []
    fid = []
    ssim = []
    for gt,mask,binary, label, name in tqdm(dataloader):
            gt = gt.to(device)
            mask = mask.to(device)
            binary = binary.to(device)
            label = label.to(device)
            with torch.no_grad():
                input_imgs = torch.cat((mask,binary),1)
                #x = mask
                #input = torch.cat((x, label_emb.repeat(1, 1, x.size(2), x.size(3))), dim=1)
                fake = gen(input_imgs, label)
            #input_imgs = torch.cat((mask,binary),1)
            #fake = gen(input_imgs, label)
   
            
                
            if True:
                print(f"Step {cur_step}")
                #show_tensor_images(gt,size=(3,target_shape,target_shape))
                #show_tensor_images(mask,size=(3,target_shape,target_shape))
                #show_tensor_images(binary,size=(3,target_shape,target_shape))
                #fake = fake.permute(1, 2, 0)
                #show_tensor_images(i, fake, label,size=(3,target_shape,target_shape) )
                image_shifted = fake
                size = (3,256,256)
                image_unflat = image_shifted.detach().cpu().view(-1, *size)
                image_grid = make_grid(image_unflat[:3], nrow=5)
                f = open("/content/drive/MyDrive/Face_Inpainting_1/gen_rafdb.txt", 'a')
                for j in range(len(name)):
                      n = '/content/drive/MyDrive/Face_Inpainting_1/Inpainted_RAFDBcomp/' + name[j]
                      torchvision.utils.save_image(image_unflat[j], n, normalize = True)
                      #print(n, "    ", str(label[j].item()), file = f)
                      i = i+1

                print("      ", get_class(label[0].item()),"      ", get_class(label[1].item()),"      ", get_class(label[2].item()))


            
            cur_step += 1
          

    print(i)


if __name__ == "__main__":
    
    main()
    print("Complete!")


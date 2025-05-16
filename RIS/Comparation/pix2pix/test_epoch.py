# from load_data getting data and into the Cyclegan model, finnally outputting outcome of 3D tensor.
from load_data import MyDataLoader
from pix2pix import Pix2PixModel
from save_img import save_img

import os
import nibabel as nib
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import normalized_mutual_information as MI
from skimage.metrics import mean_squared_error as MSE
import math
import xlrd
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch
import pandas as pd

def my_psnr(im1, im2):
    return 10 * math.log10(im1.max() * im2.max() / MSE(im1, im2))

if __name__ == '__main__':
    istrain = False
    ispretrain = False
    test_epoch = 80
    dataset = MyDataLoader(istrain,ispretrain)
    read_dataset = dataset.get_dataset()
    dataset.load_data()
    
    print("data is loading")
    dataset_size = len(dataset)
    batchsize = dataset.get_batchsize()
    model = Pix2PixModel(istrain,ispretrain)
    model.set_load_model_epoch("1")
    model.setup(read_dataset.get_slice_scope())
    save_img = save_img(True,ispretrain,read_dataset.get_slice_scope(),ispretrain,test_epoch)
    for epoch in range(1,81):
        dataset = MyDataLoader(istrain,ispretrain)
        read_dataset = dataset.get_dataset()
        dataset.load_data()
        print("data is loading")
        dataset_size = len(dataset)
        batchsize = dataset.get_batchsize()
        
        model = Pix2PixModel(istrain,ispretrain)
        model.set_load_model_epoch(epoch)
        model.setup(read_dataset.get_slice_scope())        
        save_nii_every_slices = dataset.get_dataset().get_T1_slices()
    
        for i,data in enumerate(dataset):
            model.set_input(data)
            model.test()
            image_path = model.get_image_paths()  # for example '114217_T1w_brain.nii.gz0'
            # save_img.save_as_png(model.get_current_visuals(),image_path,200,False)
            save_img.save_as_nii(model.get_current_visuals(),image_path,epoch,batchsize,True,read_dataset.get_T1_slices(),read_dataset.get_T2_slices(),read_dataset.T1_dict,read_dataset.B0_dict,read_dataset.get_affine_list(),read_dataset.get_size())
    save_img.quantitative.close_file()

    cwd = os.getcwd()
    real_brian_path1 = os.path.join(cwd,"monkey_brain","test_b0_data")
    fake_brian_path1 = os.path.join(cwd,"monkey_brain",r"results/web/nii")
    
    real_brain_path2 = os.listdir(real_brian_path1)
    fake_brain_path2 = os.listdir(fake_brian_path1)
    
    real_brain_path2.sort()
    fake_brain_path2.sort()
    ssim_list = []
    data_list = []
    psnr_list = []
    mi_list = []
    
    for index in range(len(fake_brain_path2)):
        real_path = os.path.join(real_brian_path1,real_brain_path2[index%len(real_brain_path2)])
        fdata = nib.load(real_path).get_fdata()
        real_brain_tensor = torch.tensor(fdata)
        
        fake_path = os.path.join(fake_brian_path1,fake_brain_path2[index])
        fake_fdata = nib.load(fake_path).get_fdata()
        
        dMRI_path = os.path.join(fake_dMRI_brain_path,fake_dMRI_brain_path2[index])
        dMRI_fake_fdata = nib.load(dMRI_path).get_fdata()
        
        fake_brain_tensor = torch.tensor(fake_fdata)
        zeros_0 = torch.zeros_like(real_brain_tensor[0, :, :])
        
        size_0 = real_brain_tensor.size()[0]
        j = 0
        for i in range(size_0):
            if (torch.equal(real_brain_tensor[j, :, :], zeros_0)):
                real_brain_tensor = torch.cat((real_brain_tensor[0:j],real_brain_tensor[j+1:]))
                fake_brain_tensor = torch.cat((fake_brain_tensor[0:j],fake_brain_tensor[j+1:]))
            else:
                j += 1
        j = 0
        zeros_1 = torch.zeros_like(real_brain_tensor[:, 0, :])
        for a in range(real_brain_tensor.size()[1]):
            if (torch.equal(real_brain_tensor[:, j, :], zeros_1)):
                real_brain_tensor = torch.cat((real_brain_tensor[:, 0:j, :],real_brain_tensor[:,j+1:,:]),dim=1)
                fake_brain_tensor = torch.cat((fake_brain_tensor[:, 0:j, :],fake_brain_tensor[:,j+1:,:]),dim=1)
            else:
                j += 1
        
        j = 0
        zeros_2 = torch.zeros_like(real_brain_tensor[:, :, 0])
        for b in range(real_brain_tensor.size()[2]):
            if (torch.equal(real_brain_tensor[:, :, j], zeros_2)):
                real_brain_tensor = torch.cat((real_brain_tensor[:, :, 0:j],real_brain_tensor[:,:,j+1:]),dim=2)
                fake_brain_tensor = torch.cat((fake_brain_tensor[:, :, 0:j],fake_brain_tensor[:,:,j+1:]),dim=2)
            else:
                j += 1
    
        print(real_brain_path2[index%len(real_brain_path2)])
        print(fake_brain_path2[index])
        
        print(real_brain_tensor.size())
        print(fake_brain_tensor.size())
        
        if fake_brain_tensor.size() != real_brain_tensor.size():
            continue
        ssim = SSIM(fake_brain_tensor.numpy(),real_brain_tensor.numpy())
        psnr = my_psnr(fake_brain_tensor.numpy(),real_brain_tensor.numpy())
        mi = MI(fake_brain_tensor.numpy(),real_brain_tensor.numpy())
        
        ssim_list.append(ssim)
        psnr_list.append(psnr)
        mi_list.append(mi)
        data_list.append(fake_brain_path2[index][0:-4])
        print(ssim)
        print(psnr)
        print(mi)
    
    
    ssim_data = pd.DataFrame(data=np.array([ssim_list,psnr_list,mi_list]).T,
             index=data_list,
             columns=["ssim","psnr","mi"])
    
    ssim_data.to_excel(excel_writer=os.path.join(cwd,"monkey_brain","ssim_psnr_mi_uwm_1_80.xlsx"))
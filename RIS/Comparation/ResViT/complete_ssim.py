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

cwd = os.getcwd()
real_brian_path1 = os.path.join(cwd,"monkey_brain","test_B0_data")
fake_brian_path1 = os.path.join(cwd,"monkey_brain","fake_brain")

real_brain_path2 = os.listdir(real_brian_path1)
fake_brain_path2 = os.listdir(fake_brian_path1)
real_brain_path2.sort()
fake_brain_path2.sort()
ssim_list = []
data_list = []
psnr_list = []
mi_list = []

for index in range(len(real_brain_path2)):
    real_path = os.path.join(real_brian_path1,real_brain_path2[index])
    fdata = nib.load(real_path).get_fdata()
    real_brain_tensor = torch.tensor(fdata)
    zeros_0 = torch.zeros_like(real_brain_tensor[0, :, :])

    size_0 = real_brain_tensor.size()[0]
    j = 0
    for i in range(size_0):
        if (torch.equal(real_brain_tensor[j, :, :], zeros_0)):
            real_brain_tensor = torch.cat((real_brain_tensor[0:j],real_brain_tensor[j+1:]))
        else:
            j += 1
    j = 0
    zeros_1 = torch.zeros_like(real_brain_tensor[:, 0, :])
    for a in range(real_brain_tensor.size()[1]):
        if (torch.equal(real_brain_tensor[:, j, :], zeros_1)):
            real_brain_tensor = torch.cat((real_brain_tensor[:, 0:j, :],real_brain_tensor[:,j+1:,:]),dim=1)
        else:
            j += 1

    j = 0
    zeros_2 = torch.zeros_like(real_brain_tensor[:, :, 0])
    for b in range(real_brain_tensor.size()[2]):
        if (torch.equal(real_brain_tensor[:, :, j], zeros_2)):
            real_brain_tensor = torch.cat((real_brain_tensor[:, :, 0:j],real_brain_tensor[:,:,j+1:]),dim=2)
        else:
            j += 1

    fake_path = os.path.join(fake_brian_path1,fake_brain_path2[index])
    fake_fdata = nib.load(fake_path).get_fdata()
    fake_brain_tensor = torch.tensor(fake_fdata)
    zeros_0 = torch.zeros_like(fake_brain_tensor[0, :, :])

    size_0 = fake_brain_tensor.size()[0]
    j = 0
    for i in range(size_0):
        if (torch.equal(fake_brain_tensor[j, :, :], zeros_0)):
            fake_brain_tensor = torch.cat((fake_brain_tensor[0:j],fake_brain_tensor[j+1:]))
        else:
            j += 1
    j = 0
    zeros_1 = torch.zeros_like(fake_brain_tensor[:, 0, :])
    for a in range(fake_brain_tensor.size()[1]):
        if (torch.equal(fake_brain_tensor[:, j, :], zeros_1)):
            fake_brain_tensor = torch.cat((fake_brain_tensor[:, 0:j, :],fake_brain_tensor[:,j+1:,:]),dim=1)
        else:
            j += 1
    j = 0
    zeros_2 = torch.zeros_like(fake_brain_tensor[:, :, 0])
    for b in range(fake_brain_tensor.size()[2]):
        if (torch.equal(fake_brain_tensor[:, :, j], zeros_2)):
            fake_brain_tensor = torch.cat((fake_brain_tensor[:, :, 0:j],fake_brain_tensor[:,:,j+1:]),dim=2)
        else:
            j += 1
    print(real_brain_path2[index])
    print(real_brain_tensor.size())
    print(fake_brain_tensor.size())
    ssim = SSIM(fake_brain_tensor.numpy(),real_brain_tensor.numpy())
    psnr = my_psnr(fake_brain_tensor.numpy(),real_brain_tensor.numpy())
    mi = MI(fake_brain_tensor.numpy(),real_brain_tensor.numpy())

    ssim_list.append(ssim)
    psnr_list.append(psnr)
    mi_list.append(mi)
    data_list.append(real_brain_path2[index][0:8])
    print(ssim)
    print(psnr)
    print(mi)


ssim_data = pd.DataFrame(data=np.array([ssim_list,psnr_list,mi_list]).T,
             index=data_list,
             columns=["ssim","psnr","mi"])

ssim_data.to_excel(excel_writer=os.path.join(cwd,"monkey_brain","ssim_psnr_mi.xlsx"))



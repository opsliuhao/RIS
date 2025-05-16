import math
import os

import skimage.metrics
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import normalized_mutual_information as MI

import matplotlib.pyplot as plt
import csv
import numpy as np

# completing ssim,psnr,mse of T1 and T2 and saving to csv respectively and altogether

class quantitative():
    def __init__(self, istrain, ispretrain, slice_scope, max_epoch):

        self.istrain = istrain
        self.ispretrain = ispretrain
        if self.ispretrain:
            self.results = "/pretrain_results"
        else:
            self.results = "/results"
        self.path = os.path.join(os.getcwd(), "monkey_brain")
        self.csv_list = []
        self.epoch = None
        self.length = None
        self.B0_dict = {
            'B0_psnr': [],
            'B0_ssim': [],
            'B0_mi': [],
            'B0_mse': [],
        }

        self.ave_dict = {
            'psnr': [],
            'ssim':[],
            'epoch':[],
        }

        self.max_epoch = max_epoch
        if self.istrain:
            self.csvfile = open(
                    self.path + self.results + "/train_quantitative" + ".csv", mode='w', newline="")
            header = ["epoch", "DATA", "PSNR", "SSIM", "MSE" , "MI"]
        else:
            self.csvfile = open(
                self.path + self.results + "/test_quantitative(" + str(max_epoch)+ ").csv", mode='w', newline="")
            header = ["DATA", "PSNR", "SSIM", "MSE"]
        self.csvwriter = csv.writer(self.csvfile)
        self.csvwriter.writerow(header)

        # T1 list

        if not self.istrain:
            if self.ispretrain:
                self.csvT1 = open(
                    self.path + self.results + "/val_quantitative_real_B0" + ".csv", mode='w', newline="")
                val_header = ["epoch", "DATA", "PSNR", "SSIM", "MSE"]
                self.csvwriter_T1 = csv.writer(self.csvT1)
                self.csvwriter_T1.writerow(val_header)

    def my_psnr(self,im1,im2):
        return 10*math.log10(im1.max() * im2.max() / MSE(im1,im2))

    def draw_plt(self,epoch,ssim,psnr):
        plt.subplot(1, 2, 1)
        plt.scatter(epoch, ssim, c='r')

        plt.subplot(1, 2, 2)
        plt.scatter(epoch, psnr, c='r')

        plt.show()

    def complete(self, im1, im2, data, epoch):

        psnr = self.my_psnr(im1,im2)
        ssim = SSIM(im1, im2)
        mse = MSE(im1, im2)
        mi = MI(im1, im2)
        # 计算指标平均值与存储
        if not self.istrain:
            if (self.epoch != epoch and self.epoch!=None):

                print(
                "epoch:%s, average_ssim is %s, average_psnr is %s, average_mi is %s" % (
                self.epoch,
                sum(self.B0_dict['B0_ssim'])/len(self.B0_dict['B0_ssim']),
                sum(self.B0_dict['B0_psnr'])/len(self.B0_dict['B0_psnr']),
                sum(self.B0_dict['B0_mi'])/len(self.B0_dict['B0_mi']),
                ))

                self.ave_dict['psnr'].append(sum(self.B0_dict['B0_psnr'])/len(self.B0_dict['B0_psnr']))
                self.ave_dict['ssim'].append(sum(self.B0_dict['B0_ssim'])/len(self.B0_dict['B0_ssim']))
                self.ave_dict['epoch'].append(self.epoch)

                self.length = len(self.B0_dict['B0_ssim'])
                self.draw_plt(self.ave_dict['epoch'],
                              self.ave_dict['ssim'],
                              self.ave_dict['psnr']
                              )
                self.B0_dict['B0_psnr'].clear()
                self.B0_dict['B0_mi'].clear()
                self.B0_dict['B0_ssim'].clear()
                self.B0_dict['B0_mse'].clear()

            if (data[-1] =="0"):
                # self.B0_psnr.append(psnr)
                self.B0_dict['B0_ssim'].append(ssim)
                self.B0_dict['B0_mi'].append(mi)
                self.B0_dict['B0_mse'].append(mse)
                self.B0_dict['B0_psnr'].append(psnr)
            else:
                pass

            if self.length == len(self.B0_dict['B0_mi']) and self.epoch == self.max_epoch:

                print(
                    "epoch:%s, average_ssim is %s, average_psnr is %s, average_mi is %s" % (
                        epoch,
                        sum(self.B0_dict['B0_ssim']) / len(self.B0_dict['B0_ssim']),
                        sum(self.B0_dict['B0_psnr']) / len(self.B0_dict['B0_psnr']),
                        sum(self.B0_dict['B0_mi']) / len(self.B0_dict['B0_mi']),
                    ))
                self.draw_plt(self.ave_dict['epoch'],
                              self.ave_dict['ssim'],
                              self.ave_dict['psnr'],
                              )
        self.epoch = epoch
        # 计算指标与存储
        if self.istrain:
            self.csv_list = [epoch, data, psnr,ssim, mse, mi]
            print(
            "epoch:%s, data:%s, psnr is %s, ssim is %s, mse is %s, mi is %s " % (epoch, data, psnr, ssim, mse, mi))
        else:
            self.csv_list = [epoch, data, psnr, ssim, mse, mi]
            print("data:%s, psnr is %s, ssim is %s, mse is %s, mi is %s " % (data, psnr, ssim, mse, mi))
            # self.csv_list = [data, psnr, ssim, mse, mi]
            self.csv_list = [data, psnr, ssim, mse, mi]

        self.csvwriter.writerow(self.csv_list)
        self.csv_list.clear()
        if not self.istrain and self.ispretrain:
            self.csv_list_T2 = [self.epoch-1, data, psnr, ssim, mse, mi]
            self.csvwriter_T1.writerow(self.csv_list_T2)
            self.csv_list_T2.clear()

        return {"psnr":psnr, "ssim": ssim, "mse": mse, "mi": mi}

    def close_file(self):
        self.csvfile.close()
        if not self.istrain and self.ispretrain:
            self.csvT1.close()





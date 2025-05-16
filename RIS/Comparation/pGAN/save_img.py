import os
import time
import imageio
import nibabel as nib
import torch
from quantitative import quantitative
import torch.nn.functional as F
class save_img():
    def __init__(self,istrain,pretrain,slice_scope,max_epoch):
        # torch.set_printoptions(profile="full")


        self.public = os.getcwd()
        self.istrain = istrain
        self.pretrain = pretrain
        if self.istrain:
            mask_paths = os.path.join(self.public, "monkey_brain", "train_mask")
            smask_path = os.listdir(mask_paths)
            smask_path.sort()
        else:
            if self.pretrain:
                mask_paths = os.path.join(self.public,"monkey_brain","val_mask")
                smask_path = os.listdir(mask_paths)
                smask_path.sort()
            else:
                mask_paths = os.path.join(self.public,"monkey_brain","test_mask")
                smask_path = os.listdir(mask_paths)
                smask_path.sort()
        self.mask_list = []

        if not os.path.exists(os.path.join(self.public, "monkey_brain", "test_mask_pt")):
            os.mkdir(os.path.join(self.public, "monkey_brain", "test_mask_pt"))
        if not os.path.exists(os.path.join(self.public, "monkey_brain", "train_mask_pt")):
            os.mkdir(os.path.join(self.public, "monkey_brain", "train_mask_pt"))
        if not os.path.exists(os.path.join(self.public, "monkey_brain", "val_mask_pt")):
            os.mkdir(os.path.join(self.public, "monkey_brain", "val_mask_pt"))

        for mask_path in smask_path:
            if not self.istrain:
                torch.save(torch.tensor(nib.load(os.path.join(mask_paths,mask_path)).get_fdata()),
                       os.path.join(self.public, "monkey_brain", "test_mask_pt", mask_path[0:6]+".pt"))
                self.mask_list.append(os.path.join(self.public, "monkey_brain", "test_mask_pt", mask_path[0:6]+".pt"))
            else:
                if self.pretrain:
                    torch.save(torch.tensor(nib.load(os.path.join(mask_paths, mask_path)).get_fdata()),
                               os.path.join(self.public, "monkey_brain", "val_mask_pt", mask_path[0:6] + ".pt"))
                    self.mask_list.append(
                        os.path.join(self.public, "monkey_brain", "val_mask_pt", mask_path[0:6] + ".pt"))
                else:
                    torch.save(torch.tensor(nib.load(os.path.join(mask_paths,mask_path)).get_fdata()),
                        os.path.join(self.public, "monkey_brain", "train_mask_pt", mask_path[0:6]+".pt"))
                    self.mask_list.append(os.path.join(self.public, "monkey_brain", "train_mask_pt", mask_path[0:6]+".pt"))

        if self.pretrain:
            self.results = "pretrain_results"
            self.train_img_dir = os.path.join(self.public,"monkey_brain/"+self.results+"/web/pretrain_images")
            self.train_nii_dir = os.path.join(self.public,"monkey_brain/"+self.results+"/web/pretrain_nii")
        else:
            self.results = "results"
            self.train_img_dir = os.path.join(self.public,"monkey_brain/"+self.results+"/web/images")
            self.train_nii_dir = os.path.join(self.public,"monkey_brain/"+self.results+"/web/nii")
        self.test_img_dir = os.path.join(self.public,"monkey_brain/"+self.results+"/web/test_images")
        self.test_nii_dir = os.path.join(self.public,"monkey_brain/"+self.results+"/web/test_nii")
        self.path = os.path.join(self.public,"monkey_brain")
        self.log_name = os.path.join(self.path,self.results,"loss_log.txt")
        self.test_log_name = os.path.join(self.path,self.results,"test_loss_log.txt")
        self.train_quantitative_log_name = os.path.join(self.path,self.results,"train_quantitative_log.txt")

        self.run_save_nii = 1
        self.T1_nii_data = None
        self.B0_nii_data = None
        self.save_T1_nii_data = None
        self.save_B0_nii_data = None
        # for compare the ssim,psnr and mse
        self.real_T1_nii_data = None
        self.real_T2_nii_data = None
        self.real_save_T1_nii_data = None
        self.real_save_T2_nii_data = None

        # self.T1_dict_list_down = []
        # self.T1_dict_list_minm = []
        # self.B0_dict_list_down = []
        # self.B0_dict_list_minm = []
        self.affine_index = 0
        
        # self.bool_fakeA = False
        # self.bool_fakeB = False
        #if not exists path, then create them.
        if not os.path.exists(os.path.join(self.path,self.results)):
            os.mkdir(os.path.join(self.path,self.results))
        if not os.path.exists(os.path.join(self.path,self.results,"web")):
            os.mkdir(os.path.join(self.path,self.results,"web"))
        if not os.path.exists(self.train_img_dir):
            os.mkdir(self.train_img_dir)
        if not os.path.exists(self.train_nii_dir):
            os.mkdir(self.train_nii_dir)
        if not os.path.exists(self.test_nii_dir):
            os.mkdir(self.test_nii_dir)
        if not os.path.exists(self.test_img_dir):
            os.mkdir(self.test_img_dir)

        if self.istrain:
            with open(self.log_name,"a") as log_file:
            # 返回以可读字符串表示的当地时间。
                now = time.strftime("%c")
                log_file.write('=======================Traing loss (%s) =============================\n' % now)
            with open(self.train_quantitative_log_name, "a") as train_quantitative_log_file:
            # 返回以可读字符串表示的当地时间。
                now = time.strftime("%c")
                train_quantitative_log_file.write('=======================train quantitative loss (%s) =============================\n' % now)
        else:
            with open(self.test_log_name, "a") as test_quantitative_log_file:
            # 返回以可读字符串表示的当地时间。
                now = time.strftime("%c")
                test_quantitative_log_file.write('=======================Test quantitative loss (%s) =============================\n' % now)
        # quantitative 
        self.quantitative = quantitative(istrain,self.pretrain,slice_scope,max_epoch)
    def print_current_losses(self,epoch,iters,losses,t_comp,t_data):
        message = '(epoch:%d, iters:%d, time:%.3f,data:%.3f)' % (epoch,iters,t_comp,t_data)

        for k,v in losses.items():
            message += "%s:%.3f "%(k,v)
        print(message)

        with open(self.log_name,"a") as log_file:
            log_file.write('%s\n' %message)
            
    def print_current_quantitative(self,quantitative,image_path,epoch):
        if self.istrain:
            message = '(epoch:%s )' %(epoch)
            message += '(image_path:%s )' % (image_path)
        else:
            message = '(image_path:%s )' % (image_path)

        for k,v in quantitative.items():
            message += "%s:%s "%(k,v)
            #print(message)
        if self.istrain:
            with open(self.train_quantitative_log_name, "a") as train_quantitative_log_file:
                train_quantitative_log_file.write('%s\n' % message)
        else:
            with open(self.test_log_name,"a") as test_log_file:
                test_log_file.write('%s\n' % message)

    def save_as_png(self,visuals,image_path,epoch,istrain):
        for label,image in visuals.items():
            if istrain:
                img_path_T1 = os.path.join(self.train_img_dir, "epoch%3d_%3s%sT1_%s.png" % (epoch,label,image_path[0][0:8],image_path[0][-1]))
                img_path_B0 = os.path.join(self.train_img_dir, "epoch%3d_%3s%sB0_%s.png" % (epoch,label,image_path[0][0:8],image_path[0][-1]))
            else:
                img_path_T1 = os.path.join(self.test_img_dir, "%3s%sT1_%s.png" % (label, image_path[0][0:8],image_path[0][-1]))
                img_path_B0 = os.path.join(self.test_img_dir, "%3s%sB0_%s.png" % (label, image_path[0][0:8],image_path[0][-1]))
            image = image[0,:,:,:].squeeze(dim=0).squeeze(dim=0)

            #set print tensor is full, not shenglue...
            #torch.set_printoptions(profile="full")
            imageio.imwrite(img_path_T1,(image*255).detach().cpu().numpy())
            imageio.imwrite(img_path_B0,(image*255).detach().cpu().numpy())
    def save_as_nii(self,visuals,image_path,epoch,batchsize,istrain,T1_slices,T2_slices,T1_dict,B0_dict,affine_list,size_dict,save_nii):
        #save as nii every all slices(every nii)
        #get the affine and data of T1 and T2
        T1_total_slices = T1_slices
        T2_total_slices = T2_slices
        assert T1_total_slices == T2_total_slices
        pred_B0_max_list = affine_list['pred_B0_max']
        B1000_max_list = affine_list['T1_max_list']

        fake_B_2D_tensor = torch.flip(visuals['fake_B'].squeeze(dim=1).transpose(0, 2), dims=[0, 1])
        real_A_2D_tensor = torch.flip(visuals['real_A'].squeeze(dim=1).transpose(0, 2), dims=[0, 1])
        real_B_2D_tensor = torch.flip(visuals['real_B'].squeeze(dim=1).transpose(0, 2), dims=[0, 1])

        if istrain:
            fake_B_nii_path = os.path.join(self.train_nii_dir,"epoch%3d_%3s"%(epoch,"fake_B"))
        else:
            fake_B_nii_path = os.path.join(self.test_nii_dir, "%3s%s" % ("fake_B",image_path[0][0:8]))

        # from second data beginning to cat
        if isinstance(self.real_T1_nii_data,type(None)):
            self.real_T1_nii_data = real_A_2D_tensor

        if isinstance(self.B0_nii_data, type(None)):
            self.B0_nii_data = fake_B_2D_tensor
            self.real_T2_nii_data = real_B_2D_tensor

        else:
            self.B0_nii_data = torch.cat([self.B0_nii_data,fake_B_2D_tensor],axis=2)
            self.real_T2_nii_data = torch.cat([self.real_T2_nii_data,real_B_2D_tensor],axis=2)


        self.run_save_nii+=1
        # 判断是不是该存储了
        if((self.run_save_nii-1) * batchsize == T1_total_slices):
            # 反归一化B0
            self.real_save_T2_nii_data = self.real_T2_nii_data * pred_B0_max_list[self.affine_index % affine_list["affine_num"]]
            self.save_B0_nii_data = self.B0_nii_data * pred_B0_max_list[self.affine_index % affine_list["affine_num"]]
            # print("kan xia T1nii zhang shen mo yang zi")
            # print(torch.tensor(self.T1_nii_data).size())
            # dict_T1 = self.quantitative.complete(self.T1_nii_data.detach().cpu().numpy(),self.real_T1_nii_data.detach().cpu().numpy(),image_path[0][0:5]+"T1",epoch,True)

            # 下采样
            # self.save_T1_nii_data = self.save_T1_nii_data.transpose(1, 2).unsqueeze(dim=0).unsqueeze(dim=0)
            # self.save_T1_nii_data = F.interpolate(self.save_T1_nii_data,size=size_dict["T1_size"][self.affine_index % affine_list["affine_num"]],mode="trilinear").squeeze(dim=0).squeeze(dim=0)
            self.save_B0_nii_data = self.save_B0_nii_data.transpose(1, 2).unsqueeze(dim=0).unsqueeze(dim=0)
            self.save_B0_nii_data = torch.floor(F.interpolate(self.save_B0_nii_data,size=size_dict["B0_size"][self.affine_index % affine_list["affine_num"]],mode="trilinear").squeeze(dim=0).squeeze(dim=0))
            # save_B0_mask = torch.gt(self.save_B0_nii_data, 2)
            # self.save_B0_nii_data = torch.mul(save_B0_mask,self.save_B0_nii_data)

            save_B0_mask = torch.load(self.mask_list[self.affine_index % affine_list["affine_num"]])
            self.save_B0_nii_data = torch.mul(self.save_B0_nii_data.detach().cpu(),save_B0_mask)

            T2_nii = nib.Nifti1Image(self.save_B0_nii_data.numpy(),affine_list["T2_affine_list"][self.affine_index % affine_list["affine_num"]])

            # 反归一化B1000
            self.real_save_T1_nii_data = self.real_T1_nii_data * B1000_max_list[
                self.affine_index % affine_list["affine_num"]]
            # 下采样
            self.real_save_T1_nii_data = self.real_save_T1_nii_data.transpose(1, 2).unsqueeze(dim=0).unsqueeze(dim=0)
            self.real_save_T1_nii_data = torch.floor(F.interpolate(self.real_save_T1_nii_data,size=size_dict["T1_size"][self.affine_index % affine_list["affine_num"]],mode="trilinear").squeeze(dim=0).squeeze(dim=0))
            self.real_save_T2_nii_data = self.real_save_T2_nii_data.transpose(1, 2).unsqueeze(dim=0).unsqueeze(dim=0)
            self.real_save_T2_nii_data = torch.floor(F.interpolate(self.real_save_T2_nii_data,size=size_dict["B0_size"][self.affine_index % affine_list["affine_num"]],mode="trilinear").squeeze(dim=0).squeeze(dim=0))

            if save_nii:
                real_T1_nii = nib.Nifti1Image(self.real_save_T1_nii_data.detach().cpu().numpy(),affine_list["T1_affine_list"][self.affine_index % affine_list["affine_num"]])
                real_B0_nii = nib.Nifti1Image(self.real_save_T2_nii_data.detach().cpu().numpy(),affine_list["T2_affine_list"][self.affine_index % affine_list["affine_num"]])

                nib.save(T2_nii,fake_B_nii_path)
            
                if self.istrain:
                    nib.save(real_B0_nii,os.path.join(self.train_nii_dir,"epoch%3d_%3s"%(epoch,"real_B")))
                else:
                    nib.save(real_B0_nii,os.path.join(self.test_nii_dir,"%3s_%3s"%("real_B",image_path[0][0:8])))
                    nib.save(real_T1_nii,os.path.join(self.test_nii_dir,"%3s_%3s"%("real_A",image_path[0][0:8])))
            # 计算相似度指标
            dict_B0 = self.quantitative.complete(self.real_save_T2_nii_data.detach().cpu().numpy(),
                                                 self.save_B0_nii_data.detach().cpu().numpy(),
                                                 image_path[0][0:8] + "B0", epoch)
            # dai xiu gai
            self.print_current_quantitative(dict_B0, image_path[0][0:8] + "B0", epoch)
            # zero
            self.run_save_nii = 1
            self.T1_nii_data = None
            self.B0_nii_data = None
            self.real_T1_nii_data = None
            self.real_T2_nii_data = None
            self.affine_index += 1
            # self.bool_fakeA = False
            # self.bool_fakeB = False


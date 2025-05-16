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

        test_mask_paths = os.path.join(self.public, "monkey_brain", "test_mask")
        test_smask_path = os.listdir(test_mask_paths)
        test_smask_path.sort()
        self.test_mask_list = []
        if not os.path.exists(os.path.join(self.public, "monkey_brain", "test_mask_pt")):
            os.mkdir(os.path.join(self.public, "monkey_brain", "test_mask_pt"))
        for mask_path in test_smask_path:
            torch.save(torch.tensor(nib.load(os.path.join(test_mask_paths, mask_path)).get_fdata()),
                       os.path.join(self.public, "monkey_brain", "test_mask_pt", mask_path[0:8] + ".pt"))
            self.test_mask_list.append(os.path.join(self.public, "monkey_brain", "test_mask_pt", mask_path[0:8] + ".pt"))

        #train_mask_paths = os.path.join(self.public, "monkey_brain", "train_mask")
        #train_smask_path = os.listdir(train_mask_paths)
        #train_smask_path.sort()
        #self.train_mask_list = []
        #if not os.path.exists(os.path.join(self.public, "monkey_brain", "train_mask_pt")):
        #    os.mkdir(os.path.join(self.public, "monkey_brain", "train_mask_pt"))
        #for mask_path in train_smask_path:
        #    torch.save(torch.tensor(nib.load(os.path.join(train_mask_paths, mask_path)).get_fdata()),
        #               os.path.join(self.public, "monkey_brain", "train_mask_pt", mask_path[0:8] + ".pt"))
        #    self.train_mask_list.append(os.path.join(self.public, "monkey_brain", "train_mask_pt", mask_path[0:8] + ".pt"))

        train_mask_paths = os.path.join(self.public, "monkey_brain", "test_mask")
        train_smask_path = os.listdir(train_mask_paths)
        train_smask_path.sort()
        self.train_mask_list = []
        if not os.path.exists(os.path.join(self.public, "monkey_brain", "test_mask_pt")):
            os.mkdir(os.path.join(self.public, "monkey_brain", "test_mask_pt"))
        for mask_path in train_smask_path:
            torch.save(torch.tensor(nib.load(os.path.join(train_mask_paths, mask_path)).get_fdata()),
                       os.path.join(self.public, "monkey_brain", "test_mask_pt", mask_path[0:8] + ".pt"))
            self.train_mask_list.append(os.path.join(self.public, "monkey_brain", "test_mask_pt", mask_path[0:8] + ".pt"))

        self.pretrain = pretrain
        if self.pretrain:
            self.results = "pretrain_results"
            self.train_img_dir = os.path.join(self.public,
                                              "monkey_brain/" + self.results + "/web/pretrain_images")
            self.train_nii_dir = os.path.join(self.public,
                                              "monkey_brain/" + self.results + "/web/pretrain_nii")
        else:
            self.results = "results"
            self.train_img_dir = os.path.join(self.public,
                                              "monkey_brain/" + self.results + "/web/images")
            self.train_nii_dir = os.path.join(self.public, "monkey_brain/" + self.results + "/web/nii")
        self.test_img_dir = os.path.join(self.public,
                                         "monkey_brain/" + self.results + "/web/test_images_human")
        self.test_nii_dir = os.path.join(self.public, "monkey_brain/" + self.results + "/web/test_nii_uwm")
        self.path = os.path.join(self.public, "monkey_brain")
        self.log_name = os.path.join(self.path, self.results,
                                     "loss_log"+".txt")
        self.test_log_name = os.path.join(self.path, self.results,
                                          "test_loss_log" + ".txt")
        self.train_quantitative_log_name = os.path.join(self.path, self.results,
                                                        "train_quantitative_log"+".txt")

        self.istrain = istrain
        self.run_save_nii = 1
        self.T1_nii_data = None
        self.B0_nii_data = None
        self.save_T1_nii_data = None
        self.save_B0_nii_data = None
        self.save_BB_nii_data = None
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
        # if not exists path, then create them.
        if not os.path.exists(os.path.join(self.path, self.results)):
            os.mkdir(os.path.join(self.path, self.results))
        if not os.path.exists(os.path.join(self.path, self.results, "web")):
            os.mkdir(os.path.join(self.path, self.results, "web"))
        if not os.path.exists(self.train_img_dir):
            os.mkdir(self.train_img_dir)
        if not os.path.exists(self.train_nii_dir):
            os.mkdir(self.train_nii_dir)
        if not os.path.exists(self.test_nii_dir):
            os.mkdir(self.test_nii_dir)
        if not os.path.exists(self.test_img_dir):
            os.mkdir(self.test_img_dir)

        if self.istrain:
            with open(self.log_name, "a") as log_file:
                # 返回以可读字符串表示的当地时间。
                now = time.strftime("%c")
                log_file.write('=======================Traing loss (%s) =============================\n' % now)
            with open(self.train_quantitative_log_name, "a") as train_quantitative_log_file:
                # 返回以可读字符串表示的当地时间。
                now = time.strftime("%c")
                train_quantitative_log_file.write(
                    '=======================train quantitative loss (%s) =============================\n' % now)
        else:
            with open(self.test_log_name, "a") as test_quantitative_log_file:
                # 返回以可读字符串表示的当地时间。
                now = time.strftime("%c")
                test_quantitative_log_file.write(
                    '=======================Test quantitative loss (%s) =============================\n' % now)
        # quantitative
        self.quantitative = quantitative(istrain,pretrain,slice_scope,max_epoch)

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        message = '(epoch:%d, iters:%d, time:%.3f,data:%.3f)' % (epoch, iters, t_comp, t_data)

        for k, v in losses.items():
            message += "%s:%.3f " % (k, v)
        print(message)

        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_current_quantitative(self, quantitative, image_path, epoch):
        if self.istrain:
            message = '(epoch:%s )' % (epoch)
            message += '(image_path:%s )' % (image_path)
        else:
            message = '(image_path:%s )' % (image_path)

        for k, v in quantitative.items():
            message += "%s:%s " % (k, v)
            # print(message)
        if self.istrain:
            with open(self.train_quantitative_log_name, "a") as train_quantitative_log_file:
                train_quantitative_log_file.write('%s\n' % message)
        else:
            with open(self.test_log_name, "a") as test_log_file:
                test_log_file.write('%s\n' % message)

    def save_as_png(self, visuals, image_path, epoch, istrain):
        for label, image in visuals.items():
            if istrain:
                img_path_T1 = os.path.join(self.train_img_dir, "epoch%3d_%3s%sT1_%s.png" % (
                epoch, label, image_path[0][0:8], image_path[0][-1]))
                img_path_B0 = os.path.join(self.train_img_dir, "epoch%3d_%3s%sB0_%s.png" % (
                epoch, label, image_path[0][0:8], image_path[0][-1]))
            else:
                img_path_T1 = os.path.join(self.test_img_dir,
                                           "%3s%sT1_%s.png" % (label, image_path[0][0:8], image_path[0][-1]))
                img_path_B0 = os.path.join(self.test_img_dir,
                                           "%3s%sB0_%s.png" % (label, image_path[0][0:8], image_path[0][-1]))
            image = image[0, :, :, :].squeeze(dim=0).squeeze(dim=0)

            # set print tensor is full, not shenglue...
            # torch.set_printoptions(profile="full")
            # print(image)
            imageio.imwrite(img_path_T1, (image * 255).detach().cpu().numpy())
            imageio.imwrite(img_path_B0, (image * 255).detach().cpu().numpy())

    def save_as_nii(self, visuals, image_path, epoch, batchsize, istrain, T1_slices, T2_slices, T1_dict, B0_dict,
                    affine_list, size_dict):
        # save as nii every all slices(every nii)
        # print(istrain)
        # get the affine and data of T1 and T2
        T1_total_slices = T1_slices
        T2_total_slices = T2_slices
        assert T1_total_slices == T2_total_slices
        pred_B0_max_list = affine_list['pred_B0_max']
        B1000_max_list = affine_list['T1_max_list']
        # flip the image to counter the effect of affine.
        # add a dim tensor, to prepare cat tensor be 3 dim and generate the nii image.

        # fake_B_2D_tensor = torch.flip(visuals['fake_B'].squeeze(dim=0).squeeze(dim=0).T,dims = [0,1]).unsqueeze(dim=2)
        # real_A_2D_tensor = torch.flip(visuals['real_A'].squeeze(dim=0).squeeze(dim=0).T,dims = [0,1]).unsqueeze(dim=2)
        # real_B_2D_tensor = torch.flip(visuals['real_B'].squeeze(dim=0).squeeze(dim=0).T,dims = [0,1]).unsqueeze(dim=2)

        # get tensor for prepare generate nii.

        # fake_A_2D_tensor = torch.flip(visuals['fake_A'].squeeze(dim=1).transpose(0, 2), dims=[0, 1])
        fake_B_2D_tensor = torch.flip(visuals['fake_B'].squeeze(dim=1).transpose(0, 2), dims=[0, 1])
        # fake_BB_2D_tensor = torch.flip(visuals['fake_BB'].squeeze(dim=1).transpose(0, 2), dims=[0, 1])
        real_A_2D_tensor = torch.flip(visuals['real_A'].squeeze(dim=1).transpose(0, 2), dims=[0, 1])
        real_B_2D_tensor = torch.flip(visuals['real_B'].squeeze(dim=1).transpose(0, 2), dims=[0, 1])

        if istrain:
            # if train then save every epoch.
            # fake_A_nii_path = os.path.join(self.train_nii_dir,"epoch%3d_%3s"%(epoch,"fake_A"))
            fake_B_nii_path = os.path.join(self.train_nii_dir, "epoch_%d_%s" % (epoch, image_path[0][0:8]))
            # fake_BB_nii_path = os.path.join(self.train_nii_dir, "epoch_%3d_%3s" % (epoch, image_path[0][0:8]))
        else:
            # if test then save all nii and names are redefined
            # fake_A_nii_path = os.path.join(self.test_nii_dir, "%3s%s" % ("fake_A",image_path[0][0:5]))
            fake_B_nii_path = os.path.join(self.test_nii_dir, "%3s%s" % ("fake_B", image_path[0][0:8]))
            # fake_BB_nii_path = os.path.join(self.test_nii_dir, "%3s%s" % ("fake_BB", image_path[0][0:8]))
        """
        # for reading down and minm of load_data
        for dict_index in range((self.run_save_nii-1)*batchsize,(self.run_save_nii-1)*batchsize+batchsize):
            self.T1_dict_list_down.append(T1_dict["down"][dict_index])
            self.T1_dict_list_minm.append(T1_dict["minm"][dict_index])
            self.B0_dict_list_down.append(B0_dict["down"][dict_index])
            self.B0_dict_list_minm.append(B0_dict["minm"][dict_index])
            #print(dict_index)
            #print(len(self.T1_dict_list_down))
        T1_tensor_down = torch.tensor(self.T1_dict_list_down).cuda()
        T1_tensor_minm = torch.tensor(self.T1_dict_list_minm).cuda()
        B0_tensor_down = torch.tensor(self.B0_dict_list_down).cuda()
        B0_tensor_minm = torch.tensor(self.B0_dict_list_minm).cuda()
        self.T1_dict_list_down.clear()
        self.T1_dict_list_minm.clear()
        self.B0_dict_list_down.clear()
        self.B0_dict_list_minm.clear()
        """

        # from second data beginning to cat
        if isinstance(self.real_T1_nii_data, type(None)):
            # self.T1_nii_data = fake_A_2D_tensor
            self.real_T1_nii_data = real_A_2D_tensor

        else:
            """
            # self.T1_nii_data = torch.cat([self.T1_nii_data,fake_A_2D_tensor],axis=2)
            # self.save_T1_nii_data = torch.cat([self.save_T1_nii_data,fake_A_2D_tensor * T1_tensor_down + T1_tensor_minm],axis=2)
            # self.real_save_T1_nii_data = torch.cat([self.real_save_T1_nii_data,real_A_2D_tensor * T1_tensor_down + T1_tensor_minm],axis=2)
            self.real_T1_nii_data = torch.cat([self.real_T1_nii_data,real_A_2D_tensor],axis=2)
            """

        if isinstance(self.B0_nii_data, type(None)):
            self.B0_nii_data = fake_B_2D_tensor
            # self.BB_nii_data = fake_BB_2D_tensor
            # print("归一化后的假B0", self.B0_nii_data.max())
            # 反归一化
            # self.real_save_T2_nii_data = real_B_2D_tensor * B0_tensor_down + B0_tensor_minm
            # self.save_B0_nii_data = fake_B_2D_tensor * B0_tensor_down + B0_tensor_minm

            self.real_T2_nii_data = real_B_2D_tensor
            # print("归一化后的真B0",self.real_T2_nii_data.max())
            # print("还原后的真B0",self.real_save_T2_nii_data.max())
            # print("还原后的假B0",self.save_B0_nii_data.max())
        else:
            self.B0_nii_data = torch.cat([self.B0_nii_data, fake_B_2D_tensor], axis=2)
            # self.BB_nii_data = torch.cat([self.BB_nii_data, fake_BB_2D_tensor], axis=2)
            # print("归一化后的假B0",self.B0_nii_data.max())
            # 反归一化
            # self.real_save_T2_nii_data = torch.cat([self.real_save_T2_nii_data,real_B_2D_tensor * B0_tensor_down + B0_tensor_minm],axis=2)
            # self.save_B0_nii_data = torch.cat([self.save_B0_nii_data, fake_B_2D_tensor * B0_tensor_down + B0_tensor_minm],axis=2)

            self.real_T2_nii_data = torch.cat([self.real_T2_nii_data, real_B_2D_tensor], axis=2)
            # print("归一化后的真B0", self.real_T2_nii_data.max())
            # print("还原后的真B0",self.real_save_T2_nii_data.max())
            # print("还原后的假B0",self.save_B0_nii_data.max())

        self.run_save_nii += 1
        # 判断是不是该存储了
        if ((self.run_save_nii - 1) * batchsize == T1_total_slices):
            # 反归一化B0
            self.real_save_T2_nii_data = self.real_T2_nii_data * pred_B0_max_list[
                self.affine_index % affine_list["affine_num"]]
            self.save_B0_nii_data = self.B0_nii_data * pred_B0_max_list[self.affine_index % affine_list["affine_num"]]
            # self.save_BB_nii_data = self.BB_nii_data * pred_B0_max_list[self.affine_index % affine_list["affine_num"]]
            # print("kan xia T1nii zhang shen mo yang zi")
            # print(torch.tensor(self.T1_nii_data).size())
            # dict_T1 = self.quantitative.complete(self.T1_nii_data.detach().cpu().numpy(),self.real_T1_nii_data.detach().cpu().numpy(),image_path[0][0:5]+"T1",epoch,True)
            """
            dict_T2 = self.quantitative.complete(self.B0_nii_data.detach().cpu().numpy(),self.real_T2_nii_data.detach().cpu().numpy(),image_path[0][0:6]+"B0",epoch,True)

            # dict_B1000 = self.quantitative.complete(self.real_save_T1_nii_data .detach().cpu().numpy(),
            #                                      self.save_T1_nii_data.detach().cpu().numpy(),
            #                                      image_path[0][0:6] + "T1", epoch,False)
            dict_B0 = self.quantitative.complete(self.real_save_T2_nii_data.detach().cpu().numpy(),
                                                self.save_B0_nii_data.detach().cpu().numpy(),
                                                 image_path[0][0:6] + "B0", epoch,False)
            # dai xiu gai

            # self.print_current_quantitative(dict_T1,image_path[0][0:6]+"T1",epoch)
            self.print_current_quantitative(dict_T2,image_path[0][0:6]+"B0",epoch)
            # self.print_current_quantitative(dict_B1000,image_path[0][0:6]+"B1000",epoch)
            self.print_current_quantitative(dict_B0,image_path[0][0:6]+"B0",epoch)
            """
            # 下采样
            # self.save_T1_nii_data = self.save_T1_nii_data.transpose(1, 2).unsqueeze(dim=0).unsqueeze(dim=0)
            # self.save_T1_nii_data = F.interpolate(self.save_T1_nii_data,size=size_dict["T1_size"][self.affine_index % affine_list["affine_num"]],mode="trilinear").squeeze(dim=0).squeeze(dim=0)
            self.save_B0_nii_data = self.save_B0_nii_data.transpose(1, 2).unsqueeze(dim=0).unsqueeze(dim=0)
            self.save_B0_nii_data = torch.floor(F.interpolate(self.save_B0_nii_data, size=size_dict["B0_size"][
                self.affine_index % affine_list["affine_num"]], mode="trilinear").squeeze(dim=0).squeeze(dim=0))
                
            #self.save_BB_nii_data = self.save_BB_nii_data.transpose(1, 2).unsqueeze(dim=0).unsqueeze(dim=0)
            #self.save_BB_nii_data = torch.floor(F.interpolate(self.save_BB_nii_data, size=size_dict["B0_size"][
            #    self.affine_index % affine_list["affine_num"]], mode="trilinear").squeeze(dim=0).squeeze(dim=0))
            # save_B0_mask = torch.gt(self.save_B0_nii_data, 2)
            # self.save_B0_nii_data = torch.mul(save_B0_mask,self.save_B0_nii_data)
            if not self.istrain:
                save_B0_mask = torch.load(self.test_mask_list[self.affine_index % affine_list["affine_num"]])
                self.save_B0_nii_data = torch.mul(self.save_B0_nii_data.detach().cpu(), save_B0_mask)
            # T1_nii = nib.Nifti1Image(self.save_T1_nii_data.detach().cpu().numpy(),affine_list["T1_affine_list"][self.affine_index % affine_list["affine_num"]])
                T2_nii = nib.Nifti1Image(self.save_B0_nii_data.numpy(),
                                     affine_list["T2_affine_list"][self.affine_index % affine_list["affine_num"]])
                #self.save_BB_nii_data = torch.mul(self.save_BB_nii_data.detach().cpu(), save_B0_mask)
                #BB_nii = nib.Nifti1Image(self.save_BB_nii_data.numpy(),
                #                     affine_list["T2_affine_list"][self.affine_index % affine_list["affine_num"]])
            elif self.istrain and not self.pretrain:
                save_B0_mask = torch.load(self.test_mask_list[self.affine_index % affine_list["affine_num"]])
                self.save_B0_nii_data = torch.mul(self.save_B0_nii_data.detach().cpu(), save_B0_mask)
                T2_nii = nib.Nifti1Image(self.save_B0_nii_data.numpy(),
                                     affine_list["T2_affine_list"][self.affine_index % affine_list["affine_num"]])
                #self.save_BB_nii_data = torch.mul(self.save_BB_nii_data.detach().cpu(), save_B0_mask)
                #BB_nii = nib.Nifti1Image(self.save_BB_nii_data.numpy(),
                #                     affine_list["T2_affine_list"][self.affine_index % affine_list["affine_num"]])               
                
            else:
                # self.save_BB_nii_data = torch.mul(self.save_BB_nii_data.detach().cpu(), save_B0_mask)
                #BB_nii = nib.Nifti1Image(self.save_BB_nii_data.detach().cpu().numpy(),
                #                     affine_list["T2_affine_list"][self.affine_index % affine_list["affine_num"]])
                T2_nii = nib.Nifti1Image(self.save_B0_nii_data.detach().cpu().numpy(),
                                     affine_list["T2_affine_list"][self.affine_index % affine_list["affine_num"]])
            # 反归一化B1000
            self.real_save_T1_nii_data = self.real_T1_nii_data * B1000_max_list[
                self.affine_index % affine_list["affine_num"]]
            # 下采样
            self.real_save_T1_nii_data = self.real_save_T1_nii_data.transpose(1, 2).unsqueeze(dim=0).unsqueeze(dim=0)
            self.real_save_T1_nii_data = torch.floor(F.interpolate(self.real_save_T1_nii_data,
                                                                   size=size_dict["T1_size"][
                                                                       self.affine_index % affine_list["affine_num"]],
                                                                   mode="trilinear").squeeze(dim=0).squeeze(dim=0))
            self.real_save_T2_nii_data = self.real_save_T2_nii_data.transpose(1, 2).unsqueeze(dim=0).unsqueeze(dim=0)
            self.real_save_T2_nii_data = torch.floor(F.interpolate(self.real_save_T2_nii_data,
                                                                   size=size_dict["B0_size"][
                                                                       self.affine_index % affine_list["affine_num"]],
                                                                   mode="trilinear").squeeze(dim=0).squeeze(dim=0))
            real_T1_nii = nib.Nifti1Image(self.real_save_T1_nii_data.detach().cpu().numpy(),
                                          affine_list["T1_affine_list"][self.affine_index % affine_list["affine_num"]])
            real_B0_nii = nib.Nifti1Image(self.real_save_T2_nii_data.detach().cpu().numpy(),
                                          affine_list["T2_affine_list"][self.affine_index % affine_list["affine_num"]])

            # nib.save(T1_nii,fake_A_nii_path)
            nib.save(T2_nii, fake_B_nii_path)
            # nib.save(BB_nii, fake_BB_nii_path)

            if self.istrain:
                # nib.save(real_T1_nii,os.path.join(self.train_nii_dir,"epoch%3d_%3s"%(epoch,"real_A")))
                # nib.save(real_B0_nii, os.path.join(self.train_nii_dir, "epoch_%3d_%3s" % (epoch, image_path[0][0:8])))
                pass
            else:
                nib.save(real_B0_nii, os.path.join(self.test_nii_dir, "%3s_%3s" % ("real_B", image_path[0][0:8])))
                nib.save(real_T1_nii, os.path.join(self.test_nii_dir, "%3s_%3s" % ("real_A", image_path[0][0:8])))

            self.B0_nii_data = self.B0_nii_data.transpose(1, 2).unsqueeze(dim=0).unsqueeze(dim=0)
            self.B0_nii_data = F.interpolate(self.B0_nii_data, size=size_dict["T1_size"][
                self.affine_index % affine_list["affine_num"]], mode="trilinear").squeeze(dim=0).squeeze(dim=0)
            self.real_T2_nii_data = self.real_T2_nii_data.transpose(1, 2).unsqueeze(dim=0).unsqueeze(dim=0)
            self.real_T2_nii_data = F.interpolate(self.real_T2_nii_data, size=size_dict["B0_size"][
                self.affine_index % affine_list["affine_num"]], mode="trilinear").squeeze(dim=0).squeeze(dim=0)
            # 计算相似度指标
            """
            dict_T2 = self.quantitative.complete(self.B0_nii_data.detach().cpu().numpy(),
                                                 self.real_T2_nii_data.detach().cpu().numpy(),
                                                 image_path[0][0:8] + "B0", epoch)
            """
            # dict_B1000 = self.quantitative.complete(self.real_save_T1_nii_data .detach().cpu().numpy(),
            #                                      self.save_T1_nii_data.detach().cpu().numpy(),
            #                                      image_path[0][0:6] + "T1", epoch,False)
            dict_B0 = self.quantitative.complete(self.real_save_T2_nii_data.detach().cpu().numpy(),
                                                 self.save_B0_nii_data.detach().cpu().numpy(),
                                                 image_path[0][0:8] + "B0", epoch)
            
            # dict_BB = self.quantitative.complete(self.real_save_T2_nii_data.detach().cpu().numpy(),
            #                                   self.save_BB_nii_data.detach().cpu().numpy(),
            #                                    image_path[0][0:8] + "B0", epoch, 'fake_BB')
            # dai xiu gai

            # self.print_current_quantitative(dict_T1,image_path[0][0:6]+"T1",epoch)
            # self.print_current_quantitative(dict_T2, image_path[0][0:8] + "B0", epoch)
            # self.print_current_quantitative(dict_B1000,image_path[0][0:6]+"B1000",epoch)
            self.print_current_quantitative(dict_B0, image_path[0][0:8] + "B0", epoch)
            #self.print_current_quantitative(dict_BB, image_path[0][0:8] + "B0", epoch)

            # zero
            self.run_save_nii = 1
            self.T1_nii_data = None
            self.B0_nii_data = None
            self.real_T1_nii_data = None
            self.real_T2_nii_data = None
            self.affine_index += 1
            # self.bool_fakeA = False
            # self.bool_fakeB = False


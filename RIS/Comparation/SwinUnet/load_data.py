import nibabel as nib
import torch
import os
import torch.utils.data
import numpy as np
import torch.nn.functional as F

# 加载脑数据,切片儿
# 返回列表，+
#  + ++++++++
#  存储所有数据。
# 切片函数。

class Dataset():
    def __init__(self,istrain,ispretrain,cfg):
        # MLP Regreesion model
        self.next_B0 = 0
        # 获得路径
        # self.path = os.path.join(os.getcwd(),"monkey_brain")
        self.path = cfg.data_path
        self.istrain = istrain
        self.ispretrain = ispretrain
        if self.istrain:
            if self.ispretrain:
                self.T1_paths = os.path.join(self.path, "pretrain_B1000_data")
                self.T2_paths = os.path.join(self.path, "pretrain_B0_data")
            else:
                self.T1_paths = os.path.join(self.path, cfg.train_dataset)
                self.T2_paths = os.path.join(self.path, cfg.train_target_dataset)
                # self.T1_paths = os.path.join(self.path, "test_t1")
                # self.T2_paths = os.path.join(self.path, "test_b0")
        else:
            if self.ispretrain:
                self.T1_paths = os.path.join(self.path, cfg.val_dataset)
                self.T2_paths = os.path.join(self.path, cfg.val_target_dataset)
            else:
                self.T1_paths = os.path.join(self.path, cfg.test_dataset)
                self.T2_paths = os.path.join(self.path, cfg.test_target_dataset)

        self.T1_path = os.listdir(self.T1_paths)

        self.T1_nums = len(self.T1_path)
        print("B1000图像的个数为: " + str(self.T1_nums))
        self.T2_path = os.listdir(self.T2_paths)
        self.T2_nums = len(self.T2_path)
        print("B0图像的个数为: " + str(self.T2_nums))

        self.T1_path.sort()
        self.T2_path.sort()

        # 改这里读取不同的切片数
        # set the begin slice and the end slice
        self.begin_slice = 1
        self.end_slice = 256

        # 列表用于存储所有nii
        self.T1_nii_list = []
        self.T2_nii_list = []
        self.T1_size_list = []
        self.T2_size_list = []
        self.T1_nib = None
        self.T2_nib = None
        self.T1_affine = None
        self.T2_affine = None
        self.T1_affine_list = []
        self.T2_affine_list = []

        self.T1_nii_max_list = []
        self.pred_B0_max_list = []
        if not os.path.exists(os.path.join(self.path, "data_pt")):
            os.mkdir(os.path.join(self.path, "data_pt"))

        if not os.path.exists(os.path.join(self.path, "test_data_pt")):
            os.mkdir(os.path.join(self.path, "test_data_pt"))

        if not os.path.exists(os.path.join(self.path, "val_data_pt")):
            os.mkdir(os.path.join(self.path, "val_data_pt"))

        # 把nii数据转tensor存到列表里
        for T1 in self.T1_path:
            self.T1 = T1
            self.allpath_T1 = os.path.join(self.T1_paths, self.T1)
            self.T1_nib = nib.load(self.allpath_T1)
            self.T1_nii = torch.tensor(self.T1_nib.get_fdata())
            self.T1_affine = self.T1_nib.affine

            self.T1_size_list.append(self.T1_nii.size())
            # 记住每个B1000的max，为后面计算B0 max作铺垫。
            self.T1_nii_max_list.append(self.T1_nii.max())

            #归一化T1
            #self.T1_nii = (self.T1_nii - self.T1_nii.min()) / (self.T1_nii.max()-self.T1_nii.min())
            # self.T1_nii = self.T1_nii/self.T1_nii.max()
            # 归一化B1000
            self.T1_nii = (self.T1_nii - self.T1_nii.min()) / (self.T1_nii.max()-self.T1_nii.min())
            # 上采样
            self.T1_nii = self.T1_nii.unsqueeze(dim=0).unsqueeze(dim=0)
            self.T1_nii = F.interpolate(self.T1_nii,size=(256,256,256),mode="trilinear").squeeze(dim=0).squeeze(dim=0)

            if self.istrain:
            # save to the list
                if not os.path.exists(os.path.join(self.path, "data_pt",T1[0:6]+"T1.pt")):
                    torch.save(self.T1_nii,os.path.join(self.path,"data_pt",T1[0:6]+"T1.pt"))
            else:
                if self.ispretrain:
                    if not os.path.exists(os.path.join(self.path, "val_data_pt", T1[0:6] + "T1.pt")):
                        torch.save(self.T1_nii, os.path.join(self.path, "val_data_pt", T1[0:6] + "T1.pt"))
                else:
                    if not os.path.exists(os.path.join(self.path, "test_data_pt", T1[0:6] + "T1.pt")):
                        torch.save(self.T1_nii, os.path.join(self.path, "test_data_pt", T1[0:6] + "T1.pt"))
            # self.T1_nii_list.append(self.T1_nii)
            self.T1_nii_list.append(T1[0:6]+"T1.pt")
            self.T1_affine_list.append(self.T1_affine)
        # A_path = self.T1_path[item % self.T1_nums]

        # 切片数
        # self.T1_slices = self.T1_nii.size()[2] - 65
        self.T1_slices = self.end_slice - self.begin_slice + 1

        j = 0
        for T2 in self.T2_path:
            self.T2 = T2
            self.allpath_T2 = os.path.join(self.T2_paths, self.T2)
            self.T2_nib = nib.load(self.allpath_T2)
            self.T2_nii = torch.tensor(self.T2_nib.get_fdata()).to('cuda:0')
            self.T2_affine = self.T2_nib.affine

            self.T2_size_list.append(self.T2_nii.size())
            self.pred_B0_max_list.append(self.T2_nii.max())
            #归一化T2
            # self.T2_nii = (self.T2_nii - self.T2_nii.min()) / (self.T2_nii.max()-self.T2_nii.min())
            #devide the max normlization
            self.T2_nii = self.T2_nii / self.T2_nii.max()

            # 归一化B0
            # self.T2_nii = self.T2_nii / self.pred_B0_max_list[j]
            j += 1
            self.T2_nii = self.T2_nii.unsqueeze(dim=0).unsqueeze(dim=0)
            self.T2_nii = F.interpolate(self.T2_nii, size=(256, 256, 256), mode="trilinear").squeeze(dim=0).squeeze(
                dim=0)
            if self.istrain:
              if not os.path.exists(os.path.join(self.path, "data_pt", T2[0:6] + "B0.pt")):
                torch.save(self.T2_nii, os.path.join(self.path, "data_pt", T2[0:6] + "B0.pt"))
            else:
                if self.ispretrain:
                    if not os.path.exists(os.path.join(self.path, "val_data_pt", T2[0:6] + "B0.pt")):
                        torch.save(self.T2_nii, os.path.join(self.path, "val_data_pt", T2[0:6] + "B0.pt"))
                else:
                    if not os.path.exists(os.path.join(self.path, "test_data_pt", T2[0:6] + "B0.pt")):
                        torch.save(self.T2_nii, os.path.join(self.path, "test_data_pt", T2[0:6] + "B0.pt"))
            self.T2_nii_list.append(T2[0:6] + "B0.pt")
            # self.T2_nii_list.append(self.T2_nii)
            self.T2_affine_list.append(self.T2_affine)

            # B_path = self.T2_path[item % len(self.T2_path)]
        # self.T2_slices = self.T2_nii.size()[2] - 65
        self.T2_slices = self.end_slice - self.begin_slice + 1
        print("T1图像切片个数为:" + str(self.T1_nums * self.T2_slices))
        print("B0图像切片个数为:" + str(self.T2_nums * self.T2_slices))

        # 定义切片的列表
        self.T1_list = []
        self.T2_list = []
        self.i = 0

        # huifu gui yi hua zhiqian
        # unnormlization
        """
        self.T1_down = None
        self.T1_minm = None
        self.B0_down = None
        self.B0_minm = None
        """
        self.T1_dict = {}
        self.B0_dict = {}
    def slice_flip_data(self,nii,slices):
        slice_list = []
        # down_tensor = []
        # minm_tensor = []
        # zan shi qie pian cong 75-95
        for public_slice in range(self.begin_slice-1, self.begin_slice -1 + slices):
            # print(nii.size())
            one_slice = nii[:, public_slice ,:]
            # 归一化 待改正；已改正，如果是max=min=0不做归一化。
            # 如果max=min，换种方式归一化，直接除以最大值。
            """
            if(one_slice.max() == one_slice.min() and one_slice.max()==0):
                down = 1
                minm = 0
            elif(one_slice.max() == one_slice.min()):
                down = one_slice.max()
                minm = down
                one_slice = one_slice/(one_slice.max())
            else:
                down = (one_slice.max()-one_slice.min())
                minm = one_slice.min()
                one_slice = (one_slice - one_slice.min()) / ((one_slice.max()-one_slice.min()))
            """
            new_slice = torch.flip(one_slice.T, dims=[0, 1]).unsqueeze(dim=0)
            # new_slice = one_slice.T
            slice_list.append(new_slice)

            # down_tensor.append(down)
            # minm_tensor.append(minm)

        # return {"slice_list":slice_list,"down":down_tensor,"minm":minm_tensor}
        return {"slice_list": slice_list}
        # return {"slice_list": slice_list}
    def __getitem__(self, item):
        #print(item)
        # 当迭代次数正好到切片个数(去噪后 )的时候，更新切片列表。
        if (item%self.T1_slices) == 0:

            # 清除T1，T2切片的列表，新存入一批
            self.T1_list.clear()
            self.T2_list.clear()

            if self.istrain:
                self.T1_dict = self.slice_flip_data(torch.load(os.path.join(self.path, "data_pt", self.T1_nii_list[self.i])), self.T1_slices)
                self.B0_dict = self.slice_flip_data(torch.load(os.path.join(self.path, "data_pt", self.T2_nii_list[self.i])), self.T2_slices)
            else:
                if self.ispretrain:
                    self.T1_dict = self.slice_flip_data(
                        torch.load(os.path.join(self.path, "val_data_pt", self.T1_nii_list[self.i])), self.T1_slices)
                    self.B0_dict = self.slice_flip_data(
                        torch.load(os.path.join(self.path, "val_data_pt", self.T2_nii_list[self.i])), self.T2_slices)
                else:
                    self.T1_dict = self.slice_flip_data(torch.load(os.path.join(self.path, "test_data_pt", self.T1_nii_list[self.i])), self.T1_slices)
                    self.B0_dict = self.slice_flip_data(torch.load(os.path.join(self.path, "test_data_pt", self.T2_nii_list[self.i])), self.T2_slices)

            self.T1_list = self.T1_dict["slice_list"]
            self.T2_list = self.B0_dict["slice_list"]
            """
            self.T1_down = self.T1_dict["down"]
            self.T1_minm = self.T1_dict["minm"]
            self.B0_down = self.B0_dict["down"]
            self.B0_minm = self.B0_dict["minm"]
            """

            self.i = (self.i+1)%self.T1_nums
        if item % self.T1_slices == 0 and item != 0:
            self.next_B0 +=1
        # A_path = os.path.join(self.T1_paths, self.T1_path[item % self.T1_nums]) + str(item%self.T1_nums)
        # B_path = os.path.join(self.T2_paths, self.T2_path[item % self.T2_nums]) + str(item%self.T2_nums)
        A_path = self.T1_path[self.next_B0 % self.T1_nums] + str(item % self.T1_slices) #'116217_T1w_brain.nii.gz0'
        B_path = self.T2_path[self.next_B0 % self.T2_nums] + str(item % self.T2_slices) #'116217_T2w_brain.nii.gz0'

        """
        return {'A': self.T1_list[item%(self.T1_slices)], 'B': self.T2_list[item%(self.T2_slices)],
                'A_paths': A_path, 'B_paths': B_path,"T1_down":self.T1_down,"T1_minm":self.T1_minm,
                "B0_down":self.B0_down,"B0_minm":self.B0_minm
                }
        """
        return {'A': self.T1_list[item%(self.T1_slices)], 'B': self.T2_list[item%(self.T2_slices)],
                'A_paths': A_path, 'B_paths': B_path
                }
        """
        return {'A': self.T1_list[item % (self.T1_slices)], 'B': self.T2_list[item % (self.T2_slices)],
                'A_paths': A_path, 'B_paths': B_path}
        """
    # return T1_list,T2_list
    def __len__(self):
        return self.T1_nums * self.T1_slices
    def load_data(self):
        return self
    def get_T1_slices(self):
        return self.T1_slices
    def get_T2_slices(self):
        return self.T2_slices
    def get_slices(self):
        return {"slice3":self.T1_nii.size()[1],"slice4":self.T1_nii.size()[0]}
    def get_affine_list(self):
        return {"T1_affine_list":self.T1_affine_list,"T2_affine_list":self.T2_affine_list,"affine_num":len(self.T1_affine_list),'pred_B0_max':self.pred_B0_max_list,
                "T1_max_list":self.T1_nii_max_list}
    def get_slice_scope(self):
        return{"begin_slice":str(self.begin_slice),"end_slice":str(self.end_slice)}
    def get_size(self):
        return{"T1_size":self.T1_size_list,"B0_size":self.T2_size_list}
class MyDataLoader():
    def __init__(self,istrain,ispretrain,cfg):
        self.batchsize = cfg.batchsize
        self.istrain = istrain
        self.ispretrain = ispretrain
        self.dataset = Dataset(istrain,ispretrain=ispretrain,cfg=cfg)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size = self.batchsize,
            shuffle = False
        )
    def __len__(self):
        return len(self.dataloader)
    
    def load_data(self):
        return self
    def get_dataset(self):
        return self.dataset
    def get_batchsize(self):
        return self.batchsize
    def set_epoch(self,epoch):
        self.epoch = epoch
    def __iter__(self):
        """
        self.dataset = Dataset(self.istrain, self.epoch)
        self.batchsize = 35
        print("当前epoch为: "+str(self.epoch))
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batchsize,
            shuffle=False
        )
        """
        for i,data in enumerate(self.dataloader):
            yield data

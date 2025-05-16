# from load_data getting data and into the Cyclegan model, finnally outputting outcome of 3D tensor.
from load_data import MyDataLoader
from resvit_one import ResViT_model
from save_img import save_img
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
    model = ResViT_model(istrain,ispretrain)
    model.setup(read_dataset.get_slice_scope())
    save_img = save_img(istrain,ispretrain,read_dataset.get_slice_scope(),test_epoch)

    save_nii_every_slices = dataset.get_dataset().get_T1_slices()

    for i,data in enumerate(dataset):
        model.set_input(data)
        model.test()
        image_path = model.get_image_paths()  # for example '114217_T1w_brain.nii.gz0'
        # save_img.save_as_png(model.get_current_visuals(),image_path,200,False)
        save_img.save_as_nii(model.get_current_visuals(),image_path,80,batchsize,False,read_dataset.get_T1_slices(),read_dataset.get_T2_slices(),read_dataset.T1_dict,read_dataset.B0_dict,read_dataset.get_affine_list(),read_dataset.get_size(),True)
    save_img.quantitative.close_file()
        # loss cannot be completed, so we can complete ssim psnr and mse to replace it.

        # losses = model.get_current_losses()
        # save_img.print_current_losses(image_path,losses)

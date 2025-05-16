from load_data import Dataset,MyDataLoader
from CycleGAN import CycleGANModel
from save_img import save_img
import time
from matplotlib import pyplot as plt
if __name__ == '__main__':
    plt.ion()
    istrain = True
    ispretrain = False
    #dataset = Dataset()
    #dataset.load_data()
    dataset = MyDataLoader(istrain,ispretrain)
    val_dataset = MyDataLoader(False,True)
    read_dataset = dataset.get_dataset()
    real_val_dataset = val_dataset.get_dataset()
    batchsize = dataset.get_batchsize()
    dataset.load_data()
    print("已加载数据...")
    dataset_size = len(dataset)
    model = CycleGANModel(istrain,ispretrain)

    # continue train , last train is 50 epochs, we tend to continue train about 150 epochs
    # continue to train, from 195th epoch beginning
    # model.set_continue_train(istrain)

    model.setup(read_dataset.get_slice_scope())

    save_nii_every_slices = dataset.get_dataset().get_T1_slices()

    total_iters = 0
    epoch_count = 81
    n_epochs_decay = 85
    n_epochs = 0
    is_save_nii = True
    train_save_img = save_img(istrain, ispretrain, read_dataset.get_slice_scope(), n_epochs_decay + n_epochs)
    val_save_img = save_img(False, True, read_dataset.get_slice_scope(), n_epochs_decay + n_epochs)
    for epoch in range(epoch_count,n_epochs_decay + n_epochs + 1):
        dataset.set_epoch(epoch)
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        model.set_now_epoch(epoch)
        model.update_learning_rate()
        save_nii = False

        for i,data in enumerate(dataset):
            if i % 7650 == 0:
                save_nii = True
            iter_start_time = time.time()
            if total_iters % 100 == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += 1
            epoch_iter += 1
            # print(data)
            model.set_input(data)
            # model.optimize_parameters(read_dataset.get_slices())
            model.optimize_parameters()
            # if total_iters %40000 == 0:
            #     image_path = model.get_image_paths()
            #     save_img.save_as_png(model.get_current_visuals(),image_path,epoch,True)
            if total_iters %400 == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / 1
                train_save_img.print_current_losses(epoch,epoch_iter,losses,t_comp,t_data)

            # if total_iters % save_nii_every_slices * 5 == 0:
            #     if is_save_nii ==True:
            #         is_save_nii = False
            #     else:
            #         is_save_nii = True
            # if is_save_nii == True:
            image_path = model.get_image_paths()

            train_save_img.save_as_nii(model.get_current_visuals(),image_path,epoch,batchsize,True,read_dataset.get_T1_slices(),read_dataset.get_T2_slices(),read_dataset.T1_dict,read_dataset.B0_dict,read_dataset.get_affine_list(),read_dataset.get_size(),save_nii)
            iter_data_time = time.time()
        print('saving the latest model (epoch %d,total_iters %d)' % (epoch, total_iters))
        save_suffix = 'iter_%d' % total_iters if False else 'latest'
        model.save_networks(save_suffix, read_dataset.get_slice_scope())

        print('saving the model at the end of epoch %d,iters %d'%(epoch,total_iters))
        model.save_networks('latest',read_dataset.get_slice_scope())
        model.save_networks(epoch,read_dataset.get_slice_scope())

        print('End of epoch %d / %d \t Time Taken: %d sec' %(epoch,n_epochs_decay+n_epochs,time.time()-epoch_start_time))
        print("val\n")
        for i,data in enumerate(val_dataset):
            model.set_input(data)
            test_image_path = model.get_image_paths()
            model.test(real_val_dataset.get_slices())
            val_save_img.save_as_nii(model.get_current_visuals(), test_image_path, epoch, batchsize, False,
                             real_val_dataset.get_T1_slices(), real_val_dataset.get_T2_slices(), real_val_dataset.T1_dict,
                             real_val_dataset.B0_dict, real_val_dataset.get_affine_list(), real_val_dataset.get_size(),False)
    plt.ioff()
    val_save_img.quantitative.close_file()
    train_save_img.quantitative.close_file()

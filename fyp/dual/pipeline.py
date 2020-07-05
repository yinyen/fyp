import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from pytorch.param_helper import create_dir, dump_config
from pytorch.data_helper import initialize_dataset
from pytorch.model_helper import select_model, select_optimizer, select_scheduler
from pytorch.custom_loss import my_loss, my_loss_mse, myCrossEntropyLoss
from pytorch.dual_data_helper import create_dual_label_df, split_dual_df, initialize_dual_gen

from dual.training import train, validate, PerformanceLog


class DualPipeline():
    def __init__(self, gpu_device = 0, **kwargs):
        train_name = kwargs.get("train_name")
        main_model_dir = kwargs.get("main_model_dir")
        train_name_updated = create_dir(train_name, main_model_dir)
        self.model_path = f"{main_model_dir}/{train_name_updated}"
        self.log_path = f"{main_model_dir}/{train_name_updated}/log.csv"

        print("=====================================")
        print("SAVING MODEL TO >>>", self.model_path)
        print("=====================================")

        criterion = self.init_loss(**kwargs)
        train_loader, val_loader = self.init_dataset(**kwargs)
        model, optimizer, scheduler = self.init_model(**kwargs)
        if kwargs.get("retrain"):
            self.init_retrain(model, **kwargs)

        self.dump_config(self.model_path, kwargs)
        self.train(train_loader, val_loader, model, criterion, optimizer, scheduler, **kwargs)

    def init_loss(self, loss_type, **kwargs):
        # initialize loss and benchmark 
        if loss_type == "cross_entropy":
            criterion = nn.CrossEntropyLoss().cuda()
        elif loss_type == "l1":
            criterion = nn.L1Loss().cuda()
        elif loss_type == "cross_entropy_weight":
            criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1000/20649, 1000/1956, 1000/4235, 1000/700, 1000/568])).cuda()
        elif loss_type == "my_loss":
            criterion = my_loss
        elif loss_type == "mse":
            criterion = my_loss_mse
        elif loss_type == "my_cross_entropy":
            criterion = myCrossEntropyLoss
        elif loss_type == "smoothl1loss":
            criterion = nn.SmoothL1Loss().cuda()

        cudnn.benchmark = True
        return criterion

    def init_dataset(self, main_data_dir, batch_size, size, workers, reweight_sample = 1, reweight_sample_factor = 2, **kwargs):
        # initialize dataset generators
        dual_df = create_dual_label_df(main_data_dir = "../all_train_300", train_dir_list = ["full_train", "val"])
        d1, d_train = split_dual_df(dual_df, p = 0.05, seed = 321) # use 20k*0.05 = 1k samples for training
        d1_, d_val = split_dual_df(d1, p = 0.05, seed = 321) # use 20k*0.05 = 1k samples for val
        print("Train:", d_train.shape, "Val:", d_val.shape)
        train_gen, val_gen = initialize_dual_gen(d_train, d_val, size, batch_size, reweight_sample, reweight_sample_factor, workers)
        return train_gen, val_gen

    def init_model(self, model_type, optimizer_type, opt_kwargs, scheduler_type, scheduler_kwargs, model_kwargs = {}, **kwargs):
        # create model
        model = select_model(model_type = model_type, model_kwargs = model_kwargs)
        optimizer = select_optimizer(type = optimizer_type, model = model, kwargs = opt_kwargs)
        scheduler = select_scheduler(optimizer, scheduler_type = scheduler_type, scheduler_kwargs = scheduler_kwargs)
        return model, optimizer, scheduler

    def init_retrain(self, model, premodel_path, **kwargs):
        # premodel_path = "./torch_models/xception_d400_force_v1_v09/best_avg_acc_model.pth"
        model.load_state_dict(torch.load(premodel_path))
        return model
    
    def train(self, train_loader, val_loader, model, criterion, optimizer, scheduler, epochs, batch_multiplier = 1, qk_patience = 10, **kwargs):
        
        model_path, log_path = self.model_path, self.log_path
        plog = PerformanceLog()

        # iterate training
        best_loss = float('inf')
        best_qk = 0
        pt = qk_patience        
        for epoch in range(epochs):
            print('Epoch [%d/%d]' %(epoch+1, epochs))

            # train for one epoch
            train_log = train(train_loader, model, criterion, optimizer, batch_multiplier = batch_multiplier)

            # evaluate on validation set
            val_log = validate(val_loader, model, criterion)

            # change learning rate according to scheduler policy
            scheduler.step()

            # print and save performance logs
            perf_logs = []
            perf_logs.append(f'train: ' + 'loss %.4f - acc, avg_acc, qk (%.4f, %.4f, %.4f)' % (train_log['loss'], train_log['acc'], train_log['avg_acc'], train_log['qk']))
            perf_logs.append(f'validation: ' + 'loss %.4f - acc, avg_acc, qk (%.4f, %.4f, %.4f)' % (val_log['loss'], val_log['acc'], val_log['avg_acc'], val_log['qk']))
            print("Log:", self.log_path)
            print(" -- ".join(perf_logs))
            print("LearningRate:", scheduler.get_last_lr())
            plog.append(epoch = epoch, lr = scheduler.get_last_lr()[0], train_log = train_log, val_log = val_log, test_log = val_log)
            plog.save(log_path) # save logs as csv

            # save best model
            if val_log['loss'] < best_loss:
                torch.save(model.state_dict(), f'{model_path}/model.pth')
                best_loss = val_log['loss']
                print("=> saved best model by best loss", best_loss)

            if val_log['qk'] > best_qk + 0.5:
                torch.save(model.state_dict(), f'{model_path}/best_qk_model.pth')
                best_qk = val_log['qk']
                print("=> saved best model by best qk:", best_qk)
                pt = qk_patience

            pt -= 1            
            if pt < 0: 
                print("Stop criteria met. QK did not improve by more than 0.5 of previous best ({:.4f}) after {} epochs".format(best_qk, qk_patience))
                break


    def dump_config(self, model_path, kwargs):
        output_path = os.path.join(model_path, "config.yaml")
        dump_config(output_path, kwargs)
            

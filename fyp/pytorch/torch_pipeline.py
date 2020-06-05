import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from pytorch.train_helper import train, validate
from pytorch.param_helper import import_config, dump_config, create_dir
from pytorch.data_helper import initialize_dataset
from pytorch.model_helper import select_model, select_metric, select_optimizer, select_scheduler
from pytorch.train_helper import PerformanceLog
from pytorch.custom_loss import my_loss, my_loss_mse

class TorchPipeline():
    def __init__(self, gpu_device = 0, **kwargs):
        device = torch.device(f"cuda:{gpu_device}" if torch.cuda.is_available() else "cpu")
        
        train_name = kwargs.get("train_name")
        main_model_dir = kwargs.get("main_model_dir")
        train_name_updated = create_dir(train_name, main_model_dir)
        self.model_path = f"{main_model_dir}/{train_name_updated}"
        self.log_path = f"{main_model_dir}/{train_name_updated}/log.csv"

        print("=====================================")
        print("SAVING MODEL TO >>>", self.model_path)
        print("=====================================")

        criterion = self.init_loss(**kwargs)
        train_loader, val_loader, unseen_test_loader = self.init_dataset(**kwargs)
        model, metric_fc, optimizer, scheduler = self.init_model(**kwargs)
        self.train(train_loader, val_loader, model, metric_fc, criterion, optimizer, scheduler, **kwargs)
        self.dump_config(self.model_path, kwargs)

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

        cudnn.benchmark = True
        return criterion

    def init_dataset(self, main_data_dir, batch_size, size, workers, fix_sample = None, fix_sample_val = None, use_train_dir = 'train', **kwargs):
        # initialize dataset generators
        train_loader, val_loader, unseen_test_loader = initialize_dataset(main_data_dir, batch_size, size, workers, fix_sample, fix_sample_val, use_train_dir)
        return train_loader, val_loader, unseen_test_loader 

    def init_model(self, model_type, metric_type, num_ftr, num_classes, optimizer_type, opt_kwargs, scheduler_type, scheduler_kwargs, model_kwargs = {}, **kwargs):
        # create model
        model = select_model(model_type = model_type, model_kwargs = model_kwargs)
        metric_fc = select_metric(metric = metric_type, num_ftr = num_ftr, num_classes = num_classes)
        optimizer = select_optimizer(type = optimizer_type, model = model, kwargs = opt_kwargs)
        scheduler = select_scheduler(optimizer, scheduler_type = scheduler_type, scheduler_kwargs = scheduler_kwargs)
        return model, metric_fc, optimizer, scheduler

    def init_retrain(self, model, premodel_path):
        pass
    
    def train(self, train_loader, val_loader, model, metric_fc, criterion, optimizer, scheduler, epochs, metric_type, batch_multiplier = 1, **kwargs):
        model_path, log_path = self.model_path, self.log_path
       
        # iterate training
        best_loss = float('inf')
        best_acc = 0
        best_avg_acc = 0
        plog = PerformanceLog()
        for epoch in range(epochs):
            print('Epoch [%d/%d]' %(epoch+1, epochs))

            # train for one epoch
            train_log = train(train_loader, model, metric_fc, criterion, optimizer, metric = metric_type, batch_multiplier = batch_multiplier)

            # evaluate on validation set
            val_log = validate(val_loader, model, metric_fc, criterion, metric = metric_type)

            # change learning rate according to scheduler policy
            scheduler.step()

            # print and save performance logs
            perf_logs = []
            perf_logs.append(f'train: ' + 'loss %.4f - acc (%.4f, %.4f)' % (train_log['loss'], train_log['acc_'], train_log['avg_acc_']))
            perf_logs.append(f'validation: ' + 'loss %.4f - acc (%.4f, %.4f)' % (val_log['loss'], val_log['acc_'], val_log['avg_acc_']))
            print(" -- ".join(perf_logs))
            print("LearningRate:", scheduler.get_last_lr())
            plog.append(epoch = epoch, lr = scheduler.get_last_lr()[0], 
                        train_log = train_log, val_log = val_log, test_log = val_log)
            plog.save(log_path) # save logs as csv
            
            # save best model
            if val_log['loss'] < best_loss:
                torch.save(model.state_dict(), f'{model_path}/model.pth')
                torch.save(metric_fc.state_dict(), f'{model_path}/metric_fc.pth')
                best_loss = val_log['loss']
                print("=> saved best model by best loss")

            if val_log['acc_'] > best_acc:
                torch.save(model.state_dict(), f'{model_path}/best_acc_model.pth')
                torch.save(metric_fc.state_dict(), f'{model_path}/best_acc_metric_fc.pth')
                best_acc = val_log['acc_']
                print("=> saved best model by best acc")

            if val_log['avg_acc_'] > best_avg_acc:
                torch.save(model.state_dict(), f'{model_path}/best_avg_acc_model.pth')
                torch.save(metric_fc.state_dict(), f'{model_path}/best_avg_acc_metric_fc.pth')
                best_avg_acc = val_log['avg_acc_']
                print("=> saved best model by best avg acc")

    def dump_config(self, model_path, kwargs):
        output_path = os.path.join(model_path, "config.yaml")
        dump_config(output_path, kwargs)
            

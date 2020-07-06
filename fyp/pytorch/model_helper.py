# from torchviz import make_dot
# make_dot(r).render("attached", format="png")
# https://stackoverflow.com/questions/52468956/how-do-i-visualize-a-net-in-pytorch

# import adabound
import torch
from torch import nn
import torch.optim as optim
from torchvision import models
from torch.optim import lr_scheduler

import pytorch.metrics as metrics
from pytorch.xception import xception
from pytorch.dual_xception import dual_xception, small_dual_xception
from dual.model_new_dual_xception import new_small_dual_xception


def select_model(model_type, model_kwargs):
    pretrained = model_kwargs.get("pretrained")
    if pretrained is None:
        pretrained = True
        
    if model_type == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif model_type == "resnet34":
        model = models.resnet34(pretrained=pretrained)
    elif model_type == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    elif model_type == "resnet101":
        model = models.resnet101(pretrained=pretrained)
    elif model_type == "resnet152":
        model = models.resnet152(pretrained=pretrained)   
    elif model_type == "resnext101_32x8d":
        model = models.resnext101_32x8d(pretrained=pretrained)  
    elif model_type == "vgg16":
        model = models.vgg16_bn(pretrained=pretrained) 
    elif model_type == "xception":
        model = xception() 
    elif model_type == "vgg11":
        model = models.vgg11_bn(pretrained=pretrained) 
    elif model_type == "dual_xception":
        model = dual_xception() 
    elif model_type == "small_dual_xception":
        model = small_dual_xception() 
    elif model_type == "new_small_dual_xception":
        model = new_small_dual_xception(**model_kwargs) 
    elif model_type == "single_resnext50":
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)
        model.fc = nn.Linear(2048, 1)
    elif model_type == "single_resnext101":
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext101_32x8d', pretrained=True)
        model.fc = nn.Linear(2048, 1)
        
    model = model.cuda()
    # print(model)
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    # model.to(device)

    return model


def select_loss(loss_type):
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
    return criterion


def select_metric(metric, num_ftr, num_classes=7):
    if metric == 'adacos':
        metric_fc = metrics.AdaCos(num_features=num_ftr, num_classes=num_classes)
    elif metric == 'arcface':
        metric_fc = metrics.ArcFace(num_features=num_ftr, num_classes=num_classes)
    elif metric == 'sphereface':
        metric_fc = metrics.SphereFace(num_features=num_ftr, num_classes=num_classes)
    elif metric == 'cosface':
        metric_fc = metrics.CosFace(num_features=num_ftr, num_classes=num_classes)
    elif metric == 'softmax':
        metric_fc = nn.Linear(num_ftr, num_classes)
    else:
        metric_fc = nn.Linear(num_ftr, num_classes)
    metric_fc = metric_fc.cuda()
    return metric_fc


def select_optimizer(type, model, kwargs):
    if type == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                        lr=kwargs.get("lr"), 
                        momentum=kwargs.get("momentum"), 
                        weight_decay=kwargs.get("weight_decay"))
    elif type == "AdamW":
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=kwargs.get("lr"))
    elif type == "Adabound":
        optimizer = adabound.AdaBound(filter(lambda p: p.requires_grad, model.parameters()), lr=kwargs.get("lr"), final_lr=kwargs.get("final_lr"))
    return optimizer


def select_scheduler(optimizer, scheduler_type, scheduler_kwargs):
    if scheduler_type == "CosineAnnealingWarmRestarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_kwargs)
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_kwargs)
    elif scheduler_type == "OneCycleLR":
        scheduler = lr_scheduler.OneCycleLR(optimizer, **scheduler_kwargs)
    return scheduler
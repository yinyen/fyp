# from torchviz import make_dot
# make_dot(r).render("attached", format="png")
# https://stackoverflow.com/questions/52468956/how-do-i-visualize-a-net-in-pytorch

import adabound
import torch.optim as optim
from torch import nn
from torchvision import models
from torch.optim import lr_scheduler
import pytorch.metrics as metrics
from pytorch.xception import xception

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
        
    model = model.cuda()
    return model


def select_metric(metric, num_ftr, num_classes=7):
    if metric == 'adacos':
        metric_fc = metrics.AdaCos(num_features=num_ftr, num_classes=num_classes)
    elif metric == 'arcface':
        metric_fc = metrics.ArcFace(num_features=num_ftr, num_classes=num_classes)
    elif metric == 'sphereface':
        metric_fc = metrics.SphereFace(num_features=num_ftr, num_classes=num_classes)
    elif metric == 'cosface':
        metric_fc = metrics.CosFace(num_features=num_ftr, num_classes=num_classes)
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
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# class ArgMax(torch.autograd.Function):
#     def forward(ctx, input):
#         idx = torch.argmax(input, 1)
#         output = torch.zeros_like(input)
#         output.scatter_(1, idx, 1)
#         return output

#     def backward(ctx, grad_output):
#         return grad_output

def soft_arg_max(A, mask, beta=10, dim=1, epsilon=1e-12):
    '''
        applay softargmax on A and consider mask, return \sum_i ( i * (exp(A_i * beta) / \sum_i(exp(A_i * beta))))
        according to https://bouthilx.wordpress.com/2013/04/21/a-soft-argmax/
        :param A:
        :param mask:
        :param dim:
        :param epsilon:
        :return:
        '''
    # According to https://discuss.pytorch.org/t/apply-mask-softmax/14212/7
    A_max = torch.max(A, dim=dim, keepdim=True)[0]
    A_exp = torch.exp((A - A_max)*beta)
    A_exp = A_exp * mask  # this step masks
    A_softmax = A_exp / (torch.sum(A_exp, dim=dim, keepdim=True) + epsilon)
    indices = torch.arange(start=0, end=A.size()[dim]).float()
    print(indices.size(), A_softmax.size())
    return torch.matmul(A_softmax, indices)


def asm(A):
    dim = 1
    epsilon = 1e-12
    beta = 10
    A_max = torch.max(A,dim=1,keepdim=True)[0]
    A_exp = torch.exp((A-A_max)*beta)
    A_exp = A_exp * (A == 0).type(torch.FloatTensor).cuda() # this step masks
    A_softmax = A_exp / (torch.sum(A_exp,dim=1,keepdim=True)+epsilon)
    indices = torch.arange(start=0, end=A.size()[dim]).float().cuda()
    # print(indices.size(), A_softmax.size())
    return torch.matmul(A_softmax, indices)
    # return A_softmax
def asm2(A):
    dim = 1
    epsilon = 1e-12
    beta = 100
    A_max = torch.max(A,dim=1,keepdim=True)[0]
    A_exp = torch.exp((A-A_max)*beta)
    A_softmax = A_exp / (torch.sum(A_exp,dim=1,keepdim=True)+epsilon)
    indices = torch.arange(start=0, end=A.size()[dim]).float().cuda()
    return torch.matmul(A_softmax, indices)

    
def my_loss(output, target):
    k1, k2 = 5,3
    # k1,k2 = 2,1
    
    # raise Exception()
    eps = 1e-10
    b = asm2(output)
    loss1 = (b-target+eps).pow(2).pow(0.5)
    loss1 = loss1.type(torch.float32)
    loss2 = (5-b).pow(2).pow(0.5/k1)
    loss3 = (1+100*target).pow(2).pow(0.5/k2)
    # loss_agg = loss1*loss2*loss3
    loss_agg = loss1
    loss = torch.mean(loss_agg)

    # print("PRED")
    # print(torch.argmax(output, dim = 1))
    # print("LOSS")
    # print(loss_agg)
    # print(loss)
    # loss = torch.mean(loss1)

    return loss

    
def my_loss_mse(output, target):
    b = asm2(output)
    loss1 = (b-target).pow(2)
    loss1 = loss1.type(torch.float32)
    loss = torch.mean(loss1)
    return loss
import torch
import torch.nn.functional as F


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
    beta = 100
    A_max = torch.max(A,dim=1,keepdim=True)[0]
    A_exp = torch.exp((A-A_max)*beta)
    A_softmax = A_exp / (torch.sum(A_exp,dim=1,keepdim=True)+epsilon)
    indices = torch.arange(start=0, end=A.size()[dim]).float().cuda()
    return torch.matmul(A_softmax, indices)


m = torch.nn.Softmax(dim=1)
input = torch.randn(2, 3).cuda()
print(input)

d = asm(input)
print("++++++++++++++++++")
print(d)

output = F.log_softmax(input)
print(output)
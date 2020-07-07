# def ss2(c,d ):
#     return c - d

# def ss(a,b,**kwargs):
#     print(a, b)
#     print(kwargs)

#     c2 = ss2(**kwargs)
#     print(c2)
#     return a+b

# kwargs = {"a": 1, "b": 2, "c": 3, "d":4}
# # kwargs = dict(a = 1, b= 2)

# d=ss(**kwargs)
# print(d)

import numpy as np
x0 = [1,2,3,4]
x0.append(5)
print(x0)
x = np.array([1,2,3,4])
print(x.shape)

from pytorch.dual_data_helper import create_dual_label_df
full_df = create_dual_label_df(main_data_dir = "../all_train_300", train_dir_list = ["full_train"])
print(full_df)
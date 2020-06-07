import torch
import numpy as np
import random

a = torch.zeros(1, 9, 4)
b = torch.ones(1444, 1, 4)
c = a + b
# print(c.shape)

d = torch.zeros(1, 2, 3, 3)
# print(d[0])

# x = np.random.rand(2, 3)
# x = np.arange(0, 10, 2)
# x = np.vstack([x, x])
# print(x)
# print(x.shape)
# result = np.where(x > 0.3)
# print(result)

# x = np.random.randn(10, 4)
x = np.arange(0, 20, 1).reshape(5, 4, 1)
# y = np.random.randn(5, 4)
y = np.arange(10, 30, 1).reshape(5, 4, 1)
# print(x)
# tl = np.maximum(x[:, None, :2], y[:, :2])
# print(x[:, None, :2])
# print(y[:, :2].shape)
# print(tl)
# print(tl.shape)
# br = np.minimum(x[:, None, 2:], y[:, :2])
# print(x[:, None, 2:])
# print(y[:, 2:].shape)
# print(br)
# print(br.shape)


# print(x - y)
# temp = np.prod(x - y, axis=1) * (x > y).all(axis=2)
# print(temp)


w = np.ones((4, 2))
# print(w[:, :, None].shape)

tt = np.zeros((3, 1))
ww = np.ones((3, 1))
index = [1, 2, 1]
ww[index] = tt
# print(ww)

# cc = np.random.rand(4, 2)
# print(cc > 0.4)

# aa = [1, 5, 3, 2]
# random.shuffle(aa)
# print(aa)

bb = np.ones((300, 2, 3))
zz = torch.from_numpy(np.array(bb))
print(zz)
cc = []
dd = [4, 5, 6, 7]
cc.append(bb)
cc.append(dd)
print(cc)

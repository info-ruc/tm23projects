import torch

z = torch.zeros(5, 3)
print(z)
print(z.dtype)

i = torch.ones((5, 3), dtype=torch.int16)
print(i)

torch.manual_seed(1729)
r1 = torch.rand(2, 2)
r2 = torch.rand(2, 2)
print(r1)
print(r2)

torch.manual_seed(1729)
r3 = torch.rand(2, 2)
print(r3)

ones = torch.ones(2, 3)
print(ones)

twos = torch.ones(2, 3) * 2
print(twos)

threes = ones + twos
print(threes)
print(threes.shape)

r1 = torch.rand(2, 3)
r2 = torch.rand(3, 2)
print(r1)
print(r2)

r = torch.rand(2, 2) - 0.5 * 2
print(r)
print(torch.abs(r))
print(torch.asin(r))
print(torch.det(r))
print(torch.svd(r))
print(torch.std_mean(r))
print(torch.max(r))

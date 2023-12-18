import torch

x = torch.randn(1, 10)
prev_h = torch.randn(1, 20)
W_h = torch.randn(20, 20)
W_x = torch.randn(20, 10)

i2h = torch.mm(W_x, x.t())
h2h = torch.mm(W_h, prev_h.t())
next_h = i2h + h2h

print(i2h)
print(h2h)
print(next_h)

next_h = next_h.tanh()
print(next_h)

loss = next_h.sum()
#loss.backward()
print(loss) 

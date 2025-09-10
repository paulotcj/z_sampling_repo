'''
We are going to be deterministic with the gradient numbers.
In this example we will process a batch of 4 rows and 16 cols and validate them against a set of 4 rows and 1 col.

In the first example we will process them all at once. execute the backward pass and get the gradients of the first
linear layer.

In the second example we will use a simulation of a DDP (distributed data parallel). Where we loop through the data
4 times as per 4 rows, send 16 numbers to the forward method. Get a loss and then execute the apropriate adjustments
for the adjusted loss and then execute the backward pass.

All of this should result in equal gradients as they are equivalent.

Quick note as a refresher: gradients are used in the backpropagation, they tell how much each weight in net[0].weight 
should change to reduce the loss.
'''


import torch

# super simple little MLP
net = torch.nn.Sequential(
    torch.nn.Linear(16, 32),
    torch.nn.GELU(),
    torch.nn.Linear(32, 1)
)
torch.random.manual_seed(42)

x = torch.randn(4, 16) # rand 4 rows 16 cols
y = torch.randn(4, 1)  # rand 4 rows 1 col

net.zero_grad()

yhat = net(x) #forward pass, process all at once [4,16]

loss = torch.nn.functional.mse_loss(yhat, y) #get loss

loss.backward() # apply the backward pass

print(net[0].weight.grad.view(-1)[:10]) # this is the gradient achieved after the backward pass
'''
tensor([ 0.0184,  0.0163,  0.0032, -0.0253,  0.0035, -0.0100,  0.0018,  0.0106,
         0.0270, -0.0163])
'''

# the loss objective here is (due to readuction='mean')
# L = 1/4 * [
#            (y[0] - yhat[0])**2 +
#            (y[1] - yhat[1])**2 +
#            (y[2] - yhat[2])**2 +
#            (y[3] - yhat[3])**2
#           ]
# NOTE: 1/4!


print('-------------------------------------------------------------------------')
print('\n\n')


# now let's do it with grad_accum_steps of 4, and B=1
# the loss objective here is different because
# accumulation in gradient <---> SUM in loss
# i.e. we instead get:
# L0 = 1/4(y[0] - yhat[0])**2
# L1 = 1/4(y[1] - yhat[1])**2
# L2 = 1/4(y[2] - yhat[2])**2
# L3 = 1/4(y[3] - yhat[3])**2
# L = L0 + L1 + L2 + L3
# NOTE: the "normalizer" of 1/4 is lost

net.zero_grad()


#-----------------
for i in range(4):
    yhat = net(x[i])
    loss = torch.nn.functional.mse_loss(yhat, y[i])
    loss = loss / 4 # <-- have to add back the "normalizer"!
    loss.backward()
#-----------------


print(net[0].weight.grad.view(-1)[:10])
'''
tensor([ 0.0184,  0.0163,  0.0032, -0.0253,  0.0035, -0.0100,  0.0018,  0.0106,
         0.0270, -0.0163])
'''

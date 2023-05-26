import torch
X = torch.rand(30, 2)
y = torch.rand(30)
weight = torch.rand(2, 1)
#add _ to bool object to call the func
weight.requires_grad_(True)
print(weight.requires_grad)

optimizer = torch.optim.Adam(params=[weight], lr=0.01)
lambda_lr = lambda epoch: 0.6**epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)
for i in range(30):
    optimizer.zero_grad()
    pred = X @ weight
    loss = torch.nn.MSELoss()(pred, y)
    loss.backward()
    optimizer.step()
    # scheduler.step()
    print(f'loss {i}', loss.item())
    print(f'learning rate {i}', optimizer.param_groups[0]['lr'])
from losses import HDLoss
hd_loss = HDLoss(include_background=False, to_onehot_y=True, softmax=True, batch=False)
loss = hd_loss(input, target)
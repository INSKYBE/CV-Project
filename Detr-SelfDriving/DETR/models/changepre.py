import torch

pretrained_weights = torch.load('detr-r101-2c7b67e5.pth')
num_classes = 18 + 1
pretrained_weights["model"]["class_embed.weight"].resize_(num_classes + 1, 256)
pretrained_weights["model"]["class_embed.bias"].resize_(num_classes + 1)
torch.save(pretrained_weights, "detr_r101_%d.path" % num_classes)

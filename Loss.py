import torch

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, SR, GT):
        num = GT.size(0)
        smooth = 1

        m1 = SR
        m2 = GT
        intersection = (m1 * m2)

        loss = 1 - (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        loss = loss.sum() / num


        return loss
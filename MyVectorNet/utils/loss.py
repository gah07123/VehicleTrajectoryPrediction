import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorLoss(nn.Module):
    """
    Loss = L_traj + alpha * L_node
    where L_traj is the negative Gaussian log-likelihood loss, L_node is the huber loss
    """
    def __init__(self, alpha=1.0, aux_loss=False, reduction='sum'):
        super(VectorLoss, self).__init__()

        self.alpha = alpha
        self.aux_loss = aux_loss
        if reduction in ["mean", "sum"]:
            self.reduction = reduction
        else:
            raise NotImplementedError("[Loss]: The reduction has not been implemented!")

    def forward(self, pred, gt, aux_pred=None, aux_gt=None):
        batch_size = pred.size()[0]
        loss = 0.0

        l_traj = F.mse_loss(pred, gt, reduction="sum")
        if self.reduction == "mean":
            l_traj /= batch_size
        loss += l_traj

        if self.aux_loss:
            # return nll loss if pred is None
            if not isinstance(aux_pred, torch.Tensor) or not isinstance(aux_gt, torch.Tensor):
                return loss
            assert aux_pred.size() == aux_gt.size(), "[Loss]: The dim of prediction and ground truth don't match!"
            # smooth l1 loss 就是huber loss
            l_node = F.smooth_l1_loss(aux_pred, aux_gt, reduction="sum")
            if self.reduction == "mean":
                l_node /= batch_size
            loss += self.alpha * l_node

        return loss


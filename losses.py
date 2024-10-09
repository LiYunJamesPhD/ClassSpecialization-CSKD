import torch
import torch.nn as nn
import torch.nn.functional as F


#(1) Re-normalized Distillation Loss; here we use the standard Knowledge Distillation loss used for Teacher-Student distillation; however, because we are training the student on a subset of classes, we only use a "renormalized logit" (which is to say logits renormalized with respect to the logit values produced by the teacher only for the subset of interest).
class RenormalizedKDLoss(nn.Module):   # proposed
    def __init__(self, T=1.0, alpha=0.2, device=None, interest_class=None):
        super().__init__()
        self.kld_loss = nn.KLDivLoss(reduction='batchmean')
        self.interest_class = interest_class
        self.T = T
        self.alpha = alpha
        self.device = device

    def __call__(self, student_logits, teacher_logits):

        student_res = F.log_softmax(student_logits / self.T, dim=1)
        interest_logits = torch.index_select(teacher_logits, 1, self.interest_class)
        
        teacher_res = F.softmax(interest_logits / self.T, dim=1)
        kd_loss = self.kld_loss(student_res, teacher_res) * self.alpha * self.T * self.T
        return kd_loss


# borrow the code from https://github.com/kahnchana/opl/blob/master/loss.py
class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5, opl_hyper=0.2, device=None):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma
        self.opl_hyper = opl_hyper
        self.device = device

    def forward(self, features, labels=None):
        # device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        #  features are normalized
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(self.device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(self.device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs

        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

        return loss * self.opl_hyper


# Intra-Class Variance loss (Proposed loss)
class var_min_ce(nn.Module):
    def __init__(self, icv_hyper=0.5, num_classes=2, device=None):
        super(var_min_ce, self).__init__()
        self.icv_hyper = icv_hyper
        self.device=device
        self.num_classes = num_classes

    def forward(self, features, labels=None):
        
        features = features.to(self.device)

        var_loss = 0

        for idx in range(self.num_classes):
            class_indices = torch.where(labels == idx)[0].cpu().detach().numpy()
            var_loss += torch.norm(torch.var(features[class_indices, :], axis=1))

        return var_loss * self.icv_hyper



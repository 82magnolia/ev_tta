import torch.nn.functional as F
import torch.nn as nn
import torch


class SoftCrossEntropy(nn.Module):
    """
    Measure cross entropy between two distributions.
    Assume that both input, target have shape (N, C).
    Both input, target are given as logits.
    """

    def __init__(self, temperature):
        super(SoftCrossEntropy, self).__init__()
        self.temperature = temperature

    def forward(self, input, target, return_vector=False):
        logprobs = F.log_softmax(input / self.temperature, dim=1)
        tgt_dist = F.softmax(target / self.temperature, dim=1)
        if return_vector:
            return -(tgt_dist * logprobs).sum(-1)  # (B, )
        else:
            return -(tgt_dist * logprobs).sum() / input.shape[0]


class KLDivLogits(nn.Module):
    """
    Measure KL Divergence between two distributions.
    """

    def __init__(self, input_distrib=False, tgt_distrib=False):
        super(KLDivLogits, self).__init__()
        self.kl_func = nn.KLDivLoss(reduction='batchmean')
        self.input_distrib = input_distrib
        self.tgt_distrib = tgt_distrib

    def forward(self, input, target):
        if self.input_distrib:
            input_dist = torch.log(input)
        else:
            input_dist = F.log_softmax(input, dim=1)
        if not self.tgt_distrib:
            tgt_dist = F.softmax(target, dim=1)
        else:
            tgt_dist = target
        return self.kl_func(input_dist, tgt_dist)


class InfoRadiusLoss(nn.Module):
    """
    Information radius for KL in multiple distributions
    """

    def __init__(self):
        super(InfoRadiusLoss, self).__init__()
        self.kl_func = nn.KLDivLoss(reduction='batchmean')

    def forward(self, input):
        num_dist = input.shape[1]

        # Input is assumed as logits
        input_dist = F.softmax(input, dim=-1)  # (B, K, L)
        avg_dist = torch.mean(input_dist, dim=1)  # (B, L)
        avg_dist = F.log_softmax(avg_dist, dim=1)  # (B, L), convert to log probabilities

        loss = 0.
        for idx in range(num_dist):
            loss += self.kl_func(avg_dist, input_dist[:, idx, :])

        loss /= num_dist
        return loss


class NegLogRatioLoss(nn.Module):
    """
    Measure cross entropy between two distributions.
    Assume that both input, target have shape (N, C).
    Both input, target are given as logits.
    """

    def __init__(self, temperature):
        super(NegLogRatioLoss, self).__init__()
        self.temperature = temperature

    def forward(self, input, target, return_vector=False):
        exp_neg = torch.sum(torch.exp(input), dim=-1, keepdim=True) - torch.exp(input)
        log_exp_neg = torch.log(exp_neg)
        tgt_dist = F.softmax(target / self.temperature, dim=1)
        if torch.isinf(log_exp_neg).sum() != 0:
            logprobs = F.log_softmax(input / self.temperature, dim=1)
            loss_val = -(tgt_dist * logprobs)
        else:
            loss_val = tgt_dist * (-input + log_exp_neg)

        if return_vector:
            return loss_val.sum(-1)  # (B, )
        else:
            return loss_val.sum() / loss_val.shape[0]

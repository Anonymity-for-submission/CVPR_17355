"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, real_label=None,mask=None,if_weak=False):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        # print(features.shape)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None and real_label is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, real_label.T).float().to(device)
        elif labels is None and real_label is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(real_label, real_label.T).float().to(device)
        elif labels is not None and real_label is None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        # print(mask)
        # print("bsz:")
        # print(bsz)
        contrast_count = features.shape[1]
        # print(contrast_count)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print(contrast_count)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            
            
            # print(bsz)
            # print(len(contrast_feature))
            # print(1)
            if if_weak:
                # print(2)
                # print("weak")
                anchor_feature = contrast_feature[:batch_size]
                contrast_feature = contrast_feature[batch_size:]
                # print(anchor_feature.shape)
                # print(contrast_feature.shape)
                contrast_count=1
                # contrast_count = bsz
            else:
                anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        # print(anchor_dot_contrast)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # print(logits)
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        # print(mask.shape)
        # print(mask.shape)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # print(logits_mask)
        mask = mask * logits_mask
        # print(mask.shape)
        # compute log_prob
        # print(torch.exp(logits).shape)
        # print(logits)
        # print(logits_mask)
        exp_logits = torch.exp(logits) * logits_mask
        # p/rint(exp_logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print(log_prob)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # print(mask.sum(1))
        # print(mean_log_prob_pos)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # print(loss)
        loss = loss.view(anchor_count, batch_size).mean()
        # print(loss.shpe)
        return loss
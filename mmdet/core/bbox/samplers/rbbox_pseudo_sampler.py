import torch

from .rbbox_base_sampler import RbboxBaseSampler
from .sampling_result import SamplingResult


class PseudoRbboxSampler(RbboxBaseSampler):

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        raise NotImplementedError

    def sample(self, assign_result, rbboxes, gt_rbboxes, **kwargs):
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0).squeeze(-1).unique()
        gt_flags = rbboxes.new_zeros(rbboxes.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(pos_inds, neg_inds, rbboxes, gt_rbboxes,
                                         assign_result, gt_flags)
        return sampling_result

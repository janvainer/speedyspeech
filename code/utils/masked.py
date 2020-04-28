import torch


class MaskedBatchNorm1d(torch.nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(MaskedBatchNorm1d, self).__init__(num_features, eps=eps, momentum=momentum,
                                                affine=affine, track_running_stats=track_running_stats)

    def forward(self, input, lens):
        """

        :param input: (batch, channels, time)
        :param lens:
        :return:
        """
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        m = mask(input, lens, dim=-1)

        # calculate running estimates
        if self.track_running_stats:
            if self.training:
                mean = masked_mean(input, m, dim=[0, 2], keepdim=False)
                # use biased var in train
                var = masked_var(input, m, dim=[0, 2], unbiased=False, keepdim=False)
                n = input.numel() / input.size(1)  # number of averaged items
                with torch.no_grad():
                    self.running_mean = exponential_average_factor * mean \
                                        + (1 - exponential_average_factor) * self.running_mean
                    # update running_var with unbiased var
                    self.running_var = exponential_average_factor * var * n / (n - 1) \
                                       + (1 - exponential_average_factor) * self.running_var
            else:
                mean = self.running_mean
                var = self.running_var
        else:
            mean = masked_mean(input, m, dim=[0, 2], keepdim=False)
            # use biased var in train
            var = masked_var(input, m, dim=[0, 2], unbiased=True, keepdim=False)

        input = (input - mean[None, :, None]) / (torch.sqrt(var[None, :, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None] + self.bias[None, :, None]

        return input


def mask(x, lengths, dim=-1):

    assert dim != 0, 'Masking not available for batch dimension'
    assert len(lengths) == x.shape[0], 'Lengths must contain as many elements as there are items in the batch'

    lengths = torch.as_tensor(lengths)

    to_expand = [1] * (x.ndim-1)+[-1]
    mask = torch.arange(x.shape[dim]).expand(to_expand).transpose(dim, -1).expand(x.shape).to(lengths.device)
    mask = mask < lengths.expand(to_expand).transpose(0, -1)
    return mask.to(x.device)


def masked_max(x, mask, dim=None, keepdim=False):
    if dim is None:
        return torch.max(x.masked_fill(~mask, float('-inf')))
    return torch.max(x.masked_fill(~mask, float('-inf')), dim=dim, keepdim=keepdim)


def masked_min(x, mask, dim=None, keepdim=False):
    if dim is None:
        return torch.min(x.masked_fill(~mask, float('inf')))
    return torch.min(x.masked_fill(~mask, float('inf')), dim=dim, keepdim=keepdim)


def masked_sum(x, mask, dim=None, keepdim=False):
    if dim is None:
        dim = tuple([i for i in range(x.ndim)])
    return torch.sum(x.masked_fill(~mask, 0.0), dim=dim, keepdim=keepdim)


def masked_mean(x, mask, dim=None, keepdim=False):
    if dim is None:
        dim = tuple([i for i in range(x.ndim)])
    s = masked_sum(x, mask, dim, keepdim)
    return s / mask.sum(dim=dim, keepdim=keepdim).float()


def masked_var(x, mask, dim=None, keepdim=False, unbiased=True):
    if dim is None:
        dim = tuple([i for i in range(x.ndim)])

    m = masked_mean(x, mask, dim, keepdim=True)
    m = masked_sum(torch.pow(x - m, 2.0), mask, dim, keepdim)
    if unbiased:
        return m / (mask.sum(dim=dim, keepdim=keepdim) - 1).float()

    return m / (mask.sum(dim=dim, keepdim=keepdim)).float()


def masked_std(x, mask, dim=None, keepdim=False, unbiased=True):
    return torch.sqrt(masked_var(x, mask, dim, keepdim, unbiased))

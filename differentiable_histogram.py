import torch


#############################################
# Differentiable Histogram Counting Method
#############################################
def differentiable_histogram(x, bins=255, min=0.0, max=1.0):

    if len(x.shape) == 4:
        n_samples, n_chns, _, _ = x.shape
    elif len(x.shape) == 2:
        n_samples, n_chns = 1, 1
    else:
        raise AssertionError('The dimension of input tensor should be 2 or 4.')

    hist_torch = torch.zeros(n_samples, n_chns, bins).to(x.device)
    delta = (max - min) / bins

    BIN_Table = torch.range(start=0, end=bins, step=1) * delta

    for dim in range(1, bins-1, 1):
        h_r = BIN_Table[dim].item()             # h_r
        h_r_sub_1 = BIN_Table[dim - 1].item()   # h_(r-1)
        h_r_plus_1 = BIN_Table[dim + 1].item()  # h_(r+1)

        mask_sub = ((h_r > x) & (x >= h_r_sub_1)).float()
        mask_plus = ((h_r_plus_1 > x) & (x >= h_r)).float()

        hist_torch[:, :, dim] += torch.sum(((x - h_r_sub_1) * mask_sub).view(n_samples, n_chns, -1), dim=-1)
        hist_torch[:, :, dim] += torch.sum(((h_r_plus_1 - x) * mask_plus).view(n_samples, n_chns, -1), dim=-1)

    return hist_torch / delta



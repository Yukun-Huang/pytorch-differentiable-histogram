import torch


#############################################
# Differentiable Histogram Counting Method
#############################################
def differentiable_histogram(x, bins=255, min=0.0, max=1.0):
    assert len(x.shape) >= 2 
    n_samples, n_chns = x.shape[:2]
    
    hist_torch = torch.zeros(n_samples, n_chns, bins).to(x.device)
    delta = (max - min) / (bins-1)

    BIN_Table = torch.arange(start=0, end=bins+1, step=1)*delta

    for dim in range(0, bins, 1):
        h_r = BIN_Table[dim].item() if dim > 0 else 0
        h_r_sub_1 = BIN_Table[dim - 1].item()
        h_r_plus_1 = BIN_Table[dim + 1].item()

        mask_sub = ((h_r > x) & (x >= h_r_sub_1)).float()
        mask_plus = ((h_r_plus_1 > x) & (x >= h_r)).float()

        hist_torch[:, :, dim] += torch.sum(((x - h_r_sub_1) * mask_sub).view(n_samples, n_chns, -1), dim=-1)
        hist_torch[:, :, dim] += torch.sum(((h_r_plus_1 - x) * mask_plus).view(n_samples, n_chns, -1), dim=-1)
        
    return hist_torch / hist_torch.sum(dim=-1, keepdim=True)


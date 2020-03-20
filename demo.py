import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from differentiable_histogram import differentiable_histogram


if __name__ == '__main__':

    # Load image
    src = Image.open('src.png').convert('L').resize((100, 100), Image.BILINEAR)
    src = torch.Tensor(np.array(src) / 255.0)
    src.requires_grad = True

    # Compute histogram (official implementation)
    hist_standard = torch.histc(src, bins=255, min=0.0, max=1.0)

    # Compute histogram (differentiable implementation)
    hist_differentiable = differentiable_histogram(src, bins=255, min=0.0, max=1.0)[0, 0, :]

    # Backward Test
    try:
        torch.mean(hist_standard).backward()     # Error
    except Exception as e:
        print(e)
    torch.mean(hist_differentiable).backward()
    print(src.grad)

    # Show
    plt.figure()
    plt.subplot(121)
    plt.plot(hist_standard.detach().numpy())
    plt.subplot(122)
    plt.plot(hist_differentiable.detach().numpy())
    plt.show()



import numpy as np
import torch
from torch import Tensor, nn
from torch.autograd import Variable


# Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are
# Failing to Reproduce Spectral Distributions
#
# https://arxiv.org/pdf/2003.01826.pdf


def grayscale(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def azimuthal_average(image, center=None):
    """
    from https://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/

    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fractional pixels).

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]  # location of changed radius
    nr = rind[1:] - rind[:-1]  # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof


class SpectralRegularizer:
    def __init__(self, epsilon=1e-8):
        super().__init__()

        self.coefficient = 1e-5
        self.epsilon = epsilon
        self.num = 88
        self.criterion = nn.BCELoss()

    def __call__(self, g_loss: Tensor, real_images: Tensor, fake_images: Tensor):
        # fake image 1d power spectrum
        fake_psd1d = np.zeros([fake_images.shape[0], self.num])
        for t in range(fake_images.shape[0]):
            generated_images = fake_images.permute(0, 2, 3, 1)
            images_numpy = generated_images[t, :, :, :].cpu().detach().numpy()
            images_gray = grayscale(images_numpy)

            fft = np.fft.fft2(images_gray)
            fshift = np.fft.fftshift(fft)
            fshift += self.epsilon
            magnitude_spectrum = 20 * np.log(np.abs(fshift))

            psd1d = azimuthal_average(magnitude_spectrum)
            psd1d = (psd1d - np.min(psd1d)) / (np.max(psd1d) - np.min(psd1d))
            fake_psd1d[t, :] = psd1d

        fake_psd1d = torch.from_numpy(fake_psd1d).float()
        fake_psd1d = Variable(fake_psd1d, requires_grad=True).to(real_images.device)

        # real image 1d power spectrum
        real_psd1d = np.zeros([real_images.shape[0], self.num])
        for t in range(real_images.shape[0]):
            generated_images = real_images.permute(0, 2, 3, 1)
            images_numpy = generated_images[t, :, :, :].cpu().detach().numpy()
            images_gray = grayscale(images_numpy)

            fft = np.fft.fft2(images_gray)
            fshift = np.fft.fftshift(fft)
            fshift += self.epsilon
            magnitude_spectrum = 20 * np.log(np.abs(fshift))

            psd1d = azimuthal_average(magnitude_spectrum)
            psd1d = (psd1d - np.min(psd1d)) / (np.max(psd1d) - np.min(psd1d))
            real_psd1d[t, :] = psd1d

        real_psd1d = torch.from_numpy(real_psd1d).float()
        real_psd1d = Variable(real_psd1d, requires_grad=True).to(real_images.device)

        loss_freq = self.criterion(real_psd1d, fake_psd1d.detach())
        loss_freq *= g_loss

        return self.coefficient * loss_freq

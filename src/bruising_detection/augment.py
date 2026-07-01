import torch


class CubeAugmenter:
    def __init__(self, config):
        self.config = config

    def __call__(self, x):
        c = self.config
        if torch.rand(()) < c.aug_p:
            x = torch.flip(x, dims=[1])
        if torch.rand(()) < c.aug_p:
            x = torch.flip(x, dims=[2])
        if torch.rand(()) < c.aug_p:
            x = torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=[1, 2])
        if c.max_shift > 0 and torch.rand(()) < c.shift_p:
            x = self._shift(x, c.max_shift)
        if c.noise_std > 0:
            x = x + torch.randn_like(x) * c.noise_std
        if c.intensity_scale > 0:
            x = x * (1 + (torch.rand(()) * 2 - 1) * c.intensity_scale)
        if c.intensity_shift > 0:
            x = x + (torch.rand(()) * 2 - 1) * c.intensity_shift
        if c.band_dropout_p > 0:
            x = x * (torch.rand(x.size(0), 1, 1) > c.band_dropout_p)
        if c.erase_p > 0 and torch.rand(()) < c.erase_p:
            h, w = x.shape[1:]
            eh, ew = min(c.erase_size, h), min(c.erase_size, w)
            y = int(torch.randint(0, h - eh + 1, (1,)))
            z = int(torch.randint(0, w - ew + 1, (1,)))
            x[:, y:y + eh, z:z + ew] = 0
        return x

    @staticmethod
    def _shift(x, max_shift):
        h, w = x.shape[1:]
        dy = int(torch.randint(-max_shift, max_shift + 1, (1,)))
        dx = int(torch.randint(-max_shift, max_shift + 1, (1,)))
        padded = torch.nn.functional.pad(x, (max_shift, max_shift, max_shift, max_shift), mode="replicate")
        y0 = max_shift - dy
        x0 = max_shift - dx
        return padded[:, y0:y0 + h, x0:x0 + w]

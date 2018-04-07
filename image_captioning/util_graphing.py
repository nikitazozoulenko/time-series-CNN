from torchvision import transforms

def losses_to_ewma(losses, alpha = 0.998):
    losses_ewma = []
    ewma = losses[0]
    for loss in losses:
        ewma = alpha*ewma + (1-alpha)*loss
        losses_ewma += [ewma]
    return losses_ewma


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std


    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class PredictionPreviewer(object):
    def __init__(self):
        self.ppr = PredictionPreviewerReturner()
    

    def __call__(self, var, caption, gt, lang):
        im, caption, gt_caption = self.ppr(var, caption, gt, lang)
        print("CAPTION", caption)
        print("GT", gt_caption)
        im.show()


class PredictionPreviewerReturner(object):
    def __init__(self):
        self.unnormalize = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.tensor2pil = transforms.ToPILImage()
    
    
    def __call__(self, var):
        im = self.tensor2pil(self.unnormalize(var.data.cpu()[0]))
        return im




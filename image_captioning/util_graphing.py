import matplotlib.pyplot as plt

from torchvision import transforms


def losses_to_ewma(losses, alpha = 0.998):
    losses_ewma = []
    ewma = losses[0]
    for loss in losses:
        ewma = alpha*ewma + (1-alpha)*loss
        losses_ewma += [ewma]
    return losses_ewma


def graph(x_indices, losses_train, x_indices_val, losses_val):
    plt.figure(1)
    plt.plot(x_indices_val, losses_val, "r", label="Validation Loss")
    plt.plot(x_indices, losses_train, "g", label="Train Loss")
    plt.legend(loc=1)

    # plt.figure(2)
    # plt.plot(val_x_indices, val_class_losses, "r--", label="Val Class Loss")
    # plt.plot(val_x_indices, val_coord_losses, "g--", label="Val Coord Loss")
    # plt.plot(val_x_indices, val_total_losses, "b--", label="Val Total Loss")
    # plt.legend(loc=1)

    # plt.figure(3)
    # plt.plot(x_indices, total_losses, "b", label="Loss")
    # plt.plot(val_x_indices, val_total_losses, "g--", label="Val Loss")
    # plt.legend(loc=1)

    # plt.figure(4)
    # plt.plot(x_indices, coord_losses, "b", label="coord_losses")
    # plt.plot(val_x_indices, val_coord_losses, "g--", label="val_coord_losses")
    # plt.legend(loc=1)

    # plt.figure(5)
    # plt.plot(x_indices, class_losses, "b", label="class_losses")
    # plt.plot(val_x_indices, val_class_losses, "g--", label="val_class_losses")
    # plt.legend(loc=1)
    plt.show()


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




import matplotlib.pyplot as plt
# plt.switch_backend("agg")
import numpy as np

class Grapher():
    def __init__(self):
        pass
    

    def textlog2numpy(self, path = "train_losses.txt"):
        values = []
        indices = []
        with open("savedir/logs/"+path, "r") as f:
            for line in f:
                line = line.replace("\n", "")
                line = line.split(",")
                values += [float(line[0])]
                indices += [int(line[1])]
        return values, indices


    def values2ewma(self, losses, alpha = 0.9):
        losses_ewma = []
        ewma = losses[0]
        for loss in losses:
            ewma = alpha*ewma + (1-alpha)*loss
            losses_ewma += [ewma]
        return losses_ewma

    
    def graph(self, list_values, list_indices, list_colors, legendlabels, ylabel, xlabel, list_ewmas = None):
        if list_ewmas != None:
            for values, alpha in zip(list_values, list_ewmas):
                values[:] = self.values2ewma(values, alpha=alpha)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)

        for values, indices, color, legenlabel in zip(list_values, list_indices, list_colors, legendlabels):
            plt.plot(indices, values, color, label=legenlabel)
        plt.legend(loc=1)
        # plt.show()
        plt.savefig(legendlabels[0]+".png")


if __name__ == "__main__":
    grapher = Grapher()
    values, indices = grapher.textlog2numpy("val_acc.txt")
    values2, indices2 = grapher.textlog2numpy("val_losses.txt")
    values1, indices1 = grapher.textlog2numpy("train_losses.txt")
    #grapher.graph([values1,values2], [indices1, indices2], ["r", "g"], [ "Training Loss", "Validation Loss"], "Loss", "Iterations")
    #grapher.graph([values], [indices], ["b"], ["Validation Accuracy"], "Accuracy", "Iterations")

    values2, indices2 = grapher.textlog2numpy("test_losses.txt")
    values1, indices1 = grapher.textlog2numpy("test_acc.txt")
    #grapher.graph([values2], [indices2], ["r"], [ "Test Loss", "Validation Loss"], "Loss", "Iterations")
    #grapher.graph([values1], [indices1], ["b"], ["Test Accuracy"], "Accuracy", "Iterations")
    print(np.stack([values1, indices1], axis=1))
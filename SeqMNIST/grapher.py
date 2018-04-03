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
        plt.show()
        #plt.savefig(legendlabels[0]+".png")


if __name__ == "__main__":
    grapher = Grapher()
    val_acc = []
    val_losses = []
    train_losses = []
    test_losses = []
    test_acc = []
    for prefix in ["single_", "double_", "radical_"]:
        val_acc.append(grapher.textlog2numpy(prefix+"val_acc.txt"))
        val_losses.append(grapher.textlog2numpy(prefix+"val_losses.txt"))
        train_losses.append(grapher.textlog2numpy(prefix+"train_losses.txt"))
        test_losses.append(grapher.textlog2numpy(prefix+"test_losses.txt"))
        test_acc.append(grapher.textlog2numpy(prefix+"test_acc.txt"))

    grapher.graph([train_losses[0][0], train_losses[1][0], train_losses[2][0], val_losses[0][0], val_losses[1][0], val_losses[2][0]],
                  [train_losses[0][1], train_losses[1][1], train_losses[2][1], val_losses[0][1], val_losses[1][1], val_losses[2][1]],
                  ["r", "g", "b", "r--", "g--", "b--"],
                  ["Single Training Loss", "Double Training Loss", "Radical Training Loss","Single Validation Loss", "Double Validation Loss", "Radical Validation Loss"],
                  "Loss", "Iterations",
                  list_ewmas = [0.999, 0.999, 0.999, 0.9, 0.9, 0.9])

    grapher.graph([val_acc[0][0], val_acc[1][0], val_acc[2][0]],
                  [val_acc[0][1], val_acc[1][1], val_acc[2][1]],
                  ["r--", "g--", "b--"],
                  ["Single Validation Accuracy", "Double Validation Accuracy", "Radical Validation Accuracy"],
                  "Accuracy", "Iterations",
                  list_ewmas = [0.9, 0.9, 0.9])

    grapher.graph([test_losses[0][0], test_losses[1][0], test_losses[2][0]],
                  [test_losses[0][1], test_losses[1][1], test_losses[2][1]],
                  ["r", "g", "b"],
                  ["Single Test Loss", "Double Test Loss", "Radical Test Loss"],
                  "Loss", "Iterations",
                  list_ewmas = [0, 0, 0])

    grapher.graph([test_acc[0][0], test_acc[1][0], test_acc[2][0]],
                  [test_acc[0][1], test_acc[1][1], test_acc[2][1]],
                  ["r", "g", "b"],
                  ["Single Test Accuracy", "Double Test Accuracy", "Radical Test Accuracy"],
                  "Accuracy", "Iterations",
                  list_ewmas = [0, 0, 0])

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
        return np.array(values), np.array(indices)


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

def graph_single():
    grapher = Grapher()

    values1, indices1 = grapher.textlog2numpy("train_losses.txt")
    values2, indices2 = grapher.textlog2numpy("val_losses.txt")


    grapher.graph([values1, values2],
                  [indices1, indices2],
                  ["b", "g"],
                  ["Training Loss", "Validation Loss"],
                  "Loss", "Iterations",
                  list_ewmas = [0.999, 0.999])


def graph_all():
    grapher = Grapher()
    t_values = []
    t_indices = []
    v_values = []
    v_indices = []
    offsets = [0, 700000, 1000000, 1500000]
    for i, folder in enumerate(["700k", "300k", "500k", "250k"]):
        train, train_ind = grapher.textlog2numpy(folder + "/train_losses.txt")
        val, val_ind = grapher.textlog2numpy(folder + "/val_losses.txt")
        t_values += [train]
        t_indices += [train_ind + offsets[i]]
        v_values += [val]
        v_indices += [val_ind + offsets[i]]

    t_values = np.concatenate(t_values, axis=0)
    t_indices = np.concatenate(t_indices, axis=0)
    v_values = np.concatenate(v_values, axis=0)
    v_indices = np.concatenate(v_indices, axis=0)

    grapher.graph([t_values, v_values],
                  [t_indices, v_indices],
                  ["b", "g"],
                  ["TCN Training Loss", "TCN Validation Loss"],
                  "Loss", "Iterations",
                  list_ewmas = [0.999, 0.999])

if __name__ == "__main__":
    graph_all()

import matplotlib.pyplot as plt

def losses_to_ewma(losses, alpha = 0.9):
    losses_ewma = []
    ewma = losses[0]
    for loss in losses:
        ewma = alpha*ewma + (1-alpha)*loss
        losses_ewma += [ewma]
    return losses_ewma


def graph(fcc, cnn):
    plt.figure(1)

    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    
    plt.plot(cnn.x_indices, cnn.losses, "r", label="CNN Training Loss")
    plt.plot(cnn.val_x_indices, cnn.val_losses, "r--", label="CNN Validation Loss")
    plt.plot(fcc.x_indices, fcc.losses, "g", label="FCC Training Loss")
    plt.plot(fcc.val_x_indices, fcc.val_losses, "g--", label="FCC Validation Loss")
    plt.legend(loc=1)

    plt.figure(2)

    plt.ylabel("Accuracy")
    plt.xlabel("Iterations")
    
    plt.plot(cnn.acc_indices, cnn.acc, "r", label="CNN Accuracy")
    plt.plot(fcc.acc_indices, fcc.acc, "g", label="FCC Accuracy")
    plt.legend(loc=1)

    plt.show()

def receptive_field(k, d):
    return d*(k-1)

if __name__ == "__main__":
    n_layers = 18
    r = 0
    for i in range(n_layers-1, -1, -1):
        dil = 1 + 3*i
        print(i)
        #dil = 2**i
        r += receptive_field(k=3, d=dil)
        if i != n_layers-1:
            r -= 1
    print(r)

    n_layers = 18
    r = 0
    for i in range(n_layers-1, -1, -1):
        dil = 1 + 2*i
        print(i)
        #dil = 2**i
        r += 2* receptive_field(k=3, d=dil)
        if i != n_layers-1:
            r -= 2
    print(r)

import torch
import torch.nn as nn

import numpy as np

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(size_average=False, reduce=True)
    

    def forward(self, batch_pred, captions, num_words):
        #batch_pred size [batch_size*seq, vocab_size, num_words]
        #captions  size [batch_size, seq, num_words]
        #num_words size [batch_size, seq]
        batch_size, seq_size, time_size = captions.size()
        captions = captions.view(-1, time_size)
        num_words = num_words.reshape(-1)
        #values, indices = batch_pred.topk(1, dim=1)
        #indices = indices.squeeze(1)

        loss = 0
        for i, n_words in enumerate(num_words):
            loss += self.cross_entropy(batch_pred[i, :, :n_words].t(), captions[i, :n_words])

        total_num_words = float(np.sum(num_words))
        return loss / total_num_words

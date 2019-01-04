import numpy as np


def accuracy(output, labels, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = labels.shape(0)

    topk_labels = np.argsort(output, axis=1)[:, :maxk]
    correct = np.zeros_like(topk_labels)
    for i in batch_size:
        correct[i, topk_labels[i, :].index(labels[i])] = 1

    res = []
    for k in topk:
        correct_k = np.sum(correct[:, :k])
        res.append(correct_k * 100.0 / batch_size)
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
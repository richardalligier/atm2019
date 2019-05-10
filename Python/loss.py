import torch


def batch_loss(device, featuretarget, xy, output, loss):
    '''utility function to compute the loss of a batch'''
    target = featuretarget.target(xy).to(device)
    weight = featuretarget.weights(xy).to(device)
    n = weight.sum(0)
    err = loss(output, target) * weight
    loss = err.sum(0)
    return loss, n


_mseloss = torch.nn.MSELoss(reduction='none')


def mtlLoss(device, featuretarget, xy, output):
    ''' MSE loss on the first output (the output is a tuple of 2 tensor)'''
    def mseloss(output, target):
        meanoutput, sig2output = output
        return _mseloss(meanoutput, target)
    return batch_loss(device, featuretarget, xy, output, mseloss)


def pnllNormal(device, featuretarget, xy, output):
    '''predictive negative log-likelihood '''
    def pnll(output, target):
        meanoutput, sig2output = output
        return (meanoutput-target)**2/sig2output+torch.log(sig2output)
    return batch_loss(device, featuretarget, xy, output, pnll)

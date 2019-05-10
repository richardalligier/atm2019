import torch
from torch import nn
import torch.utils.data as data_utils
from datautils import TensorDataLoader as tensorDataLoader
import pandas as pd
import numpy as np
import pickle
import logging
from config import DATA_PATH, TRAIN, VALID, TEST, vargroupfloat, vargroupcat, choosevar
import argparse
import train
import predictionproblem

NUM_THREAD = 0
BATCH_SIZE = 8192


def fargs():
    parser = argparse.ArgumentParser(
        description='generates "predicted.csv" files')
    parser.add_argument('-foldermodel', default=None, type=str,
                        help="folder containing all the models")
    parser.add_argument('-fileout', default=None, type=str,
                        help="file name of the out file, typically $(DATA_PATH)/predicted.csv")
    parser.add_argument('-cpu', help='device',
                        action='store_true', default=False)
    return parser

# load the model from the pickle file 'filename'
def loadmodel(filename):
    with open(filename, 'rb') as f:
        m = pickle.load(f)
    return m

# compute the predictive negative log-likelihood
def pnlGauss(e, sig2):
    return np.nanmean((e**2/sig2+np.log(sig2)+np.log(2*np.pi)) / 2, 0)


# compute the predicted mean and sigma2 for one model 'result'
def predict(device, result, test_loader):
    lmean = []
    lsig2 = []
    with torch.no_grad():
        for xy in test_loader:
            mean, sig2 = predictionproblem.predict(
                device, result.featuretarget, xy, result.model)
            mean = result.featuretarget.normalOutput.inverse_transform(
                mean.cpu())
            sig2 = sig2 * result.featuretarget.std ** 2
            lmean.append(mean)
            lsig2.append(sig2.cpu())
    return np.concatenate(lmean, 0), np.concatenate(lsig2, 0)


# compute the predicted mean and sigma2 using the ensemble of models in 'results'
def computepred(device, y, results, test_loader):
    for result in results:
        result.model = result.model.to(device)
        result.model = result.model.eval()

    mus = []
    sig2s = []
    for result in results:
        mu, sig2 = predict(device, result, test_loader)
        e = mu-y
        print(pnlGauss(e, sig2))
        mus.append(mu)
        sig2s.append(sig2)

    mus = np.array(mus)
    print(mus.shape)
    sig2s = np.array(sig2s)
    mu = mus.mean(0)
    print(mu.shape)
    sig2 = ((sig2s+mus**2).mean(0)-mu**2)
    e = mu-y
    png = pnlGauss(e, sig2)
    print(png, np.sum(png))
    print(np.sqrt(np.nanmean(e**2, 0)))
    mudf = pd.DataFrame(
        mu, columns=["pred"+v for v in results[0].featuretarget.yvars])
    sig2df = pd.DataFrame(
        sig2, columns=["sig2"+v for v in results[0].featuretarget.yvars])
    return pd.concat([mudf, sig2df], axis=1)


def main():
    parser = fargs()
    args = parser.parse_args()
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(args.foldermodel)
    results = [loadmodel(args.foldermodel+"/model"+str(i)+".pkl")
               for i in range(12)]
    acft_type = results[0].acft_type
    print(acft_type)
    test = train.loadcsv(acft_type, TEST)
    test_set = data_utils.TensorDataset(*results[0].featuretarget.build(test))
    test_loader = tensorDataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREAD, pin_memory=True)
    y = test[results[0].featuretarget.yvars]
    df = computepred(device, y, results, test_loader)
    df.to_csv(args.fileout, index=False)


if __name__ == '__main__':
    main()

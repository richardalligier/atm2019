from config import DATA_PATH, TRAIN, VALID, TEST, vargroupfloat, vargroupcat, choosevar
from datautils import EarlyStop, CsvLog, str2bool, Infinitecycle, TensorDataLoader as tensorDataLoader
from torch import nn
import torch
import time
import gc
import itertools
import numpy as np
import pandas as pd
import pickle
import os
import argparse

from loss import pnllNormal, mtlLoss
import predictionproblem

DEBUG = False
ALLVARS = ["massFutur", "target_cas1", "target_cas2", "target_Mach"]


def yvars(args):
    ''' returns a list of target variables'''
    return ALLVARS if args.target == "all" else [args.target]


class TrainResult:
    '''Class used to embedd the saved model'''

    def __init__(self, acft_type, dicoemb, featuretarget, model):
        self.acft_type = acft_type
        self.dicoemb = dicoemb
        self.featuretarget = featuretarget
        self.model = model


def fargs():
    parents = [predictionproblem.fargs()]
    parser = argparse.ArgumentParser(
        description='train a predictive model.', parents=parents)
    parser.add_argument('-model', help='aircraft model', default="DH8D")
    parser.add_argument('-pca', help='pca', action='store_true', default=False)
    parser.add_argument('-cpu', help='device',
                        action='store_true', default=False)
    parser.add_argument('-niteration', help='niteration',
                        default=5000, type=int)
    parser.add_argument('-rate', help='learning rate decay',
                        default=1., type=float)
    parser.add_argument('-target', help='target', default="all")
    parser.add_argument('-seed', help='seed', default=0, type=int)
    parser.add_argument('-filelog', help='filelog', default="log", type=str)
    parser.add_argument('-finalmodel', action='store_true', default=False)
    parser.add_argument('-tolbest', default=0.1, type=float)
    parser.add_argument('-xvars', default="abcdeimopstz", type=str)
    parser.add_argument('-save_model', default=None, type=str)
    parser.add_argument('-batch_size', default=512, type=int)
    parser.add_argument("-freqvalid", default=15000, type=int)
    return parser


def loadcsv(model, setname):
    ''' load csv trajectories file, filters it and returns a pandas dataframe'''
    filename = "{}_{}.csv.xz".format(model, setname)
    fullfilename = os.path.join(DATA_PATH, "foldedtrajs", filename)
    if os.path.exists(fullfilename+".pkl"):
        df = pickle.load(open(fullfilename+".pkl", "rb"))
    else:
        usecols = choosevar(vargroupfloat, "".join(vargroupfloat.keys()))+choosevar(vargroupcat, "".join(vargroupcat.keys()))+[
            "massFutur", "target_Mach", "target_cas1", "target_cas2", "maxtimestep", "timestep", "segment", "mseEnergyRateFutur"]
        df = predictionproblem.loadcsv(fullfilename, usecols)
        pickle.dump(df, open(fullfilename+".pkl", "wb"), protocol=4)
    df = df.query('maxtimestep>=timestep+600').query("timestep>=135")
    df = df.reset_index(drop=True)
    return df


def load_data(args, share_memory=False):
    ''' load the training and validation set, and initialize the prediction problem'''
    XVARS = args.xvars
    MODEL = args.model
    if args.finalmodel:
        ts = loadcsv(MODEL, TRAIN)
        ts = pd.concat([ts, loadcsv(MODEL, VALID)],
                       ignore_index=True).reset_index(drop=True)
    else:
        ts = loadcsv(MODEL, TRAIN)
    gc.collect()
    print(ts.shape)
    varxfloat = choosevar(vargroupfloat, XVARS)
    varxcat = choosevar(vargroupcat, XVARS)

    varx = varxfloat+varxcat
    for v in varxcat:
        print(v, np.sum(ts[v] == 0)/ts.shape[0])
    featuretarget = predictionproblem.PredictionProblem(
        varxfloat, varxcat, yvars(args), ts)
    train_set = torch.utils.data.TensorDataset(*featuretarget.build(ts))
    del ts
    gc.collect()
    if args.finalmodel:
        vs = loadcsv(MODEL, TEST)
    else:
        vs = loadcsv(MODEL, VALID)
    valid_set = torch.utils.data.TensorDataset(*featuretarget.build(vs))
    del vs

    print(gc.collect())
    if share_memory:
        for s in [train_set,  valid_set]:
            for x in s.tensors:
                x.share_memory_()
    return featuretarget, train_set, valid_set


def performtraining(args, device, data, criterion, best=None):
    ''' perform training steps'''
    criterionprint = mtlLoss
    featuretarget, train_set, valid_set = data
    train_loader, trainvalid_loader, valid_loader = data_loader(
        args, train_set, valid_set)
    torchmseLoss = torch.nn.MSELoss(reduction="elementwise_mean").to(device)
    model, dicoembedd, formatx, modelfinal = predictionproblem.create_model(
        device, args, featuretarget, trainvalid_loader, len(yvars(args)))
    optimizer = predictionproblem.create_optimizer(args, formatx, modelfinal)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=args.rate)
    earlystop = EarlyStop(20)
    dargs = sorted(vars(args).items())
    logging = CsvLog(args.filelog)
    logging.add2line(map(lambda x: x[0], dargs))
    colnameset = "test" if args.finalmodel else "valid"
    logging.add2line(["countbackprop", "time", "stop", colnameset])
    logging.add2line([s+y for s in ["train", "valid"] for y in ALLVARS])
    logging.writeline()

    nbatch_in_train = len(train_loader)//args.batch_size
    freqvalid = args.niteration if args.finalmodel else min(
        nbatch_in_train, args.freqvalid)
    countbackprop = 0
    countvalid = 0
    sets = [valid_loader]
    train_loader_cycle = Infinitecycle(train_loader)

    def rmse(x):
        return torch.sqrt(x)*featuretarget.std
    start = time.time()
    while countbackprop < args.niteration:
        startepoch = time.perf_counter()
        nmax = min(freqvalid, args.niteration-countbackprop)
        predictionproblem.train(device, featuretarget, train_loader_cycle, model,
                                criterion, optimizer, scheduler, niter=nmax)
        countbackprop += nmax
        endepoch = time.perf_counter()
        print(endepoch-startepoch)
        with torch.no_grad():
            losses = [predictionproblem.valid_loss(
                device, featuretarget, s, model, criterion) for s in sets]
            lossesprint = [predictionproblem.valid_loss(
                device, featuretarget, s, model, criterionprint) for s in sets]
        countvalid += 1
        convertedloss = [rmse(loss).cpu() for loss in lossesprint]
        meanlosses = [loss.mean().item() for loss in losses]
        losses = [[x.item() for x in loss.cpu()] for loss in losses]
        preambule = "iter {:d} seed {:d}".format(countbackprop, args.seed)
        print(preambule, convertedloss, losses, meanlosses, sep="\n")

        def writeline(stop=""):
            # if len(losses) == 3 else losses+[losses[-1]]
            losseswrite = losses
            logging.add2line(map(lambda x: x[1], dargs))
            logging.add2line([countbackprop, time.time()-start,
                              stop, meanlosses[0], meanlosses[-1], meanlosses[-1]])
            logging.add2line(itertools.chain.from_iterable(losseswrite))
            logging.writeline()

        if not args.finalmodel:
            score = meanlosses[-1]
            if earlystop.step(score):
                writeline("earlystop")
                break
            if best is not None:
                if best[countvalid-1] + 2*args.tolbest < score:
                    writeline("bad")
                    break
                if countvalid-1 >= 4 and best[countvalid-1] + args.tolbest < score:
                    writeline("bad")
                    break
                best[countvalid-1] = min(best[countvalid-1], score)
        writeline()
    logging.close()
    return dicoembedd, model


def data_loader(args, train_set, valid_set):
    ''' Builds the data_loader from the sets'''
    pin_memory = True
    num_workers = 0
    train_loader = tensorDataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    trainvalid_loader = tensorDataLoader(train_set, batch_size=8192,
                                         shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = tensorDataLoader(valid_set, batch_size=8192,
                                    shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, trainvalid_loader, valid_loader


def train(device, data, args, best=None):
    '''train the network and save it if necessary'''
    start = time.time()
    print("args.seed", args.seed)
    torch.set_num_threads(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    featuretarget, train_set, valid_set = data
    train_loader, trainvalid_loader, valid_loader = data_loader(
        args, train_set, valid_set)
    criterion = pnllNormal
    featuretarget.std = featuretarget.std.to(device)
    if args.initrate is None:
        args.initrate = predictionproblem.search_initrate(
            args, device, featuretarget, train_loader, trainvalid_loader, len(yvars(args)), criterion)
    print("initial weight", args.initrate)
    dicoembedd, model = performtraining(args, device, data, criterion, best)
    if args.save_model is not None:
        print("saving model:", args.save_model)
        with open(args.save_model, "wb") as f:
            pickle.dump(TrainResult(
                args.model, dicoembedd, featuretarget, model), f)
    end = time.time()
    print(end-start)


def main():
    parser = fargs()
    args = parser.parse_args()
    device = torch.device(
        "cuda" if not args.cpu and torch.cuda.is_available() else "cpu")
    print("device used ", device)
    data = load_data(args)
    train(device, data, args)


if __name__ == '__main__':
    main()

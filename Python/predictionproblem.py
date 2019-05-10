import torch
from adamw import AdamW
import numpy as np
import argparse
import copy
from torch import nn
import sklearn
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from datautils import Infinitecycle, get_rng_state, set_rng_state
from block import build_FCN, Map, Softplusmin, Unsqueeze, Squeeze, Sum, Concat, Split


def fargs():
    '''arguments used to build and train the network'''
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-archi', default="20_10", type=str)
    parser.add_argument('-lamb', help='lambda', default=0.0, type=float)
    parser.add_argument('-initrate', default=None, type=float)
    parser.add_argument('-residual',  action='store_true', default=False)
    parser.add_argument('-dropout', help='dropout', default=0, type=float)
    parser.add_argument('-sum_emb', default=0, type=int)
    parser.add_argument('-load_emb', action='store_true', default=False)
    parser.add_argument('-freeze_emb', action='store_true', default=False)
    parser.add_argument('-save_emb', action='store_true', default=False)
    parser.add_argument('-lambda_emb', default=0., type=float)
    parser.add_argument('-dropout2d', action='store_true', default=False)
    parser.add_argument('-dropout_emb', default=0., type=float)
    parser.add_argument("-selectinitrate", default=0.97, type=float)
    return parser


VARINT64 = ["time", "segment", "callsign", "modeltype",
            "icao24", "fromICAO", "toICAO", "operator"]
VARINT16 = ["n_cas1", "n_cas2", "n_mach"]


def loadcsv(filename, usecols, nrows=None, skiprows=None):
    '''load csv trajectories file and returns a pandas dataframe'''
    dicttype = dict((v, np.float32) for v in usecols)
    for v in VARINT64:
        if v in usecols:
            dicttype[v] = np.int64
    for v in VARINT16:
        if v in usecols:
            dicttype[v] = np.int32
    df = pd.read_csv(filename, usecols=usecols, dtype=dicttype,
                     engine='c', sep=',', nrows=nrows, skiprows=skiprows)
    print(df.info())
    return df


class ImputerMedian(sklearn.base.BaseEstimator,sklearn.base.TransformerMixin):
    '''my custom median imputer, I had memory issues with sklearn.preprocessing.SimpleImputer(strategy="median",copy=False)'''
    def transform(self, X, **kwargs):
        Xn = np.copy(X)#X.copy()
        for col in range(Xn.shape[1]):
            mask = np.isnan(Xn[:,col])
            if np.sum(mask)>0:
                print("np.argmax(mask)",np.argmax(mask))
                print(col,np.sum(mask))
            Xn[mask,col] = self.statistics_[col]
        return Xn
    def fit(self,X, y=None, **kwargs):
        self.statistics_ = np.nanmedian(X, axis=0)
        return self

def _get_missing(X):
    features = {}
    for col in range(X.shape[1]):
        mask_nan = np.isnan(X[:,col])
        if np.sum(mask_nan)>0:
            print(col,np.sum(mask_nan))
        if np.any(mask_nan):
            features[col]=mask_nan
    return features

class Missing(sklearn.base.BaseEstimator,sklearn.base.TransformerMixin):
    '''my custom MissingIndicator, I had memory issues with sklearn.preprocessing.MissingIndicator()'''
    def transform(self, X, **kwargs):
        print(X.dtype)
        shape = (X.shape[0],len(self.features_))
        res = np.zeros(shape,dtype=X.dtype)
        dmissing = _get_missing(X)
        for col, mask_nan in dmissing.items():
            if col in self.features_:
                res[:,self.features_.index(col)] = mask_nan
            else:
                raise Exception("{} not identified as containing missing values during the fit ({} were identified)".format(col, self.features_))
        return res

    def fit(self,X, y=None, **kwargs):
        print("fitting missing\n")
        self.features_ = list(_get_missing(X).keys())
        print(self.features_)
        return self


class PredictionProblem:
    '''specifying the problem to tackle'''

    def __init__(self, varxfloat, varxcat, yvars, ts):
        missing = Missing()
        median = ImputerMedian()
        transformerImpute_list = [
            ('indicators', missing),
            ('features', median),
        ]

        transformerImpute = FeatureUnion(
            transformer_list=transformerImpute_list)
        transformer = [('impute', transformerImpute),
                       # ('debug',Debug()),
                       ('scale', StandardScaler(copy=True))
        ]
        self.transfo = Pipeline(steps=transformer)
        self.varxfloat = varxfloat[:]
        print(varxfloat)
        self.varxcat = varxcat
        self.yvars = yvars
        self.diconcat = {v: int(ts[v].max()) for v in varxcat}
        self.nvarxfloatwithimpute = len(self.varxfloat)
        if self.varxfloat != []:
            x = ts.loc[:, self.varxfloat].to_numpy(copy=False)#.values#.copy()
            self.transfo.fit(x)
            self.nvarxfloatwithimpute += len(missing.features_)
        self.normalOutput = StandardScaler()
        self.normalOutput.fit(ts.loc[:, yvars].values)
        self.std = torch.from_numpy(self.normalOutput.scale_).float()

    def build(self, ts):
        weights_np = np.logical_not(
            np.isnan(ts.loc[:, self.yvars].values)).astype(np.float32)
        weights = torch.from_numpy(weights_np)
        target_np = self.normalOutput.transform(ts.loc[:, self.yvars].values)
        np.nan_to_num(target_np, copy=False)
        target = torch.from_numpy(target_np)
        if self.varxfloat != []:
            featuresfloat = ts.loc[:, self.varxfloat].values
            featuresfloat = self.transfo.transform(featuresfloat)
            featuresfloat = torch.from_numpy(featuresfloat)
        else:
            featuresfloat = None
        if self.varxcat == []:
            return target, weights, featuresfloat
        else:
            featurescat = ts.loc[:, self.varxcat].values
            for i, name in enumerate(self.varxcat):
                lines = featurescat[:, i] > self.diconcat[name]
                featurescat[lines, i] = 0
                featurescat = torch.LongTensor(featurescat)
            if self.varxfloat == []:
                return target, weights, featurescat
            else:
                return target, weights, featurescat, featuresfloat

    def xcat(self, x):
        if self.varxcat == []:
            return tuple()
        else:
            return tuple(x[2][:, i].contiguous() for i in range(x[2].shape[-1]))

    def weights(self, x):
        return x[1].contiguous()

    def xfloat(self, x):
        if self.varxfloat == []:
            return tuple()
        else:
            return (x[-1].contiguous(),)

    def x(self, x):
        return self.xfloat(x)+self.xcat(x)

    def target(self, x):
        return x[0].contiguous()


def archi(args):
    '''parse the architecture argument'''
    return list(map(int, args.archi.split("_")))


def create_optimizer(args, formatx, modelfinal):
    '''create the optimizer'''
    if args.freeze_emb:
        optimizer = AdamW(modelfinal.parameters(), lr=args.initrate,
                          weight_decay=args.lamb)
    else:
        optimizer = AdamW([{'params': modelfinal.parameters(), 'lr': args.initrate, 'weight_decay': args.lamb}]+[
                          {'params': formatx.parameters(), 'lr': args.initrate, 'weight_decay': args.lambda_emb}])
    return optimizer


def create_dicoembedd(args, featuretarget, train_loader):
    '''create the embeddings'''
    def getdim(v):
        if args.sum_emb != 0:
            return args.sum_emb
        else:
            n = int((featuretarget.diconcat[v]+1)**0.25)
            return min(max(n, 2), 10)  # max(n,2)
    if args.load_emb:
        dicoembedd = pickle.load(open("emb.pkl", "rb"))
    else:
        dicoembedd = {v: torch.nn.Embedding(featuretarget.diconcat[v]+1, getdim(
            v), sparse=False, max_norm=10., padding_idx=0, scale_grad_by_freq=False) for v in featuretarget.varxcat}
        if archi(args) != [0]:
            for emb in dicoembedd.values():
                emb.weight.data.zero_()
    return dicoembedd


def init_weights(layer):
    '''initialize the weights'''
    if type(layer) == nn.Linear:
        a, b = layer.parameters()
        nn.init.kaiming_uniform_(a, a=0.001)
        nn.init.constant_(b, 0.1)


def init_bias_final(len_yvars):
    '''initialize the weights of the final layer'''
    def f(layer):
        if type(layer) == nn.Linear and layer.out_features == 2*len_yvars:
            a, b = layer.parameters()
            with torch.no_grad():
                nn.init.constant_(b, 0.)
                nn.init.constant_(a, 0.)
    return f


def create_model(device, args, featuretarget, train_loader, len_yvars):
    '''build the network'''
    dicoembedd = create_dicoembedd(args, featuretarget, train_loader)
    if args.sum_emb == 0:
        xdim = featuretarget.nvarxfloatwithimpute + \
            sum(dicoembedd[v].embedding_dim for v in featuretarget.varxcat)
    else:
        xdim = featuretarget.nvarxfloatwithimpute + \
            min(dicoembedd[v].embedding_dim for v in featuretarget.varxcat)

    def getdrop(xcat):
        return args.dropout_emb  # Exception("drop"+xcat)

    def f(xcat):
        if args.dropout_emb == 0.:  # or xcat in ["dayofweek","icao24"]:
            return dicoembedd[xcat]
        else:
            if args.dropout2d:
                return nn.Sequential(dicoembedd[xcat], Unsqueeze(), nn.Dropout2d(getdrop(xcat)), Squeeze())
            else:
                return nn.Sequential(dicoembedd[xcat], nn.Dropout(getdrop(xcat)))

    clamp = Map([Softplusmin(1e-6, alpha=1)], start=1)
    modelfinal = build_FCN(archi(args), args.dropout,
                           xdim, 2*len_yvars, args.residual)
    formatx = Map([f(xcat) for xcat in featuretarget.varxcat],
                  start=1 if featuretarget.varxfloat != [] else None)
    if args.sum_emb != 0:
        formatx = nn.Sequential(formatx, Sum(start=1))
    if featuretarget.varxcat == []:
        model = nn.Sequential(Concat(), modelfinal, Split(len_yvars), clamp)
    else:
        model = nn.Sequential(formatx, Concat(), modelfinal,
                              Split(len_yvars), clamp)
    model.apply(init_weights)
    model.apply(init_bias_final(len_yvars))
    model = model.to(device)
    return model, dicoembedd, formatx, modelfinal


def predict(device, featuretarget, xy, model):
    '''compute the prediction'''
    x = featuretarget.x(xy)
    x_var = tuple(xi.to(device) for xi in x)
    return model(x_var)


def train(device, featuretarget, train_loader_iterable, model, criterion, optimizer, scheduler, niter, do_after_backward_before_opti=()):
    '''perform training steps'''
    model = model.train()
    train_loader_iter = iter(train_loader_iterable)
    for i in range(niter):  # for i, xy in enumerate(train_loader_iter):
        xy = next(train_loader_iter)
        with torch.autograd.set_detect_anomaly(i < 1):
            output = predict(device, featuretarget, xy, model)
            loss, n = criterion(device, featuretarget, xy, output)
            loss = loss.sum()/n.sum()
            optimizer.zero_grad()
            loss.backward()
            for f in do_after_backward_before_opti:
                f(loss)
            optimizer.step()
            scheduler.step()


def valid_loss(device, featuretarget, train_loader,  model, criterion):
    '''compute the loss in a validation context'''
    model = model.eval()
    s = 0
    n = 0
    for i, xy in enumerate(train_loader):
        output = predict(device, featuretarget, xy, model)
        dloss, dn = criterion(device, featuretarget, xy, output)
        s += dloss
        n += dn
    return s/n


def smooth(args, ltrain):
    '''utility function for the search the initial learning rate'''
    l = []
    xsmooth = ltrain[0]
    alpha = 0.01
    for x in ltrain:
        xsmooth = (1-alpha)*xsmooth + alpha * x
        l.append(xsmooth)
    return np.array(l)


def isolate(args, ltrain):
    '''utility function for the search the initial learning rate'''
    low = ltrain[0]
    high = np.nanmin(ltrain)
    lscore = low-ltrain
    return lscore >= (low-high) * args.selectinitrate


def indexlasttrue(high):
    '''utility function for the search the initial learning rate'''
    n = high.shape[0]
    for i in range(n-1, -1, -1):
        if high[i]:
            return i
    return None


def search_initrate(args, device, featuretarget, train_loader, trainvalid_loader, len_yvars, criterion):
    '''search the initial learning rate using the ideas of Leslie N. Smith https://arxiv.org/abs/1506.01186'''
    print("searching initial learning rate")

    def learning_rate(batch):
        return 10**(-5+batch/1000)
    nmax = 6 * 1000
    res = []

    def collect_losses(loss):
        res.append(loss.item())
    args = copy.deepcopy(args)
    args.initrate = 1
    rng_state = get_rng_state()

    model, dicoembedd, formatx, modelfinal = create_model(
        device, args, featuretarget, trainvalid_loader, len_yvars)
    optimizer = create_optimizer(args, formatx, modelfinal)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=learning_rate)
    epoch = 0
    train_loader_cycle = Infinitecycle(train_loader)
    train(device, featuretarget, train_loader_cycle, model, criterion,
          optimizer, scheduler, niter=nmax, do_after_backward_before_opti=(collect_losses,))
    smoothed = smooth(args, np.array(res))
    interval = isolate(args, smoothed)
    ibest = np.array(smoothed).argmin()
    ilow = interval.argmax()
    ihigh = indexlasttrue(interval)
    set_rng_state(rng_state)
    print("end learning rate search:", ilow, ibest, ihigh)
    print("ihigh:", ihigh)
    return learning_rate(ihigh)

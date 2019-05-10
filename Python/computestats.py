import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pickle
from config import DATA_PATH, GBM_RES, TEST, FIGURE_FOLDER, TABLE_FOLDER, resize_figure
import train
import scipy
import seaborn as sns
from constants import FEET2METER, KTS2MS
import atmos
import vitesse
from scipy import stats

ALLV = ["massFutur", "target_cas1", "target_cas2", "target_Mach"]

# name in dataframe => lateX name
axisname = {"massFutur": "mass", "target_cas1": r"\mathrm{cas}_1", "target_cas2": r"\mathrm{cas}_2", "target_Mach": "Mach", "timestep": "t",
            "cas": "\mathrm{cas}", "ener": "\mathrm{energy rate}", "baroaltitudekalman": "H_p", "baroaltitudeanalysis": "H_p", "dbaroaltitudeanalysis": "ROCD"}

# name in dataframe => unit name
axisunit = {"massFutur": "kg", "target_cas1": "kt", "target_cas2": "kt", "target_Mach": "-", "timestep": "s", "cas": "kt",
            "ener": "W/kg", "baroaltitudekalman": "ft", "baroaltitudeanalysis": "ft", "dbaroaltitudeanalysis": "ft/min"}


def unit(v):
    '''unit conversions'''
    if v in ["cas", "target_cas1", "target_cas2"]:
        return 1 / KTS2MS
    elif v in ["baroaltitudekalman", "baroaltitudeanalysis"]:
        return 1 / FEET2METER
    elif v in ["dbaroaltitudeanalysis"]:
        return 60. / FEET2METER
    else:
        return 1


def form(v):
    '''how many digits after the decimal point to display'''
    return "{:.0f}" if "massFutur" == v else ("{:.3g}" if "target_Mach" == v else "{:.2f}")


def getpathpred(model, variables="abcdeimopstz"):
    '''build the path to the predicted.csv file'''
    return os.path.join(DATA_PATH, model, 'atm2019all', variables, "predicted.csv")


class Exp:
    def __init__(self, model, method, df):
        self.model = model
        self.method = method
        self.df = df


def loadnn(model):
    '''' load values predicted by the neural network '''
    pred = pd.read_csv(getpathpred(model))
    nn = train.loadcsv(model, TEST)
    nn = pd.concat([nn, pred], axis=1)
    for v in ALLV:
        nn["prob"+v] = np.exp(-(nn[v].values-nn["pred"+v].values)**2 /
                              nn["sig2"+v].values*0.5)/np.sqrt(nn["sig2"+v].values*2*np.pi)
        naivesig2 = np.nanstd(nn[v]-nn["pred"+v])**2
        nn["sig2"+v] = np.minimum(nn["sig2"+v], float('inf'))  # naivesig2*20)
    print(nn.shape)
    return Exp(model, method="NN", df=nn)


def loadgbm(model):
    ''' load values predicted by GBM (TRC2018) '''
    fullfilename = DATA_PATH+"/foldedtrajs/{}_test.csv.xz".format(model)
    if os.path.exists(fullfilename+".pkl"):
        bf = pickle.load(open(fullfilename+".pkl", "rb"))
    else:
        bf = pd.read_csv(fullfilename, usecols=[
                         "maxtimestep", "timestep", "mseEnergyRateFutur"]+ALLV)
        pickle.dump(bf, open(fullfilename+".pkl", "wb"), protocol=4)
    for v in ALLV:
        gbmdf = pd.read_csv(
            GBM_RES + "/{}/{}/abdemopst/gbmpredicted.csv".format(model, v))
        bf["pred"+v] = gbmdf["pred"+v].values
        print(np.sqrt(np.mean((bf[v]-bf["pred"+v])**2)))
    bf = bf.query(
        'maxtimestep>=timestep+600').query("timestep>=135").reset_index(drop=True)
    for v in ALLV:
        bf["sig2"+v] = np.nanmean((bf[v]-bf["pred"+v])**2)
    print(bf.shape)
    return Exp(model, method="GBM", df=bf)


def table(crit, unit, form, dmodel, fileout):
    '''build a table'''
    vname = {"massFutur": "$\mathrm{mass}$ [kg]", "target_cas1": "$\mathrm{cas}_1$ [kt]",
             "target_cas2": "$\mathrm{cas}_2$ [kt]", "target_Mach": "Mach [-]"}
    l = []
    methods = sorted(dmodel.keys())
    for v in ALLV:
        print(v)
        pdfs = [compute_df(crit, unit(v), form(v), dmodel[method], v)
                for method in methods]
        rdf = pd.concat(pdfs, axis=1, sort=True)
        rdf = pd.DataFrame(data=np.array(rdf.values), columns=[
                           (vname[v], method) for method in methods], index=rdf.index)
        l.append(rdf)
    rdf = pd.concat(l, axis=1)
    rdf.columns = pd.MultiIndex.from_tuples(
        rdf.columns, names=['factor', 'method'])
    with open(fileout, 'w') as f:
        f.write(rdf.to_latex(multirow=True, escape=False))
    return rdf


def compute_df(crit, unit, form, lmodel, v):
    '''compute a DataFrame used to build the table for a specific method'''
    data = []
    for e in lmodel:
        print(e.model)
        df = e.df
        rmse = unit*crit(e, v)
        data.append([form.format(rmse)])
    outdf = pd.DataFrame(data=data, columns=[lmodel[0].method], index=[
                         e.model for e in lmodel])
    return outdf


def critrmse(e, v):
    ''' compute the RMSE for non nan values (not enough points in the future or for instance, cas1 phase not observed)'''
    df = e.df
    return np.sqrt(np.nanmean((df["pred"+v]-df[v])**2))


def critpicp(p):
    ''' compute the PICP for a percentage [p] '''
    z = stats.norm.ppf((1+np.array([p]))/2)[0]

    def f(e, v):
        y = e.df[v]
        mu = e.df["pred"+v]
        sig2 = e.df["sig2"+v]
        inside = np.abs(y-mu) <= z*np.sqrt(sig2)
        return np.mean(inside[y == y])
    return f


def critmis(p):
    ''' compute the Mean Interval Size of the prediction interval of coverage [p]'''
    z = stats.norm.ppf((1+np.array([p]))/2)[0]
    print(z)

    def f(e, v):
        y = e.df[v].values
        size = 2*z*np.sqrt(e.df["sig2"+v].values)
        return np.mean(size[y == y])
    return f


def critmissamecp(dmodel, p):
    ''' compute the Mean Interval Size of the prediction interval of coverage [p] if the input is the neural network method. Otherwise, it will return the size of the interval required to match the actual observed coverage (PICP). Thus for GBM, it will return the size of the interval required to have the same coverage percentage as NN '''
    z = stats.norm.ppf((1+np.array([p]))/2)[0]

    def find(c, l):
        for x in l:
            if c(x):
                return x
        raise Exception("element not found")

    def f(e, v):
        if e.method == "NN":
            return critmis(p)(e, v)
        else:
            cp = critpicp(p)(
                find(lambda m: m.model == e.model, dmodel["NN"]), v)
            y = e.df[v]
            return 2*np.percentile(np.abs(e.df[v]-e.df["pred"+v])[y == y], cp*100)
    return f


def plotsigxrmse(e, v):
    ''' Plot the actual as a function of the predicted rmse. We have a sliding window containing 1% of the data. This sliding window slides along the examples sorted according the predicted sigma. The x will be the mean of the predicted sigma on this 1% examples and the y will be RMSE on this same 1%. '''
    df = e.df[[v, "pred"+v, "sig2"+v]].copy(deep=True)
    df = df.query(v+"=="+v)
    df.sort_values("sig2"+v, inplace=True)
    print("std", e.model, v, np.std(
        df[v].values-df["pred"+v].values), np.mean(np.sqrt(df["sig2"+v].values)))
    x = []
    y = []
    n = df.shape[0]//100
    err = (df[v].values-df["pred"+v].values)**2
    sig = np.sqrt(df["sig2"+v].values)
    print(np.std(sig))

    def rmse(e):
        return np.sqrt(np.mean(e))
    for i in range(0, df.shape[0]-n+1):
        x.append((sig[i]+sig[i+n-1])/2)
        y.append(rmse(err[i:i+n]))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={'hspace': 0})
    identity_line = np.linspace(max(min(x), min(y)), min(max(x), max(y)))
    ax1.step(x, y)
    ax1.plot(identity_line, identity_line, color="red",
             linestyle="dashed", linewidth=1.0)
    nbins = min(df.shape[0]//200, 1000)
    print(nbins)
    ax2.hist(sig, bins=nbins, density=True)
    ax2.set_xlabel('$\sigma_{'+axisname[v]+'}$ ['+axisunit[v]+']')
    ax2.set_ylabel('density [-]')
    ax1.set_ylabel(
        r'$RMSE\left(S\left({\sigma_{'+axisname[v]+r'}}\right)\right)$ ['+axisunit[v]+']')
    ax1.legend(
        (r'$RMSE\left(S\left({\sigma_{'+axisname[v]+r'}}\right)\right)$', '$y=x$'), loc='lower right')
    fig.align_ylabels((ax1,ax2))
    resize_figure(fig, 0.78)
    filename = ("plotsigxrmse"+e.model+v+".pdf").replace("_", "")
    plt.savefig(os.path.join(FIGURE_FOLDER, filename),
                format="pdf", dpi=300)  # plt.show()
    plt.close()


def critkurt(e, v):
    '''compute the kurtosis'''
    df = e.df
    sel = df[v] == df[v]
    return stats.kurtosis(((df["pred"+v]-df[v])/np.sqrt(df["sig2"+v]))[sel])


def tablegamma(ps,crit,unit,form,modelsnn,fileout):
    vname={"massFutur":"$\mathrm{mass}$ [\%]","target_cas1":"$\mathrm{cas}_1$ [\%]","target_cas2":"$\mathrm{cas}_2$ [\%]","target_Mach":"Mach [\%]"}
    l=[]
    methods = ps
    dmodel={}
    for method in methods:
        print("method",method)
        dmodel[method]=[]
        for e in modelsnn:
            dmodel[method].append(Exp(e.model,method,e.df))
    for v in ALLV:
        print(v)
        pdfs=[compute_df(crit,unit(v),form(v), dmodel[method], v) for method in methods]
        rdf=pd.concat(pdfs,axis=1,sort=True)
        rdf=pd.DataFrame(data=np.array(rdf.values), columns=[(vname[v],method) for method in methods], index=rdf.index)
        l.append(rdf)
    rdf=pd.concat(l,axis=1)
    rdf.columns = pd.MultiIndex.from_tuples( rdf.columns, names=['factor', '$\gamma$'])
    with open(fileout,'w') as f:
        f.write(rdf.to_latex(multirow=True, escape=False))


def critpicpall(e,v):
    return critpicp(float(e.method))(e,v)


def dotables():
    lmodel = sorted(['A320','E190','E195','DH8D','B737','CRJ9','A332','B77W','A319','A321','B738'])
    dmodel = {}
    dmodel["NN"] = [loadnn(model) for model in lmodel]
    dmodel["GBM"] = [loadgbm(model) for model in lmodel]
    table(critrmse, unit, form, dmodel, TABLE_FOLDER+"/table3.tex")
    table(critkurt, lambda _: 1,
          lambda _: "{:.2f}", dmodel, TABLE_FOLDER+"/tablekurtosis.tex")
    tablegamma(["0.90","0.95"],critpicpall,lambda _:100,lambda _:"{:.1f}",dmodel["NN"],TABLE_FOLDER+"/table4.tex")
    p = 0.9
    table(critmissamecp(dmodel, p), unit, form, dmodel, TABLE_FOLDER+"/table5.tex")


def dofigures():
    e = loadnn("A320")
    for v in ALLV:
        plotsigxrmse(e, v)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-which', type=str)
    args = parser.parse_args()
    lwhich = (("tables", dotables), ("figures", dofigures),)
    done = False
    for (s, f) in lwhich:
        if args.which == s:
            f()
            done = True
    if not done:
        raise Exception(
            "-which accept only {}".format(tuple(s for s, _ in lwhich)))

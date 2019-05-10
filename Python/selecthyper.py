import pandas as pd
import numpy as np
from config import DATA_PATH
import argparse
import os



def add_idparam(df):
    n=0
    l=[]
    oldniteration = -1
    for i in range(df.shape[0]):
        if oldniteration>df.niteration[i]:
            n+=1
        oldniteration=df.niteration[i]
        l.append(n)
    df['idparam']=np.array(l)

def load_dir(folder):
    l=[]
    for filename in os.listdir(folder):
#        print(filename)
        l.append(pd.read_csv(os.path.join(folder,filename)))
    df=pd.concat(l,ignore_index=True)
    add_idparam(df)
    return df

def fargs():
    parser = argparse.ArgumentParser(description='batchtrain predictive models.')
    parser.add_argument('-loghyperparameters',default=None,type=str)
#    parser.add_argument('-fileout',default=None,type=str)
    parser.add_argument('-seed',default=None,type=str)
    return parser

#def build_archi(df):
#    l = [ str(int(df["archi"+str(i)])) for i in range(int(df["nlayers"]))]
#    return ",".join(l)

if __name__=='__main__':
    parser = fargs()
    args = parser.parse_args()
#    folder = os.path.join(DATA_PATH, args.model, "all/abcdeimopstz/loghyperparameters")
    df = load_dir(args.loghyperparameters)#folder)
    df = df.loc[df.valid.idxmin()]
#    print(df.shape)
    d = {}
#    d["-archi"] = df["archi"]
#    d["-amsgrad"] = ""
    d["-dropout2d"] = ""
#    d["-clipped"] = ""
    synonim={"countbackprop":"niteration"}
    convert={"niteration":int,"sum_emb":int}
    for s in ["archi","dropout","dropout_emb","lamb","lambda_emb","countbackprop","rate","sum_emb","xvars","initrate"]:
        option = synonim.get(s,s)
        d["-" + option] = convert.get(option,lambda x:x)(df[s])
    for seed in range(12):
        print(" ".join([k+" "+str(d[k]) for k in d]),"-seed",seed)


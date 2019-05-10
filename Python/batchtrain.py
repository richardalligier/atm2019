import argparse
import train
import torch
import os
import torch.multiprocessing as mp
import pandas as pd
def read_args_batch(filename):
    l=[]
    with open(filename,'r') as f:
        for line in f:
            parser=train.fargs()
            args = parser.parse_args(line.strip().split())
            l.append(args)
    return l

def check_args(l):
    for v in ["model","xvars","target","finalmodel","pca","niteration"]:
        for i,args in enumerate(l):
            if i==0:
                x=getattr(args,v)
            elif x!=getattr(args,v):
                raise Exception("check_args",v,x,getattr(args,v))

def fargs():
        parser = argparse.ArgumentParser(description='batchtrain predictive models.')
        parser.add_argument('-folderlogout',default=None,type=str)
        parser.add_argument('-foldermodelout',default=None,type=str)
        parser.add_argument('-nworker',default=3,type=int)
        parser.add_argument('-cpu', help='device', action='store_true',default=False)
        parser.add_argument('-model',default='DH8D',type=str)
        parser.add_argument('-batch', help='batch',default='paramlist')
        parser.add_argument('-xvars', help='batch',default="abcdeimosptz",type=str)
        parser.add_argument('-finalmodel',action='store_true',default=False)
        parser.add_argument('-istart', help='batch',default=None,type=int)
        parser.add_argument('-iend', help='batch',default=None,type=int)
        parser.add_argument('-resume', help='batch',action='store_true',default=False)
        return parser

def readbest(folder,epochs):
    best = torch.ones(epochs)
    l = [pd.read_csv(os.path.join(folder,fname)) for fname in os.listdir(folder) if fname.endswith(".csv")]
    l = [torch.from_numpy(x.valid.values).float() for x in l]
    l = [ torch.cat((x,torch.ones(epochs-x.shape[0])),0) for x in l]
    if l==[]:
        return torch.ones(epochs)
    else:
        return torch.min(torch.stack(l),0)[0]

def filterjobs(batchargs, ldataargs):
    l = [ (i,job) for i,job in enumerate(ldataargs) if (batchargs.istart is None or batchargs.istart <= i) and (batchargs.iend is None or i < batchargs.iend)]
    def getjobnumber(fname):
        return int(fname[:-len(".csv")])
    if batchargs.resume and not batchargs.finalmodel:
        alreadydone = [getjobnumber(fname) for fname in os.listdir(batchargs.folderlogout) if fname.endswith(".csv")]
        return [job for (i,job) in l if i not in alreadydone]
    else:
        return [job for (_,job) in l]

# B737 A332 A319 A321
def main():
    parser = fargs()
    batchargs = parser.parse_args()
    device = torch.device("cpu" if batchargs.cpu else "cuda")#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NWORKER = batchargs.nworker
    largs = read_args_batch(batchargs.batch)
    check_args(largs)
    for i,args in enumerate(largs):
        args.xvars = "".join(sorted(batchargs.xvars))
        args.model = batchargs.model
        args.cpu = batchargs.cpu
        args.filelog = "/dev/null" if batchargs.folderlogout is None else batchargs.folderlogout+"/"+str(i)+".csv"
        args.finalmodel = batchargs.finalmodel
        args.save_model = None if batchargs.foldermodelout is None else batchargs.foldermodelout+"/model"+str(i)+".pkl"
    def reorder(nworker,l):
        ls=[[] for i in range(nworker)]
        for i,x in enumerate(l):
            ls[i%nworker].append(x)
        return [x for l in ls for x in l]
#    largs=reorder(NWORKER,largs)
    best = None
    if not batchargs.finalmodel:
        best = readbest(batchargs.folderlogout,largs[0].niteration)#torch.ones(largs[0].epochs)
        best.share_memory_()
    data = train.load_data(largs[0],share_memory=True)
    ldataargs = [(device,data,args,best) for args in largs]
    ldataargs = filterjobs(batchargs, ldataargs)
    print("jobs to do:",len(ldataargs))
    with mp.Pool(NWORKER) as p:
        p.starmap(train.train, ldataargs)

if __name__ == '__main__':
    main()

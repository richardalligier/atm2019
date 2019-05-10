import numpy as np


def testrandomgen(n):
    '''generate all the random hyperparameters to test'''

    # number of layers
    nlayers = np.random.randint(1, 11, n)

    def generatearchirandom():
        a = []
        possiblehidden = list(range(10, 100, 10)) + list(range(100, 800, 100))
        for i in range(n):
            w = [np.random.randint(0, len(possiblehidden))
                 for _ in range(nlayers[i])]
            archi = "_".join([str(possiblehidden[x])
                              for x in sorted(w, reverse=True)])
            a.append(archi)
        return a
    r = {}
    # architecture of the network (number of layers and number of hidden units)
    r["-archi"] = generatearchirandom()
    r["-target"] = np.repeat("all", n)
    r["-niteration"] = np.repeat(2*10**6, n)
    r["-seed"] = np.repeat(0, n)
    # no dropout (set to 0)
    r['-dropout'] = np.repeat(0, n)
    r["-batch_size"] = np.repeat(512, n)
    # use of dropout2d
    r['-dropout2d'] = np.repeat("", n)
    # decay factor of the learning rate
    r['-rate'] = np.random.uniform(0.998, 1, n)
    # dropout rate for the embeddings
    r["-dropout_emb"] = np.random.uniform(0., 0.9, n)
    # size of the embeddings
    r["-sum_emb"] = [np.random.randint(1, 8) for _ in range(n)]
    # weight decay
    r["-lamb"] = 10 ** np.random.uniform(-10., -3, n)
    return r


def printres(d):
    res = []
    n = len(d[list(d)[0]])
    for i in range(n):
        cmd = " ".join([k + " " + str(d[k][i]) for k in d])
        res.append(cmd)
    print("\n".join(res))


if __name__ == '__main__':
    np.random.seed(0)
    # 2000 random hyperparameters but actually, the '-iend 200' option makes 'batchtrain.py' only use the first 200 hyperparameters
    printres(testrandomgen(2000))

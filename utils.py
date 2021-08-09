import numpy as np
from sklearn.preprocessing import OneHotEncoder
from astropy import stats


def normalize_biweight(x, eps=1e-10):
    median = np.median(x)
    scale = stats.biweight.biweight_scale(x)
    if np.std(x) < 1e+2 or np.isnan(scale) or scale < 1e-4:
        norm =  (x-np.mean(x))/np.std(x)
    else:
        norm = (x - median) / (scale + eps)
    return norm


def normalize(x):
    norm = lambda x: (x-np.mean(x))/np.std(x)
    return np.apply_along_axis(norm, 0, x)


def data_preprocess(df):

    def _encoding(i):
        if df.iloc[:,i].dtype == 'O' or df.iloc[:, i].dtype.name == 'category':
            tempX = df.iloc[:, i].values.reshape(-1, 1)
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(tempX)
            out = enc.transform(tempX).toarray()
        else:
            out = df.iloc[:, i].values.reshape(-1, 1)
        return out
    p = df.shape[1]
    X_encode = [_encoding(i) for i in np.arange(p)]
    return X_encode


def evaluate_binary(trueG, estG):
    TP, TN, FP, FN, FD, MD, FPMD = 0, 0, 0, 0, 0, 0, 0
    n_node = trueG.shape[0]
    for i in range(1, n_node):
        for j in range(i):
            if trueG[i, j] == 1 and trueG[j, i] == 0 and estG[i, j] == 1 and \
                    estG[j, i] == 0:
                TP += 1
            if trueG[i, j] == 0 and trueG[j, i] == 1 and estG[i, j] == 0 and \
                    estG[j, i] == 1:
                TP += 1
            if trueG[i, j] == 0 and trueG[j, i] == 0 and estG[i, j] == 0 and \
                    estG[j, i] == 0:
                TN += 1
            if trueG[i, j] == 0 and trueG[j, i] == 0 and estG[i, j] == 1 and \
                    estG[j, i] == 0:
                FP += 1
            if trueG[i, j] == 0 and trueG[j, i] == 0 and estG[i, j] == 0 and \
                    estG[j, i] == 1:
                FP += 1
            if trueG[i, j] == 1 and trueG[j, i] == 0 and estG[i, j] == 0 and \
                    estG[j, i] == 0:
                FN += 1
            if trueG[i, j] == 0 and trueG[j, i] == 1 and estG[i, j] == 0 and \
                    estG[j, i] == 0:
                FN += 1
            if trueG[i, j] == 1 and trueG[j, i] == 0 and estG[i, j] == 0 and \
                    estG[j, i] == 1:
                FD += 1
            if trueG[i, j] == 0 and trueG[j, i] == 1 and estG[i, j] == 1 and \
                    estG[j, i] == 0:
                FD += 1
            if trueG[i, j] == 0 and trueG[j, i] == 1 and estG[i, j] == 1 and \
                    estG[j, i] == 1:
                MD += 1
            if trueG[i, j] == 1 and trueG[j, i] == 0 and estG[i, j] == 1 and \
                    estG[j, i] == 1:
                MD += 1
            if trueG[i, j] == 0 and trueG[j, i] == 0 and estG[i, j] == 1 and \
                    estG[j, i] == 1:
                FPMD += 1
    if (TP + FP + FD)>0:
        Precision = TP / (TP + FP + FD)
    else:
        Precision = 0.0
    Recall = TP / sum(sum(trueG))

    if (TP + FN + FD) > 0:
        Recall1 = TP / (TP + FN + FD)
    else:
        Recall1 = 0.0
    SHD = sum(sum((trueG != estG) | np.transpose((trueG != estG)))) / 2
    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'FD': FD, 'MD': MD,
            'FPMD': FPMD,'Precision': Precision, 'Recall': Recall,
            'Recall_NOMD': Recall1, 'SHD': SHD}


def skeleton_metrics(trueG, estG):
    TP,TN,FP,FN = 0,0,0,0
    n = trueG.shape[0]
    for i in range(n):
        for j in range(i):
            if trueG[i, j] == 1 or trueG[j, i] == 1:
                if estG[i, j] != 0 or estG[j, i] != 0:
                    TP += 1
                else:
                    FN += 1
            else:
                if estG[i, j] == 0 and estG[j, i] == 0:
                    TN += 1
                else:
                    FP += 1
    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}


def check_connect_skel(skeleton, i, j):
    depth = set.union(set(np.where(skeleton[:,i]==1)[0]),
                      set(np.where(skeleton[i,:]==1)[0]))
    checked = depth
    while depth:
        if j in depth:
            return True
        next = {}
        for k in depth:
            next = set.union(next, set.union(set(np.where(skeleton[:,k]==1)[0]),
                                             set(np.where(skeleton[k,:]==1)[0])))
        depth = set.difference(next, checked)
        checked = set.union(checked, depth)
    return False


def reachable(dag, fr, to):
    depth = set(np.where(dag[fr,:]==1)[0])
    checked = depth
    while depth:
        if to in depth:
            return True
        next = set()
        for k in depth:
            next = set.union(next, set(np.where(dag[k,:]==1)[0]))
        depth = set.difference(next, checked)
        checked = set.union(checked, depth)
    return False

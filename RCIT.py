import numpy as np
import math
import random
from utils import normalize
from scipy.spatial.distance import pdist
from scipy.linalg import cholesky, solve_triangular
from rpy2.robjects.numpy2ri import numpy2rpy

import rpy2.robjects.packages as rpackages
momentchi2 = rpackages.importr('momentchi2')


def rff_mixed(x, num_f=10):
    if len(x.shape) == 1:
        x = x.reshape(-1,1)

    cat_idx = []
    for i in range(x.shape[1]):
        if len(np.unique(x[:,i])) == 2:
            cat_idx.append(i)
    x_disc = x[:, cat_idx]
    x_cont = x[:, np.setdiff1d(range(x.shape[1]), cat_idx)]

    n_disc, n_cont = np.sum(x_disc[0, :]), x_cont.shape[1]
    num_f_cont = int(n_cont/(n_disc+n_cont)*num_f)
    num_f_disc = num_f - num_f_cont

    if x_cont.shape[1] != 0:
        r, c = x_cont.shape
        r1 = min(r, 500)

        sigma = np.median(pdist(x_cont[:r1, :], "euclidean"))
        if sigma == 0 or np.isnan(sigma):
            sigma = 1

        w = (1 / sigma) * np.random.normal(0, 1, size=num_f_cont * c).reshape(num_f_cont, c)
        b = 2 * math.pi * np.random.uniform(0, 1, size=num_f_cont)
        b = np.tile(b.reshape(-1, 1), (1, r))
        cont_feat = np.sqrt(2) * np.cos(np.dot(w, x_cont.T) + b).T

    if x_disc.shape[1] != 0:
        r, c = x_disc.shape
        r1 = min(r, 500)

        sigma = 1

        w = (1 / sigma) * np.random.normal(0, 1, size=num_f_disc * c).reshape(num_f_disc, c)
        b = 2 * math.pi * np.random.uniform(0, 1, size=num_f_disc)
        b = np.tile(b.reshape(-1, 1), (1, r))
        disc_feat = np.sqrt(2) * np.cos(np.dot(w, x_disc.T) + b).T

    if x_cont.shape[1]!=0 and x_disc.shape[1] != 0:
        return np.concatenate([cont_feat,disc_feat],axis=1)
    elif x_cont.shape[1] != 0:
        return cont_feat
    else:
        return disc_feat


def RIT_core(four_x, four_y, r, num_f):
    Cxy = np.cov(four_x.T,four_y.T)[:-num_f, -num_f:]
    Sta = r*np.sum(Cxy*Cxy)

    res_x = four_x - np.tile(np.mean(four_x,0),(r,1))
    res_y = four_y - np.tile(np.mean(four_y,0),(r,1))

    d = np.array([(x, y) for x in range(four_x.shape[1]) for y in range(four_y.shape[1])])
    res = res_x[:,d[:,1]]*res_y[:,d[:,0]]
    Cov = 1/r * np.dot(res.T,res)

    eig_d = np.linalg.eig(Cov)
    eig_d = eig_d[0][eig_d[0].imag==0]
    eig_d = eig_d[eig_d.real > 0]
    eig_d = eig_d.real

    try:
        p = 1- momentchi2.lpb4(numpy2rpy(eig_d), Sta.item())[0]
    except:
        p = 1 - momentchi2.hbe(numpy2rpy(eig_d), Sta.item())[0]

    p = max(p,np.exp(-40))
    return p


def RCIT_core(four_x, four_y, four_z, r, num_f, num_f2):
    Cxy = np.cov(four_x.T,four_y.T)[:-num_f2, -num_f2:]
    Cxz = np.cov(four_x.T,four_z.T)[:-num_f, -num_f:]
    Czy = np.cov(four_z.T,four_y.T)[:-num_f2, -num_f2:]

    Czz = np.cov(four_z.T)
    Lzz = cholesky(Czz + np.eye(num_f)*(1e-10), lower=True)
    A = solve_triangular(Lzz, Cxz.T, lower=True)
    e_x_z = np.dot(four_z, solve_triangular(Lzz.T, A, lower=False))

    A = solve_triangular(Lzz, Czy, lower=True)
    B = solve_triangular(Lzz.T, A, lower=False)
    e_y_z = np.dot(four_z, B)

    res_x = four_x - e_x_z
    res_y = four_y - e_y_z

    Cxy_z = Cxy - np.dot(Cxz, B)
    Sta = r*np.sum(Cxy_z*Cxy_z)

    d = np.array([(x, y) for x in range(four_x.shape[1]) for y in range(four_y.shape[1])])
    res = res_x[:,d[:,1]]*res_y[:,d[:,0]]
    Cov = 1/r * np.dot(res.T,res)

    eig_d = np.linalg.eig(Cov)
    eig_d = eig_d[0][eig_d[0].imag==0]
    eig_d = eig_d[eig_d.real > 0]
    eig_d = eig_d.real

    try:
        p = 1- momentchi2.lpb4(numpy2rpy(eig_d), Sta.item())[0]
    except:
        p = 1 - momentchi2.hbe(numpy2rpy(eig_d), Sta.item())[0]

    p = max(p,np.exp(-40))
    return p


class RCITIndepTest(object):
    def __init__(self, suffStat,  down=False, num_f=100, num_f2=10):
        self.n, self.c = suffStat[0].shape[0], len(suffStat)
        if not down:
            self.suffStat = suffStat
            self.r = self.n
        else:
            self.idx = random.sample(range(self.n), self.c * 100) \
                if self.c * 100 < self.n  else np.array(range(self.n))
            self.suffStat = []
            for i in range(len(suffStat)):
                self.suffStat.append(suffStat[i][self.idx])
            self.r = len(self.idx)
        # keep the fft transmation
        self.fft_feature_f2 = {}
        self.fft_feature_f = {}
        self.num_f = num_f
        self.num_f2 = num_f2

    def fit(self, x, y, z=None, **kwargs):
        if z is None or len(z) == 0:
            if x not in self.fft_feature_f2:
                self.fft_feature_f2[x] = normalize(
                    rff_mixed(self.suffStat[x], num_f=self.num_f2))
            if y not in self.fft_feature_f2:
                self.fft_feature_f2[y] = normalize(
                    rff_mixed(self.suffStat[y], num_f=self.num_f2))
            return RIT_core(four_x=self.fft_feature_f2[x],
                            four_y=self.fft_feature_f2[y],
                            r=self.r, num_f=self.num_f2)
        else:
            # print(x, y, z)
            if x not in self.fft_feature_f2:
                self.fft_feature_f2[x] = normalize(
                    rff_mixed(self.suffStat[x], num_f=self.num_f2))
            y = frozenset([y] + z)
            if y not in self.fft_feature_f2:
                suffStaty = np.concatenate([self.suffStat[i]
                                            for i in y], axis=1)
                self.fft_feature_f2[y] = normalize(
                    rff_mixed(suffStaty, num_f=self.num_f2))
            setz = frozenset(z)
            if setz not in self.fft_feature_f:
                suffStatz = np.concatenate([self.suffStat[i]
                                            for i in setz], axis=1)
                self.fft_feature_f[setz] = normalize(
                    rff_mixed(suffStatz, num_f=self.num_f))
            return RCIT_core(four_x=self.fft_feature_f2[x],
                             four_y=self.fft_feature_f2[y],
                             four_z=self.fft_feature_f[setz],
                             r=self.r, num_f=self.num_f, num_f2=self.num_f2)

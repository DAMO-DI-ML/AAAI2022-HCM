import itertools
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings("ignore")


def greedy_edgeadding(df, X_encode, selMat, maxNumParents,base_model_para,
                      alpha=0.0,
                      cv_split=5, ll_type='local',
                      prior_adj=None, prior_anc=None,
                      score_type='bic', debug=False):
    # first encode the prior knowledge in selMat, remove no directed edge
    if prior_adj is not None:
        if prior_adj.shape != selMat.shape:
            raise ValueError("the shape of prior_adj is not same as selMat")
        selMat = selMat & (prior_adj >= 0)
        must_have = np.argwhere(prior_adj > 0)
    else:
        must_have = []

    if prior_anc is not None:
        if prior_anc.shape != selMat.shape:
            raise ValueError("the shape of prior_anc is not same as selMat")
        selMat = selMat & (prior_anc >= 0)
        not_prior_anc = prior_anc < 0
        np.fill_diagonal(not_prior_anc, False)
    else:
        not_prior_anc = np.zeros(selMat.shape, dtype=bool)

    path = np.zeros(selMat.shape)
    np.fill_diagonal(path, 1)
    Adj = np.zeros(selMat.shape)

    ScoreMatComputer = ScoreMatCompute(
        df, X_encode, selMat,
        maxNumParents=maxNumParents,
        cv_split=cv_split,
        ll_type=ll_type,
        score_type=score_type,
        debug=debug,
        base_model_para=base_model_para)

    # initialize scoreMat
    scoreMat, scoreNodes = ScoreMatComputer.initialScoreMat()
    # Greedily adding edges
    while np.max(scoreMat) > -float('inf'):
        diff = scoreMat-np.transpose(scoreMat)
        # the difference between two -inf is set to -inf
        diff[np.isnan(diff)] = -float('inf')
        # the difference between non -inf and -inf is set to 0 to avoid wrong
        # assign when we have none symmetric puring in previous step
        diff[np.isinf(diff)] = 0.0
        weighted_gain_diff = (1 - alpha) * scoreMat + alpha * diff

        # if we have some edges must be add
        if len(must_have) > 0:
            mind = np.argmax([weighted_gain_diff[tuple(i)] for i in must_have])
            row_index, col_index = must_have[mind][0], must_have[mind][1]
            must_have = np.delete(must_have, mind, axis=0)
        else:
            # Find the best edge
            row_index, col_index = np.unravel_index(
                weighted_gain_diff.argmax(), weighted_gain_diff.shape)

        # We should now consider whether add (row_index, col_index)
        # will avoid the cause order
        t_path = path.copy()
        t_path[row_index, col_index] = 1
        DescOfNewChild = np.append(np.where(t_path[col_index,:]==1), col_index)
        AncOfNewParent = np.append(np.where(t_path[:,row_index]==1), row_index)
        for element in list(itertools.product(AncOfNewParent, DescOfNewChild)):
            t_path[element] = 1

        # if has some avoid then change do not include the edge and
        # set the score to -inf
        if np.any(not_prior_anc & (t_path == 1)):
            scoreMat[row_index, col_index] = -float('inf')
            continue
        else:
            if debug:
                print(f"before add the edge ({row_index, col_index}), the "
                      f"score of {col_index} is {scoreNodes[col_index]}")

            scoreNodes[col_index] = (scoreNodes[col_index] +
                                     scoreMat[row_index, col_index])

            if debug:
                print(f"after the score is {scoreNodes[col_index]}")
                print(scoreNodes)

            ScoreMatComputer.set_scoreNodes(scoreNodes)
            scoreMat[row_index, col_index] = -float('inf')
            scoreMat[col_index, row_index] = -float('inf')
            Adj[row_index, col_index] = 1
            path = t_path.copy()
            scoreMat[np.transpose(path) == 1] = -float('inf')
            # update the scoreMat
            ScoreMatComputer.set_scoreMat(scoreMat)
            scoreMat, scoreNodes = ScoreMatComputer.scoreupdate(
                Adj=Adj, j=col_index)
    return Adj


def compute_init_ll(x_col, bandwidth=1.0, kernel='gaussian'):
    """
    calculate the log likelihood of each variable without model
    :param x_col: a pandas data series
    :return: log likelihood
    """
    if x_col.dtypes == 'O' or x_col.dtypes == 'bool':
        prob_dic = x_col.value_counts(normalize=True).to_dict()
        prob_list = x_col.replace(prob_dic)
        return np.mean(np.log(prob_list))
    else:
        data_x = x_col.values
        kde = gaussian_kde(data_x)
        logprob = np.log(kde.evaluate(data_x))
        """
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        kde.fit(data_x[:, None])

        # score_samples returns the log of the probability density
        logprob = kde.score_samples(data_x[:, None])
        """
        return np.mean(logprob)


class ScoreMatCompute(object):
    def __init__(self, X, X_encode, selMat, maxNumParents, base_model_para,
                 cv_split=5,
                 ll_type='local', score_type='bic', debug=False):
        self.X = X
        self.X_encode = X_encode
        self.selMat = selMat
        self.p = selMat.shape[0]
        self.maxNumParents = maxNumParents
        self.valid_pair = np.argwhere(selMat)
        self.scoreMat = np.ones(selMat.shape) * (-float('inf'))
        # scoreNodes is the log(p(x)) of each variable
        self.scoreNodes = (self.X).apply(compute_init_ll, axis=0).values
        self.score_type = score_type
        self.debug = debug
        self.Dn = X.shape[0]
        self.pn = X.shape[1]
        self.bicterm = np.log(self.Dn) / self.Dn / 2
        self.base_model_para = base_model_para
        base_model = base_model_para['base_model']
        if self.debug:
            print("score of each variable without add any edge")
            print(self.scoreNodes)

        if base_model == 'lgbm':
            from lgbm_cvmodel import ModelWrapper
        elif base_model == 'gam':
            from gam_cvmodel import ModelWrapper
        else:
            raise NotImplementedError(
                f"currently we only support 'lgbm' and 'gam'.")
        self.ModelWrapper = ModelWrapper
        self.cv_split = cv_split
        self.ll_type = ll_type

    def set_scoreMat(self, scoreMat):
        self.scoreMat = scoreMat

    def set_scoreNodes(self, scoreNodes):
        self.scoreNodes = scoreNodes

    def _compute_ll(self, x):
        Y = self.X.iloc[:, x[1]]
        if isinstance(self.X_encode, list):
            model = self.ModelWrapper(X=self.X_encode[x[0]], Y=Y,
                                      cv_split=self.cv_split,
                                      ll_type=self.ll_type,
                                      para=self.base_model_para
                                      )
        elif isinstance(self.X_encode, pd.DataFrame):
            X_input = self.X_encode.iloc[:, [x[0]]]
            model = self.ModelWrapper(X=X_input, Y=Y, cv_split=self.cv_split,
                ll_type=self.ll_type,para=self.base_model_para)
        else:
            raise ValueError("The type of X_encode must be list of numpy "
                             "array or pandas DataFrame")

        ll, edof = model.fit()
        if edof == 0:
            edof = 1
        #print(ll, edof, self.bicterm, x[0], x[1])
        if self.score_type == 'll':
            self.scoreMat[x[0], x[1]] = ll
        elif self.score_type == 'bic':
            self.scoreMat[x[0], x[1]] = ll - edof*self.bicterm
        elif self.score_type == 'aic':
            self.scoreMat[x[0], x[1]] = ll - edof/self.Dn

    def initialScoreMat(self):
        np.apply_along_axis(self._compute_ll, axis=1, arr=self.valid_pair)
        # currently the self.scoreMat is the score of each model
        if self.debug:
            print("score of each variable when adding the first edge")
            print(self.scoreMat)
        self.scoreMat = self.scoreMat - self.scoreNodes
        # currently the self.scoreMat is the difference between score of
        # adding the edge and not add the edge
        if self.debug:
            print("score improve of each variable when adding the first edge")
            print(self.scoreMat)
        return self.scoreMat, self.scoreNodes

    def _update_ll(self, x):
        Y = self.X.iloc[:, x[-1]]
        if isinstance(self.X_encode, list):
            X_input = np.concatenate([self.X_encode[i] for i in x[:-1]],
                                     axis=1)
        elif isinstance(self.X_encode, pd.DataFrame):
            X_input = self.X_encode.iloc[:, x[:-1]]
        else:
            raise ValueError("The type of X_encode must be list of numpy "
                             "array or pandas DataFrame")
        model2 = self.ModelWrapper(X=X_input, Y=Y, cv_split=self.cv_split,
                                   ll_type=self.ll_type,
                                   para=self.base_model_para)

        ll, edof = model2.fit()
        if edof==0:
            edof = len(x)-2
        if self.score_type == 'll':
            self.scoreMat[x[-2], x[-1]] = ll
        elif self.score_type == 'bic':
            self.scoreMat[x[-2], x[-1]] = ll - edof*self.bicterm
        elif self.score_type == 'aic':
            self.scoreMat[x[-2], x[-1]] = ll - edof/self.Dn

    def _fillninf(self, x):
        self.scoreMat[x[-2], x[-1]] = -float('inf')

    def scoreupdate(self, Adj, j):
        existingParOfJ = np.where(Adj[:, j] == 1)[0]
        notAllowedParOfJ = np.setdiff1d(
            np.where(self.scoreMat[:, j] == -float('inf'))[0],
            np.append(existingParOfJ, [j]))
        if len(existingParOfJ) + len(notAllowedParOfJ) < self.p:
            # get the index of undecided candidate
            toUpdate = np.setdiff1d(np.arange(self.p), np.concatenate(
                (existingParOfJ, notAllowedParOfJ, [j])))
            update_need = np.concatenate(
                (
                np.tile(existingParOfJ, (len(toUpdate), 1)),  # existingParOfJ
                toUpdate.reshape(-1, 1),  # candidate to add
                np.tile(j, (len(toUpdate), 1))  # target
                )
                , axis=1)
            if update_need.shape[0] > 0:
                if len(existingParOfJ) < self.maxNumParents:
                    np.apply_along_axis(self._update_ll, axis=1,
                                        arr=update_need)
                else:
                    np.apply_along_axis(self._fillninf, axis=1,
                                        arr=update_need)
                if self.debug:
                    print("the score matrix after adding an edge")
                    print(self.scoreMat)
                self.scoreMat[:, j] = self.scoreMat[:, j] - self.scoreNodes[j]
                if self.debug:
                    print(
                        "score improve of each variable when adding an edge")
                    print(self.scoreMat)
        return self.scoreMat, self.scoreNodes

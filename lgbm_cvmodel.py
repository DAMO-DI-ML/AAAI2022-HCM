import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
from sklearn.model_selection import StratifiedKFold, KFold


class LgbClsModel(object):
    def __init__(self,
                 is_unbalance='true',
                 boosting='gbdt',
                 num_leaves=31,
                 feature_fraction=0.5,
                 learning_rate=0.05,
                 num_boost_round=20,
                 num_class=2,
                 early_stopping_round=3,
                 bagging_fraction=0.5,
                 bagging_freq=20
                 ):
        self.parameters = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'is_unbalance': is_unbalance,
            'boosting': boosting,
            'num_leaves': num_leaves,
            'feature_fraction': feature_fraction,
            'learning_rate': learning_rate,
            'num_class': num_class,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'verbose': -1
        }
        self.num_boost_round = num_boost_round
        self.early_stopping_round = early_stopping_round
        self.model = None

    def fit(self, X_train=None, y_train=None, X_test=None, y_test=None):
        lgb_train = lgb.Dataset(X_train, y_train)
        test_data = lgb.Dataset(X_test, label=y_test)

        self.model = lgb.train(
            self.parameters,
            lgb_train,
            valid_sets=test_data,
            verbose_eval=False,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_round
        )
        y_pred = self.model.predict(X_test)
        #  log_loss is the negative log-likelihood of a logistic model
        return -log_loss(y_true=y_test, y_pred=y_pred, normalize=False)


class LgbRegModel(object):
    def __init__(self,
                 boosting='gbdt',
                 num_leaves=31,
                 feature_fraction=0.5,
                 learning_rate=0.05,
                 num_boost_round=20,
                 bandwidth=0.4,
                 kernel='gaussian',
                 early_stopping_round=3,
                 bagging_fraction=0.5,
                 bagging_freq=20
                 ):
        self.parameters = {
            'objective': 'regression',
            'boosting': boosting,
            'num_leaves': num_leaves,
            'feature_fraction': feature_fraction,
            'learning_rate': learning_rate,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'verbose': -1
        }
        self.early_stopping_round = early_stopping_round
        self.num_boost_round = num_boost_round
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.model = None

    def fit(self, X_train=None, y_train=None, X_test=None, y_test=None):
        lgb_train = lgb.Dataset(X_train, y_train)
        test_data = lgb.Dataset(X_test, label=y_test)
        self.model = lgb.train(
            self.parameters,
            lgb_train,
            valid_sets=test_data,
            verbose_eval=False,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_round
        )

        y_pred = self.model.predict(X_test)
        residule = y_test - y_pred

        kde = gaussian_kde(residule)
        logprob = np.log(kde.evaluate(residule))

        return residule, np.sum(logprob)


class ModelWrapper(object):
    def __init__(self,
                 X,
                 Y,
                 para=None,
                 cv_split=5,
                 ll_type='local'
                 ):
        """
        :param X:
        :param Y: Y is a pandas data series
        :param is_unbalance:
        :param boosting:
        :param num_leaves:
        :param feature_fraction:
        :param learning_rate:
        :param num_boost_round:
        """
        boosting = para['boosting']
        num_leaves = para['num_leaves']
        feature_fraction = para['feature_fraction']
        learning_rate = para['learning_rate']
        num_boost_round = para['num_boost_round']

        self.X = X
        self.Y = Y
        if (Y.dtypes == 'O' or Y.dtypes == 'bool' or
                Y.dtype.name == 'category' or Y.dtypes == 'int'):
            num_class = len(Y.unique())
            self.pred_model = LgbClsModel(
                boosting=boosting,
                num_leaves=num_leaves,
                feature_fraction=feature_fraction,
                learning_rate=learning_rate,
                num_boost_round=num_boost_round,
                num_class=num_class
            )
        else:
            self.pred_model = LgbRegModel(
                boosting=boosting,
                num_leaves=num_leaves,
                feature_fraction=feature_fraction,
                learning_rate=learning_rate,
                num_boost_round=num_boost_round
            )
        self.fited = False
        self.cv_split = cv_split
        self.ll_type = ll_type

    def fit(self):
        n_split = self.cv_split
        ll_type = self.ll_type
        total_ll = 0
        total_num = 0
        if (self.Y.dtypes == 'O' or self.Y.dtypes == 'bool'
                or self.Y.dtype.name == 'category'):
            le = preprocessing.LabelEncoder()
            le.fit(self.Y)
            self.Y = le.transform(self.Y)
            if n_split == 0:
                sumll = self.pred_model.fit(X_train=self.X, y_train=self.Y,
                                            X_test=self.X, y_test=self.Y)
                total_ll += sumll
                total_num += len(self.Y)
            else:
                skf = StratifiedKFold(n_splits=n_split)
                skf.get_n_splits(self.X, self.Y)
                for train_ind, test_ind in skf.split(self.X, self.Y):
                    X_train, X_test = self.X[train_ind], self.X[test_ind]
                    y_train, y_test = self.Y[train_ind], self.Y[test_ind]
                    sumll = self.pred_model.fit(X_train=X_train, y_train=y_train,
                                                X_test=X_test, y_test=y_test)
                    total_ll += sumll
                    total_num += len(y_test)
            return total_ll/total_num, 0
        else:
            residule = np.array([])
            if n_split == 0:
                presidule, sumll = self.pred_model.fit(X_train=self.X, y_train=self.Y, X_test=self.X, y_test=self.Y)
                residule = np.append(residule, presidule)
                total_ll += sumll
                total_num += len(self.Y)
            else:
                kf = KFold(n_splits=n_split)
                kf.get_n_splits(self.X)
                total_num = 0
                for train_ind, test_ind in kf.split(self.X):
                    X_train, X_test = self.X[train_ind], self.X[test_ind]
                    y_train, y_test = self.Y[train_ind], self.Y[test_ind]
                    presidule, sumll = self.pred_model.fit(
                        X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)
                    residule = np.append(residule, presidule)

                    total_ll += sumll
                    total_num += len(y_test)
            if ll_type=='local':
                return total_ll/total_num, 0
            else:
                kde = gaussian_kde(residule)
                logprob = np.log(kde.evaluate(residule))
            return np.mean(logprob), 0

DEFAULT_MODEL_PARA = {
    'step1_maxr': 1,  # the max size of conditional set in step 1
    'step3_maxr': 3,  # the max size of conditional set in step 3
    'num_f': 100,  # the number of random fft features of conditional set
    'num_f2': 10,  # the number of random fft features of test variable
    'indep_pvalue': 0.05,  # the threshold for independence test
    'alpha': 0.0,
    'll_type': 'local',  # 'local' or 'global', kow to estimate kde
    'cv_split': 5,  # number of cross validation (can take 0)
    'downsampling': False,  # weather need down sampling in MRCIT
    'maxNumParents': 10,  # number of max
    'score_type': 'bic'  # score type 'bic', 'll', 'aic'. ll: log-likelihood
}

# check for parameters in Lightgbm
DEFAULT_LGBM_PARA = {
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'learning_rate': 0.05,
    'num_boost_round': 200,
}

# check for parameters in Pygam
DEFAULT_GAM_PARA = {
    "spline_order": 10,
    "lam": 0.6,
    "n_jobs": 1,
    "use_edof": True,
}

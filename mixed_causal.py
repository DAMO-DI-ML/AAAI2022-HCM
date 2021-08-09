import pandas as pd
import numpy as np
import time
from utils import data_preprocess, evaluate_binary, \
    normalize_biweight, skeleton_metrics
from camsearch_prior import greedy_edgeadding
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn_pandas import DataFrameMapper
from skelprune import skeleton, pruning
from RCIT import RCITIndepTest

def prior_knowledge_encode(feature_names,
                           source_nodes=None, direct_edges=None,
                           not_direct_edges=None, happen_before=None):
    """
    :param feature_names: list of string, list of feature names
    :param source_nodes: list of string, list of source node
    :param direct_edges: dictionary, {start1:end1, start2:end2}
    :param not_direct_edges: dictionary, {start1:end1, start2:end2}
    :param happen_before: dictionary,
            {node: [ac1, ac2, ac3], node2: [ac1, ac2, ac3]}
    :return:
    """
    p = len(feature_names)
    feature2index = {}
    for i, feature in enumerate(feature_names):
        feature2index[feature] = i

    prior_adj = np.zeros((p, p))
    prior_anc = np.zeros((p, p))
    # the source nodes do not have any ancestor
    if source_nodes:
        source_nodes = set(source_nodes)
        for s_node in source_nodes:
            if s_node not in feature2index:
                raise ValueError(
                    f"the feature: {s_node} you provide in the source_nodes "
                    f"is not in the column names")
            prior_adj[:, feature2index[s_node]] = -1
            prior_anc[:, feature2index[s_node]] = -1
    else:
        source_nodes = set()

    # set the direct edge based on the prior knowledge
    if direct_edges:
        set_direct_edges = set()
        for start in direct_edges:
            if start not in feature2index:
                raise ValueError(
                    f"the feature: {start} you provide in the direct_edges"
                    f" is not in the column names")
            for end in direct_edges[start]:
                if end not in feature2index:
                    raise ValueError(
                        f"the feature: {end} you provide in the "
                        f"direct_edges is not in the column names")
                if end in source_nodes:
                    raise ValueError(
                        f"The prior knowledge you provide is conflict you"
                        f" claim the feature {end} is a source node but "
                        f"there is an edge to it.")
                ind_s = feature2index[start]
                ind_e = feature2index[end]
                prior_adj[ind_s, ind_e] = 1
                set_direct_edges.add((start, end))
    else:
        set_direct_edges = set()

    # set the for sure no direct edge based on the prior knowledge
    if not_direct_edges:
        for start in not_direct_edges:
            if start not in feature2index:
                raise ValueError(
                    f"the feature: {start} you provide in the "
                    f"not_direct_edges is not in the column names")
            for end in not_direct_edges[start]:
                if end not in feature2index:
                    raise ValueError(
                        f"the feature: {end} you provide in the "
                        f"not_direct_edges is not in the column names")
                if (start, end) in set_direct_edges:
                    raise ValueError(
                        f"The prior knowledge you provide is conflict please "
                        f"check the existence of edge {(start, end)}")
                ind_s = feature2index[start]
                ind_e = feature2index[end]
                prior_adj[ind_s, ind_e] = -1

    # set the for sure no ancestor based on the prior knowledge
    if happen_before:
        for late in happen_before:
            if late not in feature2index:
                raise ValueError(
                    f"the feature: {late} you provide in order information "
                    f"is not in the column names")
            for anc in happen_before[late]:
                if anc not in feature2index:
                    raise ValueError(
                        f"the feature: {anc} you provide in order information "
                        f"is not in the column names")
                if (late, anc) in set_direct_edges:
                    raise ValueError(
                        f"The prior knowledge you provide is conflict "
                        f"please check the existence of edge ({late, anc})")

                ind_s = feature2index[late]
                ind_e = feature2index[anc]
                prior_adj[ind_s, ind_e] = -1
                prior_anc[ind_s, ind_e] = -1

    return prior_adj, prior_anc


def data_processing(df, cat_index, normalize='biweight'):
    columns = df.columns
    if normalize == 'biweight':
        BiweightScaler = FunctionTransformer(normalize_biweight)
        standardize = [(col, None) if col in cat_index else
                       ([col], BiweightScaler) for col in columns]
        x_mapper = DataFrameMapper(standardize)
        df = x_mapper.fit_transform(df).astype('float32')
        df = pd.DataFrame(df, columns=columns)
    elif normalize == 'standard':
        standardize = [(col, None) if col in cat_index else
                       ([col], StandardScaler()) for col in columns]
        x_mapper = DataFrameMapper(standardize)
        df = x_mapper.fit_transform(df).astype('float32')
        df = pd.DataFrame(df, columns=columns)
    else:
        raise NotImplementedError(
            f"currently we only support 'biweight' and 'standard'.")
    # encode
    df[cat_index] = df[cat_index].astype(object)
    X_encode = data_preprocess(df)
    return df, X_encode


def mixed_causal(df, X_encode,model_para, base_model_para,
                 prior_adj=None, prior_anc=None, selMat=None):

    step1_maxr = model_para['step1_maxr']
    step3_maxr = model_para['step3_maxr']
    maxNumParents = model_para['maxNumParents']
    num_f = model_para['num_f']
    num_f2 = model_para['num_f2']
    cv_split = model_para['cv_split']
    ll_type = model_para['ll_type']
    alpha = model_para['alpha']
    downsampling= model_para['downsampling']
    indep_pvalue = model_para['indep_pvalue']
    base_model = base_model_para['base_model']
    score_type = model_para['score_type']

    p = df.shape[1]

    #######################################################################
    # step1 use pc algorithm to conduct skeleton learning
    indepTest = RCITIndepTest(suffStat=X_encode, down=downsampling,
                              num_f=num_f, num_f2=num_f2)
    if selMat is None:
        t1 = time.time()
        skel = skeleton(indepTest, labels=range(p),
                        m_max=step1_maxr, alpha=indep_pvalue,
                        priorAdj=prior_adj,
                        )
        selMat = skel['sk']

        step1_train_time = time.time() - t1
    else:
        step1_train_time = 0

    #######################################################################
    # step 2: create dag based on the greedy search
    if base_model == "lgbm":
        X = X_encode
    elif base_model =="gam":
        X = df
    else:
        raise NotImplementedError(
            f"currently we only support 'lgbm' and 'gam'.")

    t2 = time.time()
    dag2 = greedy_edgeadding(df, X, selMat,
                            maxNumParents=maxNumParents,
                            alpha=alpha,
                            cv_split=cv_split,
                            ll_type=ll_type,
                            base_model_para=base_model_para,
                            prior_adj=prior_adj,
                            prior_anc=prior_anc,
                            score_type = score_type,
                            )
    step2_train_time = time.time() - t2

    ######################################################################
    # step 3: remove edges by conditional independence test
    t3 = time.time()
    dag = pruning(indepTest, dag2, m_max=step3_maxr,
                  alpha=indep_pvalue, priorAdj=prior_adj)

    step3_train_time = time.time() - t3

    return(selMat, dag2, dag, step1_train_time,
           step2_train_time, step3_train_time)


def evaluate(trueG, skel_bool, dag2, dag):
    skel = skel_bool.astype('int')
    if skel.shape != trueG.shape != dag.shape:
        raise ValueError(f"the shape of true adjacency matrix and the "
                         f"predicted skeleton and dag is not same!")
    skl_result = pd.DataFrame(skeleton_metrics(trueG, skel), index=[0])
    dag2_result = pd.DataFrame(evaluate_binary(trueG, dag2), index=[0])
    dag_result = pd.DataFrame(evaluate_binary(trueG, dag), index=[0])
    return skl_result, dag2_result, dag_result

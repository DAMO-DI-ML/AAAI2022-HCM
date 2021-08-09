# coding: utf-8
import pandas as pd
import numpy as np
from mixed_causal import mixed_causal, \
    prior_knowledge_encode, data_processing, evaluate
import easydict

from const import DEFAULT_MODEL_PARA, DEFAULT_LGBM_PARA, DEFAULT_GAM_PARA


args = easydict.EasyDict({
    'data_file': 'alram_simulate.csv',  # location and name of data file
    'cat_index': ['2','3','4','7','8','10','11','13','16','17','18','19','21','22','27','33','36','37'],
    'true_G':'alarm.csv',  # location and name of true graph
    'model_para': {'step1_maxr': 1, 'step3_maxr': 3, 'num_f': 100,
                   'num_f2': 10, 'indep_pvalue': 0.05, 'downsampling': False,
                   'cv_split': 0, 'll_type': 'local', 'alpha': 0.0,
                   'maxNumParents': 10, 'score_type': 'll'
                  },
    # can used for test step 2's performance with different setting. i.e.,
    # if you already have skeleton file 'alaram_simulate_skl.csv'
    # then you can use it to avoid run step 1 again
    'skl_file': "",
    'base_model':'lgbm',  # 'lgbm' or 'gam'
    'base_model_para': {},
    'source_nodes': [],
    'direct_edges': {},
    'not_direct_edges': {},
    'happen_before': {},
})


def check_model_para(model_para, base_model, base_model_para):
    model_para_out = {}
    for para in DEFAULT_MODEL_PARA:
        if para in model_para:
            model_para_out[para] = model_para[para]
        else:
            model_para_out[para] = DEFAULT_MODEL_PARA[para]
    base_model_para_out = {}
    base_model_para_out['base_model'] = base_model
    if base_model=='lgbm':
        for para in DEFAULT_LGBM_PARA:
            if para in base_model_para:
                base_model_para_out[para] = base_model_para[para]
            else:
                base_model_para_out[para] = DEFAULT_LGBM_PARA[para]
    elif base_model=='gam':
        for para in DEFAULT_GAM_PARA:
            if para in base_model_para:
                base_model_para_out[para] = base_model_para[para]
            else:
                base_model_para_out[para] = DEFAULT_GAM_PARA[para]
    else:
        raise NotImplementedError(
            f"currently we only support 'lgbm' and 'gam'.")
    return model_para_out, base_model_para_out


if __name__ == '__main__':
    df = pd.read_csv(args.data_file)
    if args.skl_file == "":
        selMat = None
    else:
        selMat = pd.read_csv(args.skl_file, header=None).values > 0
    print(df.columns)
    print(df.columns.values)
    model_para_out, base_model_para_out = check_model_para(
        args.model_para, args.base_model, args.base_model_para)

    df, X_encode = data_processing(df, args.cat_index, normalize='biweight')
    prior_adj, prior_anc = prior_knowledge_encode(
        feature_names=df.columns, source_nodes=args.source_nodes,
        direct_edges=args.direct_edges, not_direct_edges=args.not_direct_edges)

    selMat, dag2, dag, step1_time, step2_time, step3_time = mixed_causal(
        df, X_encode, model_para= model_para_out,
        prior_adj=prior_adj, prior_anc=prior_anc,
        base_model_para=base_model_para_out, selMat=selMat)
    print(step1_time, step2_time, step3_time)
    np.savetxt(args.data_file[:-4]+'_skl.csv', selMat, delimiter=",")
    np.savetxt(args.data_file[:-4]+'_dag2.csv', dag2, delimiter=",")
    np.savetxt(args.data_file[:-4]+'_dag.csv', dag, delimiter=",")
    if args.true_G != '':
        trueG = pd.read_csv(args.true_G).values
        skl_result, dag2_result, dag_result = evaluate(trueG, selMat, dag2,dag)
        print(skl_result)
        print(dag2_result)
        print(dag_result)
        #print(trueG)
        #print(dag)

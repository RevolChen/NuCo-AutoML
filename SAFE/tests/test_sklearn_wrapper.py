from caafe import CAAFEClassifier # Automated Feature Engineering for tabular datasets
from tabpfn import TabPFNClassifier # Fast Automated Machine Learning method for small tabular datasets
from sklearn.ensemble import RandomForestClassifier
from caafe.run_llm_code import run_llm_code
import os
import torch
from caafe import data
from sklearn.metrics import accuracy_score,roc_auc_score, f1_score,precision_score,recall_score
from tabpfn.scripts import tabular_metrics
import openai
from functools import partial
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import random
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

# openai.api_key = 'sk-2z8MsQQbixyDGRG7Glrt3GfxFH12AbcuxHFI4Dow6g0hbEA5'
# openai.api_base = "https://api.openai-proxy.org/v1"
# metric_used = tabular_metrics.auc_metric
# # 读取数据集
# with open('data/cc1.pkl', 'rb') as f:
#     ds = pickle.load(f)
# # # 处理数据集，分为训练与预测
# # ds, df_train, df_test = data.get_data_split(ds, seed=0)
# df_train = ds[1]
# df_test = ds[2]
# # 目标列名
# target_column_name = ds[4][-1]
# # print(target_column_name)
# # 数据集描述，包含列描述
# dataset_description = ds[-1]
# # print(dataset_description)
# train_x, train_y = data.get_X_y(df_train, target_column_name)
# test_x, test_y = data.get_X_y(df_test, target_column_name)
# clf_no_feat_eng = TabPFNClassifier(device=('cuda' if torch.cuda.is_available() else 'cpu'), N_ensemble_configurations=4)
#
# clf_no_feat_eng.fit = partial(clf_no_feat_eng.fit, overwrite_warning=True)
#
# clf_no_feat_eng.fit(train_x, train_y)
# pred = clf_no_feat_eng.predict(test_x)
#
# print("Accuracy before CAAFE:", accuracy_score(test_y,pred))
# print("F1 Score before CAAFE:", f1_score(test_y,pred, average='binary'))
# print("AUC Score before CAAFE:", roc_auc_score(test_y, clf_no_feat_eng.predict_proba(test_x)[:, 1]))
# print("PRE before CAAFE:", precision_score(test_y,pred))
# print("REC before CAAFE:", recall_score(test_y,pred))
#
#  ## Setup and Run CAAFE - This will be billed to your OpenAI Account!
# # caafe生成特征后预测
#
# caafe_clf = CAAFEClassifier(base_classifier=clf_no_feat_eng,
#                             llm_model="gpt-3.5-turbo",
#                             iterations=2)
#
# caafe_clf.fit_pandas(df_train,
#                      target_column_name=target_column_name,
#                      dataset_description=dataset_description)
#
# # pred = caafe_clf.predict(df_test)
# # print("Accuracy after CAAFE:", accuracy_score(test_y,pred))
# # print("F1 Score after CAAFE:", f1_score(test_y,pred,average='weighted'))
# # print("AUC Score after CAAFE:", roc_auc_score(test_y, caafe_clf.predict_proba(df_test)[:, 1]))
# # print("PRE after CAAFE:", precision_score(test_y,pred))
# # print("REC after CAAFE:", recall_score(test_y,pred))
#
# print("Generated feature code:")
# print(caafe_clf.code)
#
# def to_pd(df_train, target_name):
#     y = df_train[target_name].astype(int)
#     x = df_train.drop(target_name, axis=1)
#
#     return x, y
#
# # 执行特征代码后的数据集
# df_train_aug = run_llm_code(caafe_clf.code, df_train)
# df_test_aug = run_llm_code(caafe_clf.code, df_test)
#
# tra_x, tra_y = to_pd(df_train_aug, target_column_name)
# te_x, te_y = to_pd(df_test_aug, target_column_name)
#
# rforest = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 可重复的随机数据划分
# param_grid = {
#     "min_samples_leaf": [0.001, 0.01, 0.05],  # 调整范围
#     "max_depth": [5, 10, None]  # 新增深度控制
# }
# gsmodel = GridSearchCV(rforest, param_grid, cv=cv, scoring='f1')
# gsmodel.fit(tra_x, tra_y)
#
# test_pred = gsmodel.predict(te_x)
# test_proba = gsmodel.predict_proba(te_x)[:, 1]  # 取正类（类别1）的概率
#
# print("Accuracy before CAAFE:", accuracy_score(te_y,test_pred))
# print("F1 Score before CAAFE:", f1_score(te_y,test_pred, average='binary'))
# print("AUC Score before CAAFE:", roc_auc_score(te_y, test_proba))
# print("PRE before CAAFE:", precision_score(te_y,test_pred))
# print("REC before CAAFE:", recall_score(te_y,test_pred))
def get_data_split(ds, seed):
    def get_df(X, y):
        df = pd.DataFrame(
            data=np.concatenate([X, np.expand_dims(y, -1)], -1), columns=ds[4]
        )
        cat_features = ds[3]
        for c in cat_features:
            if len(np.unique(df.iloc[:, c])) > 50:
                cat_features.remove(c)
                continue
            df[df.columns[c]] = df[df.columns[c]].astype("int32")
        return df.infer_objects()

    X = ds[1].numpy() if type(ds[1]) == torch.Tensor else ds[1]
    y = ds[2].numpy() if type(ds[2]) == torch.Tensor else ds[2]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )

    df_train = get_df(X_train, y_train)
    df_test = get_df(X_test, y_test)
    df_train.iloc[:, -1] = df_train.iloc[:, -1].astype("category")
    df_test.iloc[:, -1] = df_test.iloc[:, -1].astype("category")

    return ds, df_train, df_test

# 加载本地数据集
def load_origin_data(loc):
    # 读取数据集
    with open(loc, 'rb') as f:
        ds = pickle.load(f)

    df_train = ds[1]
    df_test = ds[2]
    # ds, df_train, df_test = get_data_split(ds, seed=0)

    # 目标列名
    target_column_name = ds[4][-1]
    # print(target_column_name)
    # 数据集描述，包含列描述
    dataset_description = ds[-1]
    # print(dataset_description)

    return df_train, df_test, target_column_name, dataset_description

# 随机森林下游模型算法
def RandomForest_feat(seed):
    print("# Random Forest")
    rforest = RandomForestClassifier(n_estimators=100, random_state=seed,class_weight='balanced')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)# 可重复的随机数据划分
    param_grid = {
        "min_samples_leaf": [0.001, 0.01, 0.05],  # 调整范围
        "max_depth": [5, 10, None]  # 新增深度控制
    }
    gsmodel = GridSearchCV(rforest, param_grid, cv=cv, scoring='f1')

    return gsmodel

# xgboost下游模型算法
def XGBoost_feat(seed):
    print("# XGBoost")
    xgb = XGBClassifier(random_state=seed, scale_pos_weight=1)  # XGBoost 分类器，设置随机种子

    # 定义交叉验证和参数网格
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    param_grid = {
        "learning_rate": [0.01, 0.1, 0.2],  # 学习率
        "max_depth": [3, 5, 10],             # 树的最大深度
        "n_estimators": [100, 200, 300],    # 树的数量
        "subsample": [0.8, 1.0],             # 样本采样比例
        "colsample_bytree": [0.8, 1.0],      # 特征采样比例
    }

    # 网格搜索进行交叉验证
    gsmodel = GridSearchCV(xgb, param_grid, cv=cv, scoring='f1')

    return gsmodel

# lightgbm下游模型算法
def LightGBM_feat(seed):
    print("# LightGBM")
    # 优化参数：更严格的叶子节点控制和正则化
    lgbm = LGBMClassifier(
        random_state=seed,
        class_weight='balanced',  # 处理类别不平衡
        boosting_type='gbdt',
        min_child_samples=50,  # 增加最小叶子样本数
        min_split_gain=0.1,  # 提高分裂增益阈值
        reg_alpha=0.5,  # 更强的L1正则化
        reg_lambda=0.5,  # 更强的L2正则化
        subsample=0.8,  # 默认增加采样随机性
        colsample_bytree=0.8,
        verbosity=-1  # 关闭警告输出（可选）
    )
    # 定义交叉验证和参数网格
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # 简化参数网格，避免冲突组合
    param_grid = {
        "learning_rate": [0.05, 0.1],  # 移除过小的学习率
        "max_depth": [3, 5],  # 限制树深
        "n_estimators": [100, 200],
        "num_leaves": [15, 31],  # 与max_depth匹配：num_leaves <= 2^max_depth
    }

    # 网格搜索进行交叉验证
    gsmodel = GridSearchCV(lgbm, param_grid, cv=cv, scoring='f1', n_jobs=-1)

    return gsmodel

def to_pd(df_train, target_name):
    y = df_train[target_name].astype(int)
    x = df_train.drop(target_name, axis=1)

    return x, y

# 生成特征后效果
def generate_feat_effect(
    base_classifier,
    df_train,
    df_test,
    llm_model = 'gpt-3.5-turbo',
    iterations = 10,
    target_column_name = 'class',
    dataset_description = None,
):
    clf_no_feat_eng = TabPFNClassifier(device=('cuda' if torch.cuda.is_available() else 'cpu'), N_ensemble_configurations=4)

    clf_no_feat_eng.fit = partial(clf_no_feat_eng.fit, overwrite_warning=True)

    caafe_clf = CAAFEClassifier(base_classifier=clf_no_feat_eng,
                                llm_model=llm_model,
                                iterations=iterations)
    caafe_clf.fit_pandas(df_train,
                         target_column_name=target_column_name,
                         dataset_description=dataset_description)

    test_pred = caafe_clf.predict(df_test)
    test_proba = caafe_clf.predict_proba(df_test)[:, 1]

    te_x, te_y = to_pd(df_test, target_column_name)

    # # 执行特征代码后的数据集
    # df_train_aug = run_llm_code(caafe_clf.code, df_train)
    # df_test_aug = run_llm_code(caafe_clf.code, df_test)
    #
    # tra_x, tra_y = to_pd(df_train_aug, target_column_name)
    # te_x, te_y = to_pd(df_test_aug, target_column_name)
    #
    # base_classifier.fit(tra_x,tra_y)
    # test_pred = base_classifier.predict(te_x)
    # test_proba = base_classifier.predict_proba(te_x)[:, 1]

    # 计算评估指标
    test_acc = accuracy_score(te_y, test_pred)
    test_f1 = f1_score(te_y, test_pred, average='binary')
    test_auc = roc_auc_score(te_y, test_proba)
    test_pre = precision_score(te_y, test_pred)
    test_rec = recall_score(te_y, test_pred)

    return test_acc, test_f1, test_auc, test_pre, test_rec

# 计算统计指标
def print_stats(name, values):
    print(f"{name}: {np.mean(values):.2f} ± {np.std(values):.2f}")

if __name__ == '__main__':
    # openai.api_key = 'sk-2z8MsQQbixyDGRG7Glrt3GfxFH12AbcuxHFI4Dow6g0hbEA5'
    # openai.api_base = "https://api.openai-proxy.org/v1"
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_path', default="/root/autodl-tmp/gzh/Llama2-chat-13B-Chinese-50W", type=str,
                        help='Base model路径')
    parser.add_argument('-g', '--gpus', default="0", type=str, help='GPU设置')
    parser.add_argument('-l', '--data_location', default="data/cd1.pkl", type=str, help='数据集路径')
    parser.add_argument('-s', '--default_seed', default=42, type=int, help='随机种子')
    parser.add_argument('-m', '--model', default='gpt-3.5-turbo', type=str, help='大模型')
    parser.add_argument('-i', '--iterations', default=10, type=int, help='迭代次数')
    args = parser.parse_args()

    for ds_name in ['cf2']:
    # for ds_name in ['ds_credit']:
        print(f"\n=========== Dataset {ds_name} ===========")
        # 新增：存储每次实验结果的列表
        test_acc_list = []
        test_f1_list = []
        test_auc_list = []
        test_pre_list = []
        test_rec_list = []

        loc = "data/"+ds_name+".pkl"
        for exp in range(5):  # 运行5次独立实验
            print(f"\n=========== Experiment {exp + 1}/5 ===========")
            # 每次实验使用不同的随机种子（基础种子42 + 实验编号）
            seed = args.default_seed + exp

            # 设置随机种子
            random.seed(seed)
            np.random.seed(seed)

            df_train, df_test, target_column_name, dataset_description = load_origin_data(loc)
            baseline_model = RandomForest_feat(seed)

            test_acc, test_f1, test_auc, test_pre, test_rec = generate_feat_effect(
                base_classifier=baseline_model,
                df_train=df_train,
                df_test=df_test,
                llm_model=args.model,
                iterations=args.iterations,
                target_column_name=target_column_name,
                dataset_description=dataset_description
            )
            # 存储结果
            test_acc_list.append(test_acc * 100)
            test_f1_list.append(test_f1 * 100)
            test_auc_list.append(test_auc * 100)
            test_pre_list.append(test_pre * 100)
            test_rec_list.append(test_rec * 100)
            # 打印当前实验详细结果
            print(f"\nTest  Acc: {test_acc * 100:.2f}, F1: {test_f1 * 100:.2f}, AUC: {test_auc * 100:.2f}, PRE: {test_pre*100:.2f}, REL: {test_rec*100:.2f}")

        print("\n=========== Final Statistics ===========")
        print("\n[Testing]")
        print_stats("Accuracy", test_acc_list)
        print_stats("F1 Score", test_f1_list)
        print_stats("AUC     ", test_auc_list)
        print_stats("PRE     ", test_pre_list)
        print_stats("REL     ", test_rec_list)
        print("\n")

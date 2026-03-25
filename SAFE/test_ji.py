from caafe import CAAFEClassifier, CAAFERegressor  # Automated Feature Engineering for tabular datasets
from tabpfn import TabPFNClassifier  # Fast Automated Machine Learning method for small tabular datasets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from caafe.run_llm_code import run_llm_code
import os
import torch
import time
from caafe import data
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
# from tabpfn.scripts import tabular_metrics
from functools import partial
import pickle
import csv
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
# 屏蔽Pandas的Downcasting FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning,
                        message='Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated')
# 可选：提前设置Pandas未来行为，从根源避免警告
pd.set_option('future.no_silent_downcasting', True)


# openai.api_key = 'sk-2z8MsQQbixyDGRG7Glrt3GfxFH12AbcuxHFI4Dow6g0hbEA5'
# openai.api_base = "https://api.openai-proxy.org/v1"

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
    data_rows = []
    # 读取数据集
    with open(loc, 'r') as f:
        csv_reader = csv.reader(f)
        # 获取列名（第一行）
        column_names = next(csv_reader)
        # 获取所有数据行
        for row in csv_reader:
            data_rows.append(row)

    df = pd.DataFrame(data=data_rows, columns=column_names)

    return df


def load_target_column_name(loc):
    with open(loc, 'r') as f:
        csv_reader = csv.reader(f)
        # 获取列名（第一行）
        column_names = next(csv_reader)
    last_column_name = column_names[-1]
    return last_column_name


def load_description(loc):
    with open(loc, "r", encoding="utf-8") as f:
        full_content = f.read()
    return full_content


# 随机森林下游模型算法
def RandomForest_feat(seed):
    print("# Random Forest")
    rforest = RandomForestClassifier(n_estimators=100, random_state=seed, class_weight='balanced')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)  # 可重复的随机数据划分
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
        "max_depth": [3, 5, 10],  # 树的最大深度
        "n_estimators": [100, 200, 300],  # 树的数量
        "subsample": [0.8, 1.0],  # 样本采样比例
        "colsample_bytree": [0.8, 1.0],  # 特征采样比例
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
        df_train,
        df_test,
        llm_model='gpt-3.5-turbo',
        iterations=10,
        target_column_name='class',
        dataset_description=None,
        task='classification'
):
    # clf_no_feat_eng = TabPFNClassifier(device=('cuda' if torch.cuda.is_available() else 'cpu'), N_ensemble_configurations=4)
    #
    # clf_no_feat_eng.fit = partial(clf_no_feat_eng.fit, overwrite_warning=True)
    
    if task == 'regression':
        clf_no_feat_eng = RandomForestRegressor(n_estimators=100, max_depth=2)
        caafe_clf = CAAFERegressor(base_classifier=clf_no_feat_eng,
                                    llm_model=llm_model,
                                    iterations=iterations)
    else:
        clf_no_feat_eng = RandomForestClassifier(n_estimators=100, max_depth=2)
        caafe_clf = CAAFEClassifier(base_classifier=clf_no_feat_eng,
                                    llm_model=llm_model,
                                    iterations=iterations)
    caafe_clf.fit_pandas(df_train,
                         target_column_name=target_column_name,
                         dataset_description=dataset_description)

    X_train_original = df_train.drop(columns=[target_column_name])
    y_train = df_train[target_column_name]
    X_train_with_features = run_llm_code(caafe_clf.code, X_train_original)
    df_train_with_features = pd.concat([
        X_train_with_features.reset_index(drop=True),
        y_train.reset_index(drop=True)
    ], axis=1)

    X_test_original = df_test.drop(columns=[target_column_name])
    y_test = df_test[target_column_name]
    X_test_with_features = run_llm_code(caafe_clf.code, X_test_original)
    df_test_with_features = pd.concat([
        X_test_with_features.reset_index(drop=True),
        y_test.reset_index(drop=True)
    ], axis=1)

    return df_train_with_features, df_test_with_features


# 计算统计指标
def print_stats(name, values):
    print(f"{name}: {np.mean(values):.2f} ± {np.std(values):.2f}")


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_path', default="/root/autodl-tmp/gzh/Llama2-chat-13B-Chinese-50W", type=str,
                        help='Base model路径')
    parser.add_argument('-g', '--gpus', default="0", type=str, help='GPU设置')
    parser.add_argument('-l', '--data_location', default="data/cd1.pkl", type=str, help='数据集路径')
    parser.add_argument('-s', '--default_seed', default=52, type=int, help='随机种子')
    parser.add_argument('-m', '--model', default='gpt-4o', type=str, help='大模型')
    parser.add_argument('-i', '--iterations', default=5, type=int, help='迭代次数')
    parser.add_argument('-t', '--task', default='regression', type=str, help='任务名')
    parser.add_argument('-d', '--dataset', default=None, type=str, help='数据集名称')
    args = parser.parse_args()

    seed = args.default_seed
    datasets = [args.dataset] if args.dataset else ['boston','california']
    for ds_name in datasets:
        print(f"\n=========== Dataset {ds_name} ===========")
        # 新增：存储每次实验结果的列表
        test_acc_list = []
        test_f1_list = []
        test_auc_list = []
        test_pre_list = []
        test_rec_list = []

        if ds_name not in ['boston', 'california']:
            loc_train = "tests/data_ji/" + args.task + "/" + ds_name + "/train_split.csv"
            loc_test = "tests/data_ji/" + args.task + "/" + ds_name + "/test_split.csv"
            loc_description = "tests/data_ji/" + args.task + "/" + ds_name + "/dataset_description.txt"
        else:
            loc_train = "tests/data_ji/" + args.task + "/" + ds_name + "/" + ds_name + "_original_train.csv"
            loc_test = "tests/data_ji/" + args.task + "/" + ds_name + "/" + ds_name + "_original_test.csv"
            loc_description = "tests/data_ji/" + args.task + "/" + ds_name + "/origin_data_task_description.txt"
        target_dir = "tests/data_ji/seed" + str(seed) + "/" + args.task + "/outputs_CAAFE"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)  # 递归创建多级目录

        random.seed(seed)
        np.random.seed(seed)
        df_train = load_origin_data(loc_train)
        df_test = load_origin_data(loc_test)
        target_column_name = load_target_column_name(loc_train)
        dataset_description = load_description(loc_description)

        feat_start = time.time()
        df_train_with_features, df_test_with_features = generate_feat_effect(
            df_train=df_train,
            df_test=df_test,
            llm_model=args.model,
            iterations=args.iterations,
            target_column_name=target_column_name,
            dataset_description=dataset_description,
            task=args.task
        )
        feat_end = time.time()
        total_time = round(feat_end - feat_start, 2)
        print(f"特征生成完成，耗时: {total_time} 秒")

        train_save_path = os.path.join(target_dir, ds_name + "_original_CAAFE_train.csv")
        df_train_with_features.to_csv(train_save_path, index=False)
        test_save_path = os.path.join(target_dir, ds_name + "_original_CAAFE_test.csv")
        df_test_with_features.to_csv(test_save_path, index=False)

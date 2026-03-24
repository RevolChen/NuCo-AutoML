import copy
import pandas as pd
# import tabpfn
import numpy as np
from .data import get_X_y
from .preprocessing import make_datasets_numeric, make_dataset_numeric
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_squared_error

def evaluate_dataset(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    prompt_id,
    name,
    method,
    metric_used,
    target_name,
    max_time=300,
    seed=0,
):
    df_train, df_test = copy.deepcopy(df_train), copy.deepcopy(df_test)
    df_train, _, mappings = make_datasets_numeric(
        df_train, None, target_name, return_mappings=True
    )
    df_test = make_dataset_numeric(df_test, mappings=mappings)

    if df_test is not None:
        test_x, test_y = get_X_y(df_test, target_name=target_name)

    x, y = get_X_y(df_train, target_name=target_name)
    feature_names = list(df_train.drop(target_name, axis=1).columns)

    np.random.seed(0)
    
    if method == "autogluon" or method == "autosklearn2":
        raise NotImplementedError("TabPFN scripts are missing, autogluon/autosklearn2 not supported")
    elif type(method) == str:
        raise NotImplementedError("TabPFN scripts are missing, string methods not supported")
    # If sklearn estimator
    elif isinstance(method, BaseEstimator):
        # Convert tensors to numpy if needed
        X_np = x.numpy() if hasattr(x, 'numpy') else x
        y_np = y.numpy() if hasattr(y, 'numpy') else y
        test_x_np = test_x.numpy() if hasattr(test_x, 'numpy') else test_x
        
        if isinstance(method, RegressorMixin):
            method.fit(X=X_np, y=y_np)
            ys = method.predict(test_x_np)
        else:
            # Default to classifier behavior (ClassifierMixin or others)
            # y should be int for classifier
            y_np = y_np.astype(int)
            method.fit(X=X_np, y=y_np)
            ys = method.predict_proba(test_x_np)
    else:
        metric, ys, res = method(
            x,
            y,
            test_x,
            test_y,
            [],
            metric_used,
        )

    # Calculate metrics using sklearn
    if hasattr(test_y, 'numpy'):
        test_y_np = test_y.numpy()
    else:
        test_y_np = test_y
    
    if isinstance(method, RegressorMixin):
        acc = r2_score(test_y_np, ys)
        roc = 0.0 # No ROC for regression
    else:
        # ys is proba (n_samples, n_classes)
        if ys.ndim > 1:
            y_pred = np.argmax(ys, axis=1)
        else:
            y_pred = (ys > 0.5).astype(int)
            
        acc = accuracy_score(test_y_np, y_pred)
        
        try:
            if ys.shape[1] > 2:
                 roc = roc_auc_score(test_y_np, ys, multi_class='ovr')
            elif ys.shape[1] == 2:
                 roc = roc_auc_score(test_y_np, ys[:, 1])
            else:
                 roc = roc_auc_score(test_y_np, ys)
        except:
            roc = 0.5

    method_str = method if type(method) == str else "transformer"
    return {
        "acc": float(acc),
        "roc": float(roc),
        "prompt": prompt_id,
        "seed": seed,
        "name": name,
        "size": len(df_train),
        "method": method_str,
        "max_time": max_time,
        "feats": x.shape[-1],
    }


def get_leave_one_out_importance(
    df_train, df_test, ds, method, metric_used, max_time=30
):
    """Get the importance of each feature for a dataset by dropping it in the training and prediction."""
    res_base = evaluate_dataset(
        df_train=df_train,
        df_test=df_test,
        prompt_id="",
        name=ds[0],
        method=method,
        metric_used=metric_used,
        max_time=max_time,
        target_name=ds[4][-1],
    )

    importances = {}
    for feat_idx, feat in enumerate(set(df_train.columns)):
        if feat == ds[4][-1]:
            continue
        df_train_ = df_train.copy().drop(feat, axis=1)
        df_test_ = df_test.copy().drop(feat, axis=1)
        ds_ = copy.deepcopy(ds)

        res = evaluate_dataset(
            df_train=df_train_,
            df_test=df_test_,
            prompt_id="",
            name=ds[0],
            method=method,
            metric_used=metric_used,
            max_time=max_time,
            target_name=ds[4][-1],
        )
        importances[feat] = (round(res_base["roc"] - res["roc"], 3),)
    return importances

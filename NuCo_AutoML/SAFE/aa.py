from caafe import CAAFEClassifier # Automated Feature Engineering for tabular datasets
from tabpfn import TabPFNClassifier # Fast Automated Machine Learning method for small tabular datasets
from sklearn.ensemble import RandomForestClassifier

import os
import openai
import torch
from caafe import data
from sklearn.metrics import accuracy_score,roc_auc_score, f1_score
from tabpfn.scripts import tabular_metrics
from functools import partial
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

metric_used = tabular_metrics.auc_metric
cc_test_datasets_multiclass = data.load_all_data()
ds = cc_test_datasets_multiclass[3]
ds, df_train, df_test, _, _ = data.get_data_split(ds, seed=0)
target_column_name = ds[4][-1]
dataset_description = ds[-1]

from caafe.preprocessing import make_datasets_numeric
df_train, df_test = make_datasets_numeric(df_train, df_test, target_column_name)
train_x, train_y = data.get_X_y(df_train, target_column_name)
test_x, test_y = data.get_X_y(df_test, target_column_name)

### Setup Base Classifier

# clf_no_feat_eng = RandomForestClassifier()
clf_no_feat_eng = TabPFNClassifier(device=('cuda' if torch.cuda.is_available() else 'cpu'), N_ensemble_configurations=4)
clf_no_feat_eng.fit = partial(clf_no_feat_eng.fit, overwrite_warning=True)

clf_no_feat_eng.fit(train_x, train_y)
pred = clf_no_feat_eng.predict(test_x)

print("Accuracy before CAAFE:", accuracy_score(pred, test_y))
print("F1 Score before CAAFE:", f1_score(pred, test_y, average='weighted'))
print("AUC Score before CAAFE:", roc_auc_score(test_y, clf_no_feat_eng.predict_proba(test_x)[:, 1]))


### Setup and Run CAAFE - This will be billed to your OpenAI Account!

caafe_clf = CAAFEClassifier(base_classifier=clf_no_feat_eng,
                            llm_model="gpt-3.5-turbo",
                            iterations=2)

caafe_clf.fit_pandas(df_train,
                     target_column_name=target_column_name,
                     dataset_description=dataset_description)

pred = caafe_clf.predict(df_test)
print("Accuracy after CAAFE:", accuracy_score(pred, test_y))
print("F1 Score after CAAFE:", f1_score(pred, test_y, average='weighted'))
print("AUC Score after CAAFE:", roc_auc_score(test_y, caafe_clf.predict_proba(df_test)[:, 1]))

print(caafe_clf.code)

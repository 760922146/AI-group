import matplotlib.pyplot as plt
import seaborn as sns
import gc
import re
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier

class_dict = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7
}

month_to_num = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12
}

def parse_work_year(x):
    if str(x) == 'nan':
        return x
    x = x.replace('< 1', '0')
    return int(re.search('(\d+)', x).group())

def parse_month(x):
    pattern = r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b'
    match = re.search(pattern, x, re.IGNORECASE)
    if match:
        return month_to_num[match.group()]
    return None

def find_outliers_by_3segama(data,fea):
    data_std = np.std(data[fea])
    data_mean = np.mean(data[fea])
    outliers_cut_off = data_std * 3
    lower_rule = data_mean - outliers_cut_off
    upper_rule = data_mean + outliers_cut_off
    data[fea+'_outliers'] = data[fea].apply(lambda x:str('异常值') if x > upper_rule or x < lower_rule else '正常值')
    return data

def train_model(data_, test_, y_, folds_):
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in data_.columns if f not in ['load_id', 'isDefault']]
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        clf = LGBMClassifier(
            max_depth=5,  # 树的最大深度
            num_leaves=20,
            min_child_samples=800,
            subsample=1.0,
            colsample_bytree=0.9,
            reg_alpha=0.5,

            n_estimators=1000,  # 弱分类器的数目
            learning_rate=0.1,  # 为此需要给每个弱学习器拟合的残差值都乘上取值范围在(0, 1]的eta,设置较小的eta就可以多学习几个弱学习器来弥补不足的残差
            min_child_weight=1,
            silent=False,
            verbose=-1
        )

        clf.fit(trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                eval_metric='auc'
                )

        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_[feats], num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y_, oof_preds))

    test_['isDefault'] = sub_preds

    return oof_preds, test_[['loan_id', 'isDefault']], feature_importance_df

def display_importances(feature_importance_df_):
    # Plot feature importances
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
           :50].index

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature",
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')

if __name__ == '__main__':
    # pd.set_option('max_columns', None)
    # pd.set_option('max_rows', 200)
    pd.set_option('float_format', lambda x: '%.3f' % x)

    train_data = pd.read_csv('data/train_public.csv')
    test_data = pd.read_csv('data/test_public.csv')

    # numerical_fea = list(train_data.select_dtypes(exclude=['object']).columns)
    # category_fea = list(filter(lambda x: x not in numerical_fea, list(train_data.columns)))
    # label = 'isDefault'
    # numerical_fea.remove(label)
    #
    # # 按照中位数填充数值型特征
    # train_data[numerical_fea] = train_data[numerical_fea].fillna(train_data[numerical_fea].median())
    # test_data[numerical_fea] = test_data[numerical_fea].fillna(train_data[numerical_fea].median())
    # # 按照众数填充类别型特征
    # train_data[category_fea] = train_data[category_fea].fillna(train_data[category_fea].mode())
    # test_data[category_fea] = test_data[category_fea].fillna(train_data[category_fea].mode())

    for data in [train_data, test_data]:
        data['work_year'] = data['work_year'].map(parse_work_year)

        data['class'] = data['class'].map(class_dict)

        data['earlies_credit_mon'] = data['earlies_credit_mon'].map(parse_month)

        data['issue_date'] = pd.to_datetime(data['issue_date'])
        data['issue_month'] = data['issue_date'].dt.month
        data['issue_year'] = data['issue_date'].dt.year

    train_data = pd.get_dummies(train_data,
                          columns=['employer_type', 'industry', 'use', 'region'],
                          drop_first=True)
    test_data = pd.get_dummies(test_data,
                          columns=['employer_type', 'industry', 'use', 'region'],
                          drop_first=True)

    # 删除异常值
    # numerical_fea = ['loanAmnt', 'term', 'interestRate', 'installment']
    # for fea in numerical_fea:
    #     train_data = find_outliers_by_3segama(train_data, fea)
    #     train_data = train_data[train_data[fea + '_outliers'] == '正常值']
    #     train_data = train_data.reset_index(drop=True)
    #     train_data = train_data.drop(fea + '_outliers', axis=1)
    #
    col_to_drop = ['user_id', 'issue_date', 'title', 'post_code']
    train_data = train_data.drop(col_to_drop, axis=1)
    test_data = test_data.drop(col_to_drop, axis=1)

    # # 计算协方差
    # x_train = train_data.drop(['isDefault', 'id'], axis=1)
    # data_corr = x_train.corrwith(train_data.isDefault)  # 计算相关性
    # result = pd.DataFrame(columns=['features', 'corr'])
    # result['features'] = data_corr.index
    # result['corr'] = data_corr.values
    #
    # # 删除不相关变量
    # print(result.query('0.005 > corr > -0.005'))
    # col_to_drop = ['policyCode']
    # train_data = train_data.drop(col_to_drop, axis=1)
    # test_data = test_data.drop(col_to_drop, axis=1)

    # 计算特征间协方差
    # n10 - openACC, n2 - n3, ficoRangeHigh - ficoRangeLow
    # for i in train_data.columns:
    #     print(train_data.corr()[i].sort_values(ascending=False)[1:3])

    # col_to_drop = ['n2', 'ficoRangeHigh']
    # train_data = train_data.drop(col_to_drop, axis=1)
    # test_data = test_data.drop(col_to_drop, axis=1)

    # 方差选择法
    # print(train_data.var().sort_values())

    # print(train_data)

    y = train_data['isDefault']
    folds = KFold(n_splits=10, shuffle=True, random_state=2024)

    oof_preds, test_preds, importances = train_model(train_data, test_data, y, folds)
    test_preds.rename({'loan_id': 'id'}, axis=1)[['id', 'isDefault']].to_csv('submission.csv', index=False)



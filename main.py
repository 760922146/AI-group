import matplotlib.pyplot as plt
import seaborn as sns
import gc
import re
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
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

def lgbm_model(data_, test_, y_, folds_):
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in data_.columns if f not in ['loan_id', 'isDefault']]
    print('LightGBM:')
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
    return oof_preds,  sub_preds, feature_importance_df

def rf_model(data_, test_, y_train):
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    feats = [f for f in data_.columns if f not in ['loan_id', 'isDefault']]
    X_train = data_[feats]
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=2024)

    # 进行K折交叉验证
    k = 10
    cv_iterator = StratifiedKFold(n_splits=k, shuffle=True, random_state=2024)

    print('Random Forest:')

    for fold, (train_indices, val_indices) in enumerate(cv_iterator.split(X_train, y_train)):
        X_train_fold, X_val_fold = X_train.iloc[train_indices], X_train.iloc[val_indices]
        y_train_fold, y_val_fold = y_train.iloc[train_indices], y_train.iloc[val_indices]

        # 训练模型
        rf_classifier.fit(X_train_fold, y_train_fold)

        # 在验证集上进行预测并计算AUC
        y_val_proba = rf_classifier.predict_proba(X_val_fold)[:, 1]
        auc_val = roc_auc_score(y_val_fold, y_val_proba)
        print('Fold %2d AUC : %.6f' % (fold, auc_val))

        # 在测试集上进行预测并保存结果
        y_test_proba = rf_classifier.predict_proba(test_[feats])[:, 1]
        sub_preds += y_test_proba / k
        oof_preds[val_indices] = rf_classifier.predict_proba(X_val_fold)[:, 1]

    print('Full AUC score %.6f' % roc_auc_score(y_train, oof_preds))
    return oof_preds, sub_preds

def display_importances(feature_importance_df_):
    # Plot feature importances
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
           :25].index

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature",
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')

# 细分class
def feature_Kmeans(data, label):
    mms = MinMaxScaler()
    feats = [f for f in data.columns if f not in ['loan_id', 'user_id', 'isDefault']]
    data = data[feats]
    mmsModel = mms.fit_transform(data.loc[data['class'] == label])
    clf = KMeans(5, n_init='auto' ,random_state=2024)
    pre = clf.fit(mmsModel)
    test = pre.labels_
    final_data = pd.Series(test, index=data.loc[data['class'] == label].index)
    if label == 1:
        final_data = final_data.map({0: 'A1', 1: 'A2', 2: 'A3', 3: 'A4', 4: 'A5'})
    elif label == 2:
        final_data = final_data.map({0: 'B1', 1: 'B2', 2: 'B3', 3: 'B4', 4: 'B5'})
    elif label == 3:
        final_data = final_data.map({0: 'C1', 1: 'C2', 2: 'C3', 3: 'C4', 4: 'C5'})
    elif label == 4:
        final_data = final_data.map({0: 'D1', 1: 'D2', 2: 'D3', 3: 'D4', 4: 'D5'})
    elif label == 5:
        final_data = final_data.map({0: 'E1', 1: 'E2', 2: 'E3', 3: 'E4', 4: 'E5'})
    elif label == 6:
        final_data = final_data.map({0: 'F1', 1: 'F2', 2: 'F3', 3: 'F4', 4: 'F5'})
    elif label == 7:
        final_data = final_data.map({0: 'G1', 1: 'G2', 2: 'G3', 3: 'G4', 4: 'G5'})
    return final_data


if __name__ == '__main__':
    pd.set_option('float_format', lambda x: '%.3f' % x)

    train_data = pd.read_csv('data/train_public.csv')
    test_data = pd.read_csv('data/test_public.csv')

    # 填充缺失值
    col_to_fill = ['pub_dero_bankrup', 'f0', 'f1', 'f2', 'f3', 'f4']
    for column in col_to_fill:
        median_value = train_data[column].median()
        train_data[column].fillna(median_value, inplace=True)
        test_data[column].fillna(median_value, inplace=True)
    
    train_data['work_year'] = train_data['work_year'].fillna(train_data['work_year'].mode()[0])
    test_data['work_year'] = test_data['work_year'].fillna(train_data['work_year'].mode()[0])

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

    col_to_drop = ['user_id', 'issue_date', 'title', 'post_code']
    train_data = train_data.drop(col_to_drop, axis=1)
    test_data = test_data.drop(col_to_drop, axis=1)

    # 计算相关性
    x_train = train_data.drop(['isDefault', 'loan_id'], axis=1)
    data_corr = x_train.corrwith(train_data.isDefault)  
    result = pd.DataFrame(columns=['features', 'corr'])
    result['features'] = data_corr.index
    result['corr'] = data_corr.values
    
    # 删除不相关变量
    col_to_drop = result[result['corr'].abs() < 0.005]['features']
    train_data = train_data.drop(col_to_drop, axis=1)
    test_data = test_data.drop(col_to_drop, axis=1)

    # 删除异常值
    numerical_fea = ['recircle_u', 'recircle_b', 'debt_loan_ratio']
    for fea in numerical_fea:
        train_data = find_outliers_by_3segama(train_data, fea)
        train_data = train_data[train_data[fea + '_outliers'] == '正常值']
        train_data = train_data.reset_index(drop=True)
        train_data = train_data.drop(fea + '_outliers', axis=1)
    
    
    # 训练集合并
    train_data1 = feature_Kmeans(train_data, 1)
    train_data2 = feature_Kmeans(train_data, 2)
    train_data3 = feature_Kmeans(train_data, 3)
    train_data4 = feature_Kmeans(train_data, 4)
    train_data5 = feature_Kmeans(train_data, 5)
    train_data6 = feature_Kmeans(train_data, 6)
    train_data7 = feature_Kmeans(train_data, 7)
    train_dataall = pd.concat(
        [train_data1, train_data2, train_data3, train_data4, train_data5, train_data6, train_data7]).reset_index(drop=True)
    train_data['sub_class'] = train_dataall
    # 测试集合并
    test_data1 = feature_Kmeans(test_data, 1)
    test_data2 = feature_Kmeans(test_data, 2)
    test_data3 = feature_Kmeans(test_data, 3)
    test_data4 = feature_Kmeans(test_data, 4)
    test_data5 = feature_Kmeans(test_data, 5)
    test_data6 = feature_Kmeans(test_data, 6)
    test_data7 = feature_Kmeans(test_data, 7)
    test_dataall = pd.concat(
        [test_data1, test_data2, test_data3, test_data4, test_data5, test_data6, test_data7]).reset_index(drop=True)
    test_data['sub_class'] = test_dataall

    cat_cols = ['sub_class']
    for col in cat_cols:
        lbl = LabelEncoder().fit(train_data[col])
        train_data[col] = lbl.transform(train_data[col])
        test_data[col] = lbl.transform(test_data[col])

    # 添加新特征（效果不好）  
    # for data in [train_data, test_data]:
    #     data['post_code_interst_mean'] = data.groupby(['post_code'])['interest'].transform('mean')
    #     data['recircle_u_b_std'] = data.groupby(['recircle_u'])['recircle_b'].transform('std')
    #     data['early_return_amount_early_return'] = data['early_return_amount'] / data['early_return']
    #     data['early_return_amount_early_return'][np.isinf(data['early_return_amount_early_return'])] = 0
    #     data['total_loan_monthly_payment'] = data['monthly_payment'] * data['year_of_loan'] * 12 - data['total_loan'] 


    y = train_data['isDefault']
    folds = KFold(n_splits=10, shuffle=True, random_state=2024)

    lgbm_train_preds, lgbm_test_preds, importances = lgbm_model(train_data, test_data, y, folds)
    display_importances(importances)
    rf_train_preds, rf_test_preds = rf_model(train_data, test_data, y)

    # 模型融合
    stacking_X_train = np.column_stack((lgbm_train_preds, rf_train_preds))
    meta_model = LogisticRegression()
    meta_model.fit(stacking_X_train, y)
    stacking_X_test = np.column_stack((lgbm_test_preds, rf_test_preds))
    stacking_y_pred = meta_model.predict(stacking_X_test)
    test_data['isDefault'] = stacking_y_pred

    test_data.rename({'loan_id': 'id'}, axis=1)[['id', 'isDefault']].to_csv('submission.csv', index=False)



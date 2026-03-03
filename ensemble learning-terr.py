import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import collections

from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, log_loss)
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from catboost import CatBoostClassifier, Pool
from bayes_opt import BayesianOptimization

# ---------------------------
# 1. 数据加载与预处理
# ---------------------------
file_path = 'Terrestrial.csv'
data = pd.read_csv(file_path)

# 定义特征列和目标列
feature_columns = ["CTI", "SPI", "DTG", "ETa_mean_dry", "ETa_mean_annual",
                   "clay_mean", "cv_lst", "elevation", "mTPI", "msavi",
                   "ndvi", "ndwi_leaf", "ndwi_water", "pr_mean_dry",
                   "wtd_2015", "pr_mean_annual"]
target_column = "class"

# 移除缺失值记录
data = data.dropna(subset=feature_columns + [target_column])
print("数据预览：")
print(data.head(10))

# 提取特征和目标，并调整标签从 0 开始
X = data[feature_columns]
y = data[target_column].astype('int')
y -= y.min()

# ---------------------------
# 2. 模型参数设置
# ---------------------------
# 固定随机种子：2025
# XGBoost 参数配置
xgb_params = {
    'objective': 'multi:softprob',  # 多分类任务，输出各类别概率
    'num_class': len(y.unique()),
    'max_depth': 6,
    'tree_method': 'hist',  # 使用直方图算法
    'device': 'cuda',  # 使用 GPU（如无GPU可改为 'auto'）
    'eta': 0.14106518431479867,
    'subsample': 0.7666334519300132,
    'colsample_bytree': 0.5225981161537219,
    'eval_metric': 'mlogloss',
    'seed': 2025
}
xgb_num_round = 1772  # 固定迭代次数

# TabNet 参数配置
tabnet_params = dict(
    n_d=69, n_a=69,
    n_steps=4,
    gamma=1.3,
    lambda_sparse=1e-4,
    optimizer_params=dict(lr=0.039),
    scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
    scheduler_params={"T_max": 50, "eta_min": 1e-4},
    mask_type='entmax',
    device_name="cuda"  # 如无GPU，可改为 "cpu"
)

# CatBoost 参数配置
params_cat = {
    'loss_function': 'MultiClass',  # 多分类任务
    'iterations': 891,  # 默认迭代次数（内层优化时后续覆盖）
    'depth': 9,  # 树的最大深度
    'learning_rate': 0.2869477920648122,  # 学习率
    'l2_leaf_reg': 3.295577591504187,  # L2 正则化
    'bagging_temperature': 0.7941119471755271,  # 模拟 subsample 效果
    'random_seed': 2025,  # 随机种子
    'verbose': False,  # 关闭训练日志
    'task_type': 'GPU'
}

# LightGBM 参数配置
lgb_params = {
    'objective': 'multiclass',
    'num_class': len(y.unique()),
    'max_depth': 8,
    'learning_rate':0.07540954712403862,
    'feature_fraction':0.6449796319631782,
    'num_leaves':458,
    'bagging_fraction': 0.6361247434884769,
    'n_estimators': 1276,
    'random_state': 2025,
    'metric': 'multi_logloss'  # 使用多分类交叉熵损失作为默认评估指标
}

# ---------------------------
# 3. 外部 5 折交叉验证（嵌套 CV 外层）
# ---------------------------
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)

# 用于保存所有外部折真实标签、预测标签及预测概率
outer_all_true = []
outer_all_pred = []
outer_all_pred_proba = []
# 保存每个外部折内层优化得到的最优融合权重
outer_best_weights = []
# 保存每个外部折的 ROC AUC
outer_fold_aucs = []

fold_idx = 1
for outer_train_idx, outer_val_idx in outer_cv.split(X, y):
    print(f"\n=== 外部折 {fold_idx} ===")
    # 划分外部训练集和外部验证集
    X_outer_train = X.iloc[outer_train_idx].copy()
    y_outer_train = y.iloc[outer_train_idx].copy()
    X_outer_val = X.iloc[outer_val_idx].copy()
    y_outer_val = y.iloc[outer_val_idx].copy()

    # 对外部训练集进行归一化，并对外部验证集使用相同 scaler
    scaler = MinMaxScaler()
    X_outer_train_scaled = scaler.fit_transform(X_outer_train)
    X_outer_val_scaled = scaler.transform(X_outer_val)

    # ---------------------------
    # 内部划分：在外部训练集上随机划分 80%（内部训练）和 20%（内部验证）
    # ---------------------------
    X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
        X_outer_train_scaled, y_outer_train, test_size=0.2, stratify=y_outer_train, random_state=2025
    )

    # ---------------------------
    # 在内部训练集上训练各个基模型，并在内部验证集上获取预测概率
    # ---------------------------
    # XGBoost 模型
    dtrain_inner = xgb.DMatrix(X_inner_train, label=y_inner_train, feature_names=feature_columns)
    dval_inner = xgb.DMatrix(X_inner_val, label=y_inner_val, feature_names=feature_columns)
    model_xgb_inner = xgb.train(xgb_params, dtrain_inner, num_boost_round=xgb_num_round, verbose_eval=False)
    pred_xgb_inner = model_xgb_inner.predict(dval_inner)

    # 随机森林模型
    model_rf_inner = RandomForestClassifier(
        n_estimators=49,
        max_features=12,
        max_depth=22,
        min_samples_leaf=1,
        bootstrap=True,
        max_samples=0.5911839286089882,
        max_leaf_nodes=2960,
        random_state=2025
    )
    model_rf_inner.fit(X_inner_train, y_inner_train)
    pred_rf_inner = model_rf_inner.predict_proba(X_inner_val)

    # TabNet 模型
    model_tabnet_inner = TabNetClassifier(**tabnet_params)
    model_tabnet_inner.fit(
        X_inner_train, y_inner_train,
        eval_set=[(X_inner_train, y_inner_train)],
        eval_name=['train'],
        eval_metric=['logloss'],
        max_epochs=60,
        patience=20,
        batch_size=2048,
        virtual_batch_size=256,
        drop_last=False
    )
    pred_tabnet_inner = model_tabnet_inner.predict_proba(X_inner_val)

    # CatBoost 模型
    params_cat_inner = params_cat.copy()
    model_cat_inner = CatBoostClassifier(**params_cat_inner)
    model_cat_inner.fit(X_inner_train, y_inner_train, verbose=params_cat_inner['verbose'])
    pred_cat_inner = model_cat_inner.predict_proba(X_inner_val)

    # LightGBM 模型
    model_lgb_inner = lgb.LGBMClassifier(**lgb_params)
    model_lgb_inner.fit(X_inner_train, y_inner_train)
    pred_lgb_inner = model_lgb_inner.predict_proba(X_inner_val)


    # ---------------------------
    # 内层贝叶斯优化：调节各模型的融合权重
    # ---------------------------
    # 定义目标函数：传入各权重后计算归一化后的融合预测概率，然后基于内部验证集计算 log loss（越低越好）
    def inner_objective(w_xgb, w_rf, w_tabnet, w_cat, w_lgb):
        total = w_xgb + w_rf + w_tabnet + w_cat + w_lgb
        nw_xgb = w_xgb / total
        nw_rf = w_rf / total
        nw_tabnet = w_tabnet / total
        nw_cat = w_cat / total
        nw_lgb = w_lgb / total

        ensemble_pred = (nw_xgb * pred_xgb_inner +
                         nw_rf * pred_rf_inner +
                         nw_tabnet * pred_tabnet_inner +
                         nw_cat * pred_cat_inner +
                         nw_lgb * pred_lgb_inner)
        # 归一化每一行，使其和为 1
        ensemble_pred = ensemble_pred / ensemble_pred.sum(axis=1, keepdims=True)
        loss = log_loss(y_inner_val, ensemble_pred)
        return -loss  # 返回负的 log loss（贝叶斯优化默认求最大值）


    pbounds = {
        'w_xgb': (1e-10, 1),
        'w_rf': (1e-10, 1),
        'w_tabnet': (1e-10, 1),
        'w_cat': (1e-10, 1),
        'w_lgb': (1e-10, 1)
    }

    optimizer = BayesianOptimization(
        f=inner_objective,
        pbounds=pbounds,
        random_state=2025,
        verbose=0
    )

    print("开始内部贝叶斯优化调融合权重...")
    optimizer.maximize(init_points=5, n_iter=50)

    # 打印每一次调参的结果
    print("贝叶斯优化每次调参的结果:")
    for i, res in enumerate(optimizer.res):
        print(f"Iteration {i + 1}: {res}")

    best_params = optimizer.max['params']
    total = best_params['w_xgb'] + best_params['w_rf'] + best_params['w_tabnet'] + best_params['w_cat'] + best_params[
        'w_lgb']
    best_weights = {
        'w_xgb': best_params['w_xgb'] / total,
        'w_rf': best_params['w_rf'] / total,
        'w_tabnet': best_params['w_tabnet'] / total,
        'w_cat': best_params['w_cat'] / total,
        'w_lgb': best_params['w_lgb'] / total
    }
    print(f"内部最优融合权重: {best_weights}, 内部 log loss: {-optimizer.max['target']:.4f}")
    outer_best_weights.append(best_weights)

    # ---------------------------
    # 在整个外部训练集上训练各基模型，并在外部验证集上融合预测
    # ---------------------------
    # XGBoost 模型
    dtrain_outer = xgb.DMatrix(X_outer_train_scaled, label=y_outer_train, feature_names=feature_columns)
    dval_outer = xgb.DMatrix(X_outer_val_scaled, label=y_outer_val, feature_names=feature_columns)
    model_xgb_outer = xgb.train(xgb_params, dtrain_outer, num_boost_round=xgb_num_round, verbose_eval=False)
    pred_xgb_outer = model_xgb_outer.predict(dval_outer)

    # 随机森林模型
    model_rf_outer = RandomForestClassifier(
        n_estimators=49,
        max_features=12,
        max_depth=22,
        min_samples_leaf=1,
        bootstrap=True,
        max_samples=0.5911839286089882,
        max_leaf_nodes=2960,
        random_state=2025
    )
    model_rf_outer.fit(X_outer_train_scaled, y_outer_train)
    pred_rf_outer = model_rf_outer.predict_proba(X_outer_val_scaled)

    # TabNet 模型
    model_tabnet_outer = TabNetClassifier(**tabnet_params)
    model_tabnet_outer.fit(
        X_outer_train_scaled, y_outer_train,
        eval_set=[(X_outer_train_scaled, y_outer_train)],
        eval_name=['train'],
        eval_metric=['logloss'],
        max_epochs=60,
        patience=20,
        batch_size=2048,
        virtual_batch_size=256,
        drop_last=False
    )
    pred_tabnet_outer = model_tabnet_outer.predict_proba(X_outer_val_scaled)

    # CatBoost 模型
    params_cat_outer = params_cat.copy()
    model_cat_outer = CatBoostClassifier(**params_cat_outer)
    model_cat_outer.fit(X_outer_train_scaled, y_outer_train, verbose=params_cat_outer['verbose'])
    pred_cat_outer = model_cat_outer.predict_proba(X_outer_val_scaled)

    # LightGBM 模型
    model_lgb_outer = lgb.LGBMClassifier(**lgb_params)
    model_lgb_outer.fit(X_outer_train_scaled, y_outer_train)
    pred_lgb_outer = model_lgb_outer.predict_proba(X_outer_val_scaled)

    # 融合外部验证集的预测（使用内部调优得到的最优融合权重）
    ensemble_pred_proba_outer = (best_weights['w_xgb'] * pred_xgb_outer +
                                 best_weights['w_rf'] * pred_rf_outer +
                                 best_weights['w_tabnet'] * pred_tabnet_outer +
                                 best_weights['w_cat'] * pred_cat_outer +
                                 best_weights['w_lgb'] * pred_lgb_outer)
    # 归一化每一行，使其和为 1
    ensemble_pred_proba_outer = ensemble_pred_proba_outer / ensemble_pred_proba_outer.sum(axis=1, keepdims=True)
    ensemble_pred_outer = np.argmax(ensemble_pred_proba_outer, axis=1)

    # 计算当前外部折的 ROC AUC
    try:
        # 这里需要将外部验证标签二值化，注意类别数为 len(np.unique(y))
        y_outer_val_bin = label_binarize(y_outer_val, classes=np.arange(len(np.unique(y))))
        fold_auc = roc_auc_score(y_outer_val_bin, ensemble_pred_proba_outer, multi_class='ovr')
    except Exception as e:
        fold_auc = None
    outer_fold_aucs.append(fold_auc)
    print(f"外部折 {fold_idx} ROC AUC: {fold_auc:.4f}" if fold_auc is not None else "ROC AUC 计算失败")

    # 保存外部折结果
    outer_all_true.extend(y_outer_val.tolist())
    outer_all_pred.extend(ensemble_pred_outer.tolist())
    outer_all_pred_proba.extend(ensemble_pred_proba_outer.tolist())

    # 输出外部折性能
    print(f"外部折 {fold_idx} 混淆矩阵:")
    print(confusion_matrix(y_outer_val, ensemble_pred_outer))
    print(f"\n外部折 {fold_idx} 分类报告:")
    print(classification_report(y_outer_val, ensemble_pred_outer, digits=4))

    fold_idx += 1

# ---------------------------
# 4. 汇总所有外部折的结果
# ---------------------------
print("\n=== 外部 5 折交叉验证总体性能 ===")
overall_conf_mat = confusion_matrix(np.array(outer_all_true), np.array(outer_all_pred))
overall_class_rep = classification_report(np.array(outer_all_true), np.array(outer_all_pred), digits=4)
print("总体混淆矩阵:")
print(overall_conf_mat)
print("\n总体分类报告:")
print(overall_class_rep)

# 计算总体多分类 ROC AUC
all_true_arr = np.array(outer_all_true)
all_pred_proba_arr = np.array(outer_all_pred_proba)
try:
    overall_roc_auc = roc_auc_score(all_true_arr, all_pred_proba_arr, multi_class='ovr')
    print(f"\n总体 ROC AUC（One-vs-Rest）: {overall_roc_auc:.4f}")
except Exception as e:
    print(f"\nROC AUC 计算失败: {str(e)}")

# 输出每个外部折的最优融合权重和 ROC AUC
print("\n各外部折最优融合权重及 ROC AUC:")
for i, (weights, auc) in enumerate(zip(outer_best_weights, outer_fold_aucs), 1):
    print(
        f"折 {i}: 融合权重: {weights}, ROC AUC: {auc:.4f}" if auc is not None else f"折 {i}: 融合权重: {weights}, ROC AUC 计算失败")

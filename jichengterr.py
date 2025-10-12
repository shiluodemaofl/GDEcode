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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from catboost import CatBoostClassifier

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
# 2. 设置 5 折分层交叉验证
# ---------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)

# 用于保存各折真实标签、预测标签及预测概率（后续整体评估）
ensemble_all_true = []
ensemble_all_pred = []
ensemble_all_pred_proba = []

# 用于累计 XGBoost 特征重要性（以“weight”为依据）
aggregate_importance = {feat: 0.0 for feat in feature_columns}

# ---------------------------
# 3. 模型设置与参数
# ---------------------------

# 模型集成权重（可根据实际情况调整）
w_xgb = 0.00
w_rf = 0.33
w_tabnet = 0.34
w_cat = 0.00
w_lgb = 0.33

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
xgb_num_round = 1562  # 固定迭代次数（可根据调优结果设置）

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
    'iterations': 891,  # 迭代次数
    'depth': 9,  # 树的最大深度
    'learning_rate': 0.2869477920648122,  # 学习率
    'l2_leaf_reg': 3.295577591504187,  # L2 正则化（这里参考 xgb 的 alpha）
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
    'learning_rate': 0.18753006123173419,
    'feature_fraction': 0.820886027382496,
    'max_leaf_nodes': 220,
    'bagging_fraction': 0.5433552437556363,
    'n_estimators': 1539,
    'random_state': 2025,
    'metric': 'multi_logloss'  # 使用多分类交叉熵损失作为默认评估指标
}

# ---------------------------
# 4. 5 折交叉验证循环
# ---------------------------
fold_idx = 1
for train_index, val_index in skf.split(X, y):
    print(f"\n=== Fold {fold_idx} ===")
    # 划分当前折训练集和验证集
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]
    X_val = X.iloc[val_index]
    y_val = y.iloc[val_index]

    # 分别对训练集和验证集进行归一化（仅用当前折训练集拟合归一化器）
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # ---------------------------
    # XGBoost 模型训练
    # ---------------------------
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=feature_columns)
    dval = xgb.DMatrix(X_val_scaled, label=y_val, feature_names=feature_columns)
    model_xgb = xgb.train(xgb_params, dtrain, num_boost_round=xgb_num_round, verbose_eval=False)
    pred_xgb = model_xgb.predict(dval)
    # 累加 XGBoost 特征重要性
    fold_importance = model_xgb.get_score(importance_type='weight')
    for feat, score in fold_importance.items():
        aggregate_importance[feat] += score

    # ---------------------------
    # 随机森林模型训练
    # ---------------------------
    model_rf = RandomForestClassifier(
        n_estimators=49,
        max_features=12,
        max_depth=22,
        min_samples_leaf=1,
        bootstrap=True,
        max_samples=0.5911839286089882,
        max_leaf_nodes=2960,
        random_state=2025
    )
    model_rf.fit(X_train_scaled, y_train)
    pred_rf = model_rf.predict_proba(X_val_scaled)

    # ---------------------------
    # TabNet 模型训练
    # ---------------------------
    model_tabnet = TabNetClassifier(**tabnet_params)
    model_tabnet.fit(
        X_train_scaled, y_train,
        eval_set=[(X_train_scaled, y_train)],
        eval_name=['train'],
        eval_metric=['logloss'],
        max_epochs=60,
        patience=20,
        batch_size=2048,
        virtual_batch_size=256,
        drop_last=False
    )
    pred_tabnet = model_tabnet.predict_proba(X_val_scaled)

    # ---------------------------
    # CatBoost 模型训练
    # ---------------------------
    # 设置迭代次数为 820（根据 CV 得到的平均最佳迭代轮次）
    params_cat['iterations'] = 820
    model_cat = CatBoostClassifier(**params_cat)
    model_cat.fit(X_train_scaled, y_train, verbose=params_cat['verbose'])
    pred_cat = model_cat.predict_proba(X_val_scaled)

    # ---------------------------
    # LightGBM 模型训练
    # ---------------------------
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_train_scaled, y_train)
    pred_lgb = model_lgb.predict_proba(X_val_scaled)

    # ---------------------------
    # 集成预测（软投票：加权平均各模型预测概率）
    # ---------------------------
    ensemble_pred_proba = (w_xgb * pred_xgb +
                           w_rf * pred_rf +
                           w_tabnet * pred_tabnet +
                           w_cat * pred_cat +
                           w_lgb * pred_lgb)
    ensemble_pred = np.argmax(ensemble_pred_proba, axis=1)

    # 保存当前折真实标签和预测结果
    ensemble_all_true.extend(y_val.tolist())
    ensemble_all_pred.extend(ensemble_pred.tolist())
    ensemble_all_pred_proba.extend(ensemble_pred_proba.tolist())

    # 输出当前折的混淆矩阵和分类报告
    print(f"Fold {fold_idx} Confusion Matrix:")
    print(confusion_matrix(y_val, ensemble_pred))
    print(f"\nFold {fold_idx} Classification Report:")
    print(classification_report(y_val, ensemble_pred, digits=4))

    fold_idx += 1

# ---------------------------
# 5. 汇总所有折的结果
# ---------------------------
print("\n=== Overall 5-Fold Cross Validation Performance ===")
overall_conf_mat = confusion_matrix(np.array(ensemble_all_true), np.array(ensemble_all_pred))
overall_class_rep = classification_report(np.array(ensemble_all_true), np.array(ensemble_all_pred), digits=4)
print("Overall Confusion Matrix:")
print(overall_conf_mat)
print("\nOverall Classification Report:")
print(overall_class_rep)

# 计算多分类 ROC AUC
all_true_arr = np.array(ensemble_all_true)
all_pred_proba_arr = np.array(ensemble_all_pred_proba)
try:
    overall_roc_auc = roc_auc_score(all_true_arr, all_pred_proba_arr, multi_class='ovr')
    print(f"\nOverall ROC AUC Score (One-vs-Rest): {overall_roc_auc:.4f}")
except Exception as e:
    print(f"\nROC AUC 计算失败: {str(e)}")

# ---------------------------
# 6. 可视化平均特征重要性（仅针对 XGBoost）
# ---------------------------
num_folds = 5
avg_importance = {feat: score / num_folds for feat, score in aggregate_importance.items()}
importance_df = pd.DataFrame(list(avg_importance.items()), columns=['Feature', 'Importance'])
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("\nAverage Feature Importance (across folds):")
print(importance_df)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.title('Average Feature Importance from 5-Fold CV')
plt.gca().invert_yaxis()  # 将最高重要性的特征显示在顶部
plt.tight_layout()
plt.show()

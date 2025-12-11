# 文件名: main_HTNet_Baseline.py
import os
import numpy as np
import time
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
import random
from os import path

# 注意：保持使用 Baseline 的模型导入
from Model_Baseline import HTNet

# --- 全局设备设置 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# 工具函数 (严格保持一致)
# ==========================================

def setup_seed(seed):
    """固定随机种子以确保结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def strtobool(val):
    if isinstance(val, bool): return val
    val = str(val).lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'): return 1
    return 0


def get_dataset_source(subject_name):
    if subject_name.startswith('s') and subject_name[1:].isdigit():
        return 'SMIC'
    elif subject_name.startswith('sub'):
        return 'CASME II'
    else:
        return 'SAMM'


def confusionMatrix(gt, pred):
    """计算 UF1, UAR 和 混淆矩阵 (Baseline 特有逻辑)"""
    cm = confusion_matrix(gt, pred, labels=[0, 1, 2])
    f1_scores = []
    recalls = []
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(len(cm)):
            TP = cm[i, i]
            FP = np.sum(cm[:, i]) - TP
            FN = np.sum(cm[i, :]) - TP
            precision = np.nan_to_num(TP / (TP + FP))
            recall = np.nan_to_num(TP / (TP + FN))
            f1 = np.nan_to_num(2 * (precision * recall) / (precision + recall))
            f1_scores.append(f1)
            recalls.append(recall)
    UF1 = np.mean(f1_scores)
    UAR = np.mean(recalls)
    return UF1, UAR, cm


def main(config):
    # ==========================================
    # 统一超参数和设置
    # ==========================================
    CURRENT_SEED = config.seed
    setup_seed(CURRENT_SEED)

    learning_rate = 0.00005
    batch_size = 128
    epochs = 800

    # Baseline 使用标准 CE Loss
    loss_fn = nn.CrossEntropyLoss()

    # 动态文件夹名称
    weights_folder = f'baseline_weights_{CURRENT_SEED}'

    if not path.exists(weights_folder):
        os.makedirs(weights_folder, exist_ok=True)

    print('\n========================================')
    print('Baseline (CrossEntropy) Setup:')
    print(f'System: Running on device: {device}, Seed: {CURRENT_SEED}')
    print(f'Mode: GPU Optimized (Loading processed_data.pt)')
    print(f'Weights will be saved to: {weights_folder}')
    print(f'lr={learning_rate}, epochs={epochs}, batch_size={batch_size}\n')
    print('========================================')

    # --- 1. 加载预处理数据 (极速加载) ---
    data_file = 'processed_data.pt'
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please upload it.")
        return

    print(f"Loading {data_file} into RAM...")
    t_load = time.time()
    all_data = torch.load(data_file)
    print(f"Data loaded in {time.time() - t_load:.2f}s. Total samples: {len(all_data)}")

    # 获取 Subject 列表 (排序)
    subName = sorted(list(set([d['subject'] for d in all_data])))

    # 统计容器
    cumulative_gt = []
    cumulative_pred = []
    dataset_results = {
        'SMIC': {'gt': [], 'pred': []},
        'CASME II': {'gt': [], 'pred': []},
        'SAMM': {'gt': [], 'pred': []}
    }

    t_start = time.time()

    # --- LOSO 循环 ---
    for n_subName in subName:
        print(f"Subject: {n_subName}")

        # --- 内存筛选 ---
        train_list = [d for d in all_data if d['subject'] == n_subName and d['split'] == 'train']
        test_list = [d for d in all_data if d['subject'] == n_subName and d['split'] == 'test']

        if len(train_list) == 0 or len(test_list) == 0:
            print(f"Skipping {n_subName}: Insufficient data.")
            continue

        # 转换为 Tensor
        x_train_t = torch.stack([item['data'] for item in train_list])
        y_train_t = torch.tensor([item['label'] for item in train_list], dtype=torch.long)

        x_test_t = torch.stack([item['data'] for item in test_list])
        y_test_t = torch.tensor([item['label'] for item in test_list], dtype=torch.long)

        # DataLoader
        dataset_train = TensorDataset(x_train_t, y_train_t)
        train_dl = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

        dataset_test = TensorDataset(x_test_t, y_test_t)
        test_dl = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

        weight_path = os.path.join(weights_folder, n_subName + '.pth')

        # --- 模型初始化 (Baseline) ---
        model = HTNet(
            image_size=28,
            patch_size=7,
            dim=256,
            heads=3,
            num_hierarchies=3,
            block_repeats=(2, 2, 10),
            num_classes=3  # Baseline 直接输出分类 logits
        ).to(device)

        # --- 智能加载逻辑 ---
        do_training = config.train
        if os.path.exists(weight_path):
            try:
                model.load_state_dict(torch.load(weight_path, map_location=device))
                print(f"[Smart Load] Loaded weights for {n_subName}. Skipping training.")
                do_training = False
            except Exception as e:
                print(f"[Smart Load] Corrupted ({e}). Retraining.")
                do_training = True
        else:
            if not config.train:
                print(f"[Test Mode] No weights for {n_subName}. Skipping.")
                continue

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_accuracy = 0
        best_preds_sub = []
        current_gt = y_test_t.tolist()

        # --- 训练/测试循环 ---
        loops = range(1, epochs + 1) if do_training else [1]

        for epoch in loops:
            if do_training:
                model.train()
                for batch in train_dl:
                    optimizer.zero_grad()
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    yhat = model(x)
                    loss = loss_fn(yhat, y)
                    loss.backward()
                    optimizer.step()

            # Testing
            model.eval()
            temp_preds = []

            with torch.no_grad():
                for batch in test_dl:
                    x = batch[0].to(device)
                    yhat = model(x)
                    # Baseline 直接取 max
                    preds = torch.max(yhat, 1)[1]
                    temp_preds.extend(preds.tolist())

            val_acc = np.mean(np.array(temp_preds) == np.array(current_gt))

            if val_acc >= best_accuracy:
                best_accuracy = val_acc
                best_preds_sub = temp_preds
                if do_training:
                    torch.save(model.state_dict(), weight_path)

        # 结果记录
        if not best_preds_sub:
            best_preds_sub = [0] * len(current_gt)

        print(f"Best Predicted : {best_preds_sub}")
        print(f"Ground Truth : {current_gt}")

        cumulative_gt.extend(current_gt)
        cumulative_pred.extend(best_preds_sub)

        # 分数据集记录
        ds_name = get_dataset_source(n_subName)
        dataset_results[ds_name]['gt'].extend(current_gt)
        dataset_results[ds_name]['pred'].extend(best_preds_sub)

        curr_uf1, curr_uar, _ = confusionMatrix(cumulative_gt, cumulative_pred)
        print(f'Evaluation until this subject: best UF1: {curr_uf1:.4g} | best UAR: {curr_uar:.4g}')

    # ==========================================
    # Final Results
    # ==========================================
    print("\n========================================")
    print("Final Results (Like Table 2)")
    print("========================================")

    if len(cumulative_gt) > 0:
        full_uf1, full_uar, full_cm = confusionMatrix(cumulative_gt, cumulative_pred)
        print(f"Full (Composite): UF1={full_uf1:.4f}, UAR={full_uar:.4f}")

        # Dataset Breakdown
        for ds in ['SMIC', 'CASME II', 'SAMM']:
            gt_ds = dataset_results[ds]['gt']
            pred_ds = dataset_results[ds]['pred']
            if len(gt_ds) > 0:
                ds_uf1, ds_uar, _ = confusionMatrix(gt_ds, pred_ds)
                print(f"{ds:<9} : UF1={ds_uf1:.4f}, UAR={ds_uar:.4f}")
            else:
                print(f"{ds:<9} : No Data")

        print("========================================")
        print("")
        print("=== Confusion Matrix (Full) ===")
        print(full_cm)
        print("==============================")
        print(f"Total Samples: {len(cumulative_gt)}")
        print(f"Total Time Taken: {time.time() - t_start:.2f} s")
    else:
        print("No predictions made.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=strtobool, default='True')
    parser.add_argument('--seed', type=int, default=1)
    config = parser.parse_args()
    main(config)
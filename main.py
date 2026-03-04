import os
import numpy as np
import time
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import argparse
import random
from os import path

# 导入模型和Loss
from Model_compare import HTNet
import Compare_losses

# --- 全局设备设置 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 随机种子控制模块 ---
def set_seed(seed):
    """固定随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def strtobool(val):
    if isinstance(val, bool): return val
    val = str(val).lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'): return 1
    return 0

# --- 评价指标工具 ---
def confusionMatrix(gt, pred):
    try:
        conf_mat = confusion_matrix(gt, pred).ravel()
        if len(conf_mat) == 4:
            TN, FP, FN, TP = conf_mat
        else:
            return 0.0, 0.0
        f1_score = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
        num_samples = len([x for x in gt if x == 1])
        average_recall = TP / num_samples if num_samples > 0 else 0
        return f1_score, average_recall
    except Exception:
        return 0.0, 0.0

def recognition_evaluation(final_gt, final_pred):
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}
    f1_list = []
    ar_list = []
    try:
        for emotion_index in label_dict.values():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]

            if 1 in gt_recog:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)

        UF1 = np.mean(f1_list) if f1_list else 0
        UAR = np.mean(ar_list) if ar_list else 0
        return UF1, UAR
    except:
        return 0.0, 0.0

def main(config):
    # === 参数设置 ===
    learning_rate = 0.00005
    batch_size = 128
    epochs = 800
    embedding_dim = 512
    nb_classes = 3

    CURRENT_SEED = config.seed
    set_seed(CURRENT_SEED)

    save_dir = f'Anchor_weights_{CURRENT_SEED}'
    if not path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print('=' * 40)
    print(f'System: Running on device: {device}, Seed: {CURRENT_SEED}')
    print(f'Mode: GPU Optimized (Strict Blind LOSO)')
    print('=' * 40)

    data_file = 'processed_data.pt'
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please upload it.")
        return

    print(f"Loading {data_file} into RAM...")
    t_load = time.time()
    all_data = torch.load(data_file)
    print(f"Data loaded in {time.time() - t_load:.2f}s. Total samples: {len(all_data)}")

    all_subjects = sorted(list(set([d['subject'] for d in all_data])))

    criterion = Compare_losses.Proxy_Anchor(nb_classes=nb_classes, sz_embed=embedding_dim, mrg=0.1, alpha=32).to(device)

    print(f'Model: HTNet + Proxy Anchor Loss')
    print(f'lr={learning_rate}, epochs={epochs}, BS={batch_size}\n')

    total_gt = []
    best_total_pred = []

    smic_gt, smic_pred = [], []
    casme_gt, casme_pred = [], []
    samm_gt, samm_pred = [], []

    t_start = time.time()

    for n_subName in all_subjects:
        print(f'>>> Processing Subject: {n_subName}')

        # =========================================================================
        # 核心修复 1：真正的 LOSO 划分 (Identity Leakage 修复)
        # 用除被试以外的所有人做训练，仅用当前被试做测试。移除了错误的 'split' 过滤。
        # =========================================================================
        train_list = [d for d in all_data if d['subject'] != n_subName]
        test_list = [d for d in all_data if d['subject'] == n_subName]

        if len(train_list) == 0 or len(test_list) == 0:
            print(f"Skipping {n_subName}: Insufficient data.")
            continue

        x_train_t = torch.stack([item['data'] for item in train_list])
        y_train_t = torch.tensor([item['label'] for item in train_list], dtype=torch.long)

        x_test_t = torch.stack([item['data'] for item in test_list])
        y_test_t = torch.tensor([item['label'] for item in test_list], dtype=torch.long)

        train_dl = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True, num_workers=0)
        test_dl = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=batch_size, shuffle=False, num_workers=0)

        weight_path = os.path.join(save_dir, n_subName + '.pth')

        model = HTNet(
            image_size=28, patch_size=7, dim=256, heads=3,
            num_hierarchies=3, block_repeats=(2, 2, 10),
            embedding_size=embedding_dim
        ).to(device)

        do_training = config.train
        if os.path.exists(weight_path):
            try:
                checkpoint = torch.load(weight_path, map_location=device)
                model.load_state_dict(checkpoint['model'])
                if 'proxies' in checkpoint:
                    criterion.load_state_dict(checkpoint['proxies'])
                print(f"[Smart Load] Loaded weights for {n_subName}. Skipping training.")
                do_training = False
            except Exception as e:
                print(f"[Smart Load] Corrupted ({e}). Retraining.")
                do_training = True
        else:
            if not config.train:
                print(f"[Test Mode] No weights for {n_subName}. Skipping.")
                continue

        param_groups = [
            {'params': model.parameters(), 'lr': learning_rate},
            {'params': criterion.parameters(), 'lr': learning_rate * 100} 
        ]
        optimizer = torch.optim.AdamW(param_groups)

        current_gt = y_test_t.tolist()

        # =========================================================================
        # 核心修复 2：彻底剥离评估与训练，实行“盲测” (Data Leakage 修复)
        # =========================================================================
        if do_training:
            for epoch in range(1, epochs + 1):
                model.train()
                for bx, by in train_dl:
                    optimizer.zero_grad()
                    bx, by = bx.to(device), by.to(device)
                    embedding = model(bx)
                    loss = criterion(embedding, by)
                    loss.backward()
                    optimizer.step()
            
            # 训练彻底完成后，只保存最后一轮（或收敛状态）的权重
            torch.save({
                'model': model.state_dict(),
                'proxies': criterion.state_dict()
            }, weight_path)

        # 盲测评估：一锤定音
        model.eval()
        temp_preds = []
        with torch.no_grad():
            for bx, by in test_dl:
                bx, by = bx.to(device), by.to(device)
                embedding = model(bx)

                proxies_norm = F.normalize(criterion.proxies, p=2, dim=1)
                similarity = torch.matmul(embedding, proxies_norm.t())
                predictions = torch.argmax(similarity, dim=1)
                temp_preds.extend(predictions.tolist())

        # 最终成绩不挑不拣
        best_preds_sub = temp_preds
        best_accuracy = np.mean(np.array(best_preds_sub) == np.array(current_gt))

        print(f'  Result: Acc={best_accuracy:.4f} (Retrained: {do_training})')

        total_gt.extend(current_gt)
        best_total_pred.extend(best_preds_sub)

        if n_subName.startswith('sub'):
            casme_gt.extend(current_gt); casme_pred.extend(best_preds_sub)
        elif n_subName.startswith('s') and len(n_subName) > 1 and n_subName[1].isdigit():
            smic_gt.extend(current_gt); smic_pred.extend(best_preds_sub)
        else:
            samm_gt.extend(current_gt); samm_pred.extend(best_preds_sub)

    print('\n' + '=' * 40)
    print('          Final Results (Proxy Anchor)')
    print('=' * 40)

    if len(total_gt) > 0:
        uf1_full, uar_full = recognition_evaluation(total_gt, best_total_pred)
        print(f"Full (Composite): UF1={uf1_full:.4f}, UAR={uar_full:.4f}")

        if len(smic_gt) > 0:
            u1, ua = recognition_evaluation(smic_gt, smic_pred)
            print(f"SMIC           : UF1={u1:.4f}, UAR={ua:.4f}")
        if len(casme_gt) > 0:
            u1, ua = recognition_evaluation(casme_gt, casme_pred)
            print(f"CASME II       : UF1={u1:.4f}, UAR={ua:.4f}")
        if len(samm_gt) > 0:
            u1, ua = recognition_evaluation(samm_gt, samm_pred)
            print(f"SAMM           : UF1={u1:.4f}, UAR={ua:.4f}")

        print('=' * 40)
        print(f'Total Time Taken: {time.time() - t_start:.2f} s')
    else:
        print("No predictions made.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=strtobool, default='True')
    parser.add_argument('--seed', type=int, default=1)
    config = parser.parse_args()
    main(config)

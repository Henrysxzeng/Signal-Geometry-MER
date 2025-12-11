# 文件名: main_Arc_Cos.py
import os
import numpy as np
import time
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import argparse
import random
# 注意：移除了 cv2, pandas, re, facenet_pytorch，因为数据已处理好

from Model_compare import HTNet
import Compare_losses

# --- 全局设备设置 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- 随机种子设置 (保持不变) ---
def set_seed(seed):
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


# --- 评价指标工具 (保持不变) ---
def confusionMatrix(gt, pred):
    try:
        TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
        f1_score = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
        num_samples = len([x for x in gt if x == 1])
        average_recall = TP / num_samples if num_samples > 0 else 0
        return f1_score, average_recall
    except ValueError:
        return 0.0, 0.0
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
    # 1. 接收参数
    LOSS_TYPE = config.loss_type
    CURRENT_SEED = config.seed
    BATCH_SIZE = 128

    set_seed(CURRENT_SEED)

    relative_save_dir = f'weights_{LOSS_TYPE}_{CURRENT_SEED}'
    SAVE_DIR = os.path.abspath(relative_save_dir)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)

    print('\n========================================')
    print(f'System: Running on device: {device}, Seed: {CURRENT_SEED}')
    print(f'Mode: GPU Optimized (Loading Preprocessed Data)')
    print(f'Model: HTNet + {LOSS_TYPE}')
    print('========================================')

    # --- 2. 加载预处理好的数据 (替代了原代码中漫长的 IO 过程) ---
    data_file = 'processed_data.pt'
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please run preprocess_local.py first!")
        return

    print(f"Loading {data_file} into RAM...")
    t_load = time.time()
    # 加载数据列表，包含 {'subject', 'split', 'label', 'data'}
    all_data_memory = torch.load(data_file)
    print(f"Data loaded in {time.time() - t_load:.2f}s. Total samples: {len(all_data_memory)}")

    # 获取所有 Subject 列表 (并排序，保证循环顺序与原代码一致)
    subName = sorted(list(set([d['subject'] for d in all_data_memory])))

    # 3. 初始化 Loss 类
    try:
        if LOSS_TYPE == 'ArcFace':
            LossClass = Compare_losses.ArcFace
            m_param = 0.50
        elif LOSS_TYPE == 'CosFace':
            LossClass = Compare_losses.CosFace
            m_param = 0.35
        else:
            raise ValueError(f"Unknown loss type: {LOSS_TYPE}")
    except AttributeError:
        print(f"Error: Loss class {LOSS_TYPE} not found.")
        return

    total_gt, total_pred = [], []
    smic_gt, smic_pred = [], []
    casme_gt, casme_pred = [], []
    samm_gt, samm_pred = [], []

    t_start = time.time()

    # --- LOSO 循环 ---
    for n_subName in subName:
        print(f"\nSubject: {n_subName}")

        # --- 内存筛选 (替代原代码的文件遍历) ---
        # 这种列表推导式非常快
        train_list = [d for d in all_data_memory if d['subject'] == n_subName and d['split'] == 'train']
        test_list = [d for d in all_data_memory if d['subject'] == n_subName and d['split'] == 'test']

        if len(train_list) == 0 or len(test_list) == 0:
            print(f"Skipping {n_subName} due to empty data.")
            continue

        # 组装 Tensor
        # torch.stack 将 list of (C,H,W) -> (B,C,H,W)，等同于原代码的 permute 结果
        x_train_t = torch.stack([item['data'] for item in train_list])
        y_train_t = torch.tensor([item['label'] for item in train_list], dtype=torch.long)

        x_test_t = torch.stack([item['data'] for item in test_list])
        y_test_t = torch.tensor([item['label'] for item in test_list], dtype=torch.long)

        # DataLoader (因为 Seed 一致且数据顺序一致，Shuffle 结果也会一致)
        train_dl = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
        test_dl = DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

        # 模型初始化
        model = HTNet(
            image_size=28, patch_size=7, dim=256, heads=3,
            num_hierarchies=3, block_repeats=(2, 2, 10),
            embedding_size=512
        ).to(device)

        # Loss 初始化
        criterion = LossClass(in_features=512, out_features=3, s=30.0, m=m_param).to(device)

        weight_path = os.path.join(SAVE_DIR, n_subName + '.pth')

        optimizer = torch.optim.AdamW([
            {'params': model.parameters(), 'lr': 5e-5},
            {'params': criterion.parameters(), 'lr': 5e-4}
        ])

        # ==========================================
        # 权重检查逻辑 (保持不变)
        # ==========================================
        need_train = config.train

        if os.path.exists(weight_path):
            print(f"[{n_subName}] Found existing weights. Attempting to load...")
            try:
                checkpoint = torch.load(weight_path)
                model.load_state_dict(checkpoint['model'])
                criterion.load_state_dict(checkpoint['loss'])
                print(f"[{n_subName}] Weights & Loss loaded successfully. Skipping training.")
                need_train = False
            except Exception as e:
                print(f"[{n_subName}] Weights corrupted ({e}). Will retrain.")
                need_train = True
        else:
            print(f"[{n_subName}] No weights found. Will train.")
            need_train = True

        best_preds = []

        # === 训练阶段 ===
        if need_train:
            best_acc = 0
            epochs = 800
            for epoch in range(1, epochs + 1):
                model.train()
                for bx, by in train_dl:
                    optimizer.zero_grad()
                    bx, by = bx.to(device), by.to(device)
                    embed = model(bx)
                    loss = criterion(embed, by)
                    loss.backward()
                    optimizer.step()

                # 验证
                model.eval()
                temp_correct = 0
                temp_total = 0
                with torch.no_grad():
                    norm_weight = F.normalize(criterion.weight, p=2, dim=1)
                    for bx, by in test_dl:
                        bx, by = bx.to(device), by.to(device)
                        embed = model(bx)
                        norm_embed = F.normalize(embed, p=2, dim=1)
                        logits = F.linear(norm_embed, norm_weight)
                        preds = torch.argmax(logits, dim=1)
                        temp_correct += (preds == by).sum().item()
                        temp_total += by.shape[0]

                val_acc = temp_correct / temp_total if temp_total > 0 else 0

                if val_acc >= best_acc:
                    best_acc = val_acc
                    torch.save({
                        'model': model.state_dict(),
                        'loss': criterion.state_dict()
                    }, weight_path)

            print(f"Training finished. Loading best weights (Acc: {best_acc:.4f})...")
            checkpoint = torch.load(weight_path)
            model.load_state_dict(checkpoint['model'])
            criterion.load_state_dict(checkpoint['loss'])

        # === 统一推理阶段 ===
        model.eval()
        best_preds = []
        current_gt = []

        with torch.no_grad():
            norm_weight = F.normalize(criterion.weight, p=2, dim=1)
            for bx, by in test_dl:
                bx, by = bx.to(device), by.to(device)
                embed = model(bx)
                norm_embed = F.normalize(embed, p=2, dim=1)
                logits = F.linear(norm_embed, norm_weight)
                preds = torch.argmax(logits, dim=1)
                best_preds.extend(preds.tolist())
                current_gt.extend(by.tolist())

        print(f"Predicted : {best_preds}")
        total_pred.extend(best_preds)
        total_gt.extend(current_gt)

        # 分数据集统计
        if n_subName.startswith('sub'):
            casme_gt.extend(current_gt);
            casme_pred.extend(best_preds)
        elif n_subName.startswith('s') and len(n_subName) > 1 and n_subName[1].isdigit():
            smic_gt.extend(current_gt);
            smic_pred.extend(best_preds)
        else:
            samm_gt.extend(current_gt);
            samm_pred.extend(best_preds)

    print('\n' + '=' * 40)
    print(f'Final Results: {LOSS_TYPE} (Seed {CURRENT_SEED})')
    print('=' * 40)

    if len(total_gt) > 0:
        uf1, uar = recognition_evaluation(total_gt, total_pred)
        print(f"Full Composite : UF1={uf1:.4f}, UAR={uar:.4f}")
        if len(smic_gt) > 0:
            u1, ua = recognition_evaluation(smic_gt, smic_pred)
            print(f"SMIC           : UF1={u1:.4f}, UAR={ua:.4f}")
        if len(casme_gt) > 0:
            u1, ua = recognition_evaluation(casme_gt, casme_pred)
            print(f"CASME II       : UF1={u1:.4f}, UAR={ua:.4f}")
        if len(samm_gt) > 0:
            u1, ua = recognition_evaluation(samm_gt, samm_pred)
            print(f"SAMM           : UF1={u1:.4f}, UAR={ua:.4f}")
        print(f"Total Time: {time.time() - t_start:.2f}s")
    else:
        print("No predictions made.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=strtobool, default='True')
    parser.add_argument('--loss_type', type=str, default='ArcFace', help='ArcFace or CosFace')
    parser.add_argument('--seed', type=int, default=1)
    config = parser.parse_args()

    main(config)
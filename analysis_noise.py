import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import argparse
from sklearn.metrics import f1_score, recall_score

# 导入模型定义
from Model_Baseline import HTNet as HTNet_Baseline
from Model_Anchor import HTNet as HTNet_Anchor
# 导入 Loss 定义以获取类中心结构
from Compare_losses import Proxy_Anchor, ArcFace, CosFace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def add_noise(images, noise_level):
    if noise_level == 0: return images
    return images + torch.randn_like(images) * noise_level


def calculate_metrics(gt, pred):
    return f1_score(gt, pred, average='macro'), recall_score(gt, pred, average='macro')


def detect_depth_and_get_model(state_dict, models_dict):
    """自动检测 8层 或 10层"""
    is_10_layers = any('layers.2.0.layers.8.' in k for k in state_dict.keys())
    return models_dict['10_layers' if is_10_layers else '8_layers']


def run_loso_evaluation(all_data, subjects, models_dict, weight_folder, noise_level, model_type, criterion=None):
    total_preds = []
    total_gt = []

    for sub in subjects:
        sub_test_data = [d for d in all_data if d['subject'] == sub and d['split'] == 'test']
        if len(sub_test_data) == 0: continue

        x_sub = torch.stack([d['data'] for d in sub_test_data]).to(device)
        y_sub = torch.tensor([d['label'] for d in sub_test_data], dtype=torch.long).to(device)
        x_sub = add_noise(x_sub, noise_level)

        weight_path = os.path.join(weight_folder, f"{sub}.pth")
        if not os.path.exists(weight_path): continue

        try:
            checkpoint = torch.load(weight_path, map_location=device)

            # --- 1. 加载主干网络 ---
            if model_type == 'baseline':
                state_dict = checkpoint
            else:
                # Anchor/CosFace/ArcFace 通常保存为 {'model':..., 'loss':...}
                state_dict = checkpoint.get('model', checkpoint)  # 兼容性写法

            model = detect_depth_and_get_model(state_dict, models_dict)
            model.load_state_dict(state_dict)
            model.eval()

            # --- 2. 加载 Head/Proxies 并推理 ---
            with torch.no_grad():
                if model_type == 'baseline':
                    logits = model(x_sub)
                    preds = torch.argmax(logits, dim=1)

                else:  # Metric Learning Methods
                    embeddings = model(x_sub)  # [Batch, 512]

                    # 获取类中心 (Centers/Proxies)
                    centers = None
                    if model_type == 'anchor':
                        if 'proxies' in checkpoint:
                            criterion.load_state_dict(checkpoint['proxies'])
                            centers = criterion.proxies
                    elif model_type in ['cosface', 'arcface']:
                        # 假设保存的 key 是 'loss' 或 'head'
                        loss_state = checkpoint.get('loss', checkpoint.get('head', None))
                        if loss_state:
                            criterion.load_state_dict(loss_state)
                            # ArcFace/CosFace 的类中心通常叫 .weight
                            centers = criterion.weight

                    if centers is not None:
                        # 计算余弦相似度进行分类
                        # 这里的 centers shape 应该是 [3, 512]
                        # ArcFace/CosFace 的 weight 也是 [out_features, in_features] 即 [3, 512]
                        P = F.normalize(centers, p=2, dim=1)
                        E = F.normalize(embeddings, p=2, dim=1)
                        sims = torch.matmul(E, P.t())
                        preds = torch.argmax(sims, dim=1)
                    else:
                        print(f"Warning: No centers found for {model_type} subject {sub}")
                        continue

            total_preds.extend(preds.cpu().numpy())
            total_gt.extend(y_sub.cpu().numpy())

        except Exception as e:
            # print(f"Error {sub}: {e}")
            continue

    return total_gt, total_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=3407)  # 默认用表现最好的 3407
    args = parser.parse_args()
    seed = args.seed

    # === 文件夹路径配置 (请根据实际情况修改) ===
    folders = {
        'baseline': f'baseline_weights_{seed}',
        'anchor': f'Aeights_{seed}',
        'cosface': f'weights_CosFace_{seed}',  # 假设的文件夹名，请确认！
        'arcface': f'weights_ArcFace_{seed}'  # 假设的文件夹名，请确认！
    }

    print(f"Loading data for Seed {seed}...")
    if not os.path.exists('processed_data.pt'): return
    all_data = torch.load('processed_data.pt')
    all_subjects = sorted(list(set([d['subject'] for d in all_data])))

    # === 初始化模型 ===
    # 1. Backbones
    base_models = {
        '8_layers': HTNet_Baseline(image_size=28, patch_size=7, dim=256, heads=3, num_hierarchies=3,
                                   block_repeats=(2, 2, 8), num_classes=3).to(device),
        '10_layers': HTNet_Baseline(image_size=28, patch_size=7, dim=256, heads=3, num_hierarchies=3,
                                    block_repeats=(2, 2, 10), num_classes=3).to(device)
    }
    metric_models = {
        '8_layers': HTNet_Anchor(image_size=28, patch_size=7, dim=256, heads=3, num_hierarchies=3,
                                 block_repeats=(2, 2, 8), embedding_size=512).to(device),
        '10_layers': HTNet_Anchor(image_size=28, patch_size=7, dim=256, heads=3, num_hierarchies=3,
                                  block_repeats=(2, 2, 10), embedding_size=512).to(device)
    }

    # 2. Loss Headers (用于提取 Centers)
    # 注意：in_features=512, out_features=3
    heads = {
        'anchor': Proxy_Anchor(nb_classes=3, sz_embed=512).to(device),
        'cosface': CosFace(in_features=512, out_features=3).to(device),
        'arcface': ArcFace(in_features=512, out_features=3).to(device)
    }

    # === 实验循环 ===
    noise_levels = [0, 10, 20, 30, 40, 50, 80, 100]
    results = {k: [] for k in folders.keys()}

    print(f"\n{'Noise':<8} | {'Base':<8} | {'Anchor':<8} | {'CosFace':<8} | {'ArcFace':<8}")
    print("-" * 55)

    for lvl in noise_levels:
        row_str = f"{lvl:<8} | "

        for method in ['baseline', 'anchor', 'cosface', 'arcface']:
            folder = folders[method]
            if not os.path.exists(folder):
                results[method].append(0)
                row_str += f"{'N/A':<8} | "
                continue

            # 选择正确的模型架构和 Loss Head
            models = base_models if method == 'baseline' else metric_models
            head = heads.get(method)

            gt, pred = run_loso_evaluation(
                all_data, all_subjects, models, folder, lvl, method, head
            )

            if len(gt) > 0:
                uf1, _ = calculate_metrics(gt, pred)
                results[method].append(uf1)
                row_str += f"{uf1:.4f}   | "
            else:
                results[method].append(0)
                row_str += f"{'Err':<8} | "

        print(row_str)

    # === 绘图 ===
    plt.figure(figsize=(10, 7))
    styles = {
        'baseline': ('gray', 'o--', 'Baseline (CE)'),
        'anchor': ('#D62728', 's-', 'Ours (Anchor)'),
        'cosface': ('#1f77b4', '^-.', 'CosFace'),
        'arcface': ('#2ca02c', 'v-.', 'ArcFace')
    }

    for method, res in results.items():
        if sum(res) == 0: continue  # 跳过没跑通的
        c, m, l = styles[method]
        plt.plot(noise_levels, res, marker=m, color=c, linewidth=2, label=l, alpha=0.9)

    plt.xlabel('Gaussian Noise Level ($\sigma$)', fontsize=12)
    plt.ylabel('Unweighted F1 Score (UF1)', fontsize=12)
    plt.title(f'Noise Robustness Comparison (Seed {seed})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.savefig(f'noise_robustness_all_seed_{seed}.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to noise_robustness_all_seed_{seed}.png")


if __name__ == '__main__':
    main()
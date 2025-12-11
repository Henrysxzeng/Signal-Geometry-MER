import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset, DataLoader
import os
import torch.nn as nn
import torch.nn.functional as F

# 导入模型 (请确保目录下有 Model_Baseline.py 和 Model_Anchor.py)
try:
    from Model_Baseline import HTNet as HTNet_Baseline
    from Model_compare import HTNet as HTNet_Anchor
except ImportError:
    print(
        "Error: Model files not found. Please ensure Model_Baseline.py and Model_compare.py are in the working directory.")
    exit()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model_instances():
    """初始化所有可能的模型架构"""
    return {
        'baseline_8': HTNet_Baseline(image_size=28, patch_size=7, dim=256, heads=3, num_hierarchies=3,
                                     block_repeats=(2, 2, 8), num_classes=3).to(device),
        'baseline_10': HTNet_Baseline(image_size=28, patch_size=7, dim=256, heads=3, num_hierarchies=3,
                                      block_repeats=(2, 2, 10), num_classes=3).to(device),
        'anchor_8': HTNet_Anchor(image_size=28, patch_size=7, dim=256, heads=3, num_hierarchies=3,
                                 block_repeats=(2, 2, 8), embedding_size=512).to(device),
        'anchor_10': HTNet_Anchor(image_size=28, patch_size=7, dim=256, heads=3, num_hierarchies=3,
                                  block_repeats=(2, 2, 10), embedding_size=512).to(device)
    }


def detect_depth_and_get_model(state_dict, models_pool, model_type):
    """自动检测权重层数"""
    is_10_layers = any('layers.2.0.layers.8.' in k for k in state_dict.keys())
    key = f"{model_type}_{'10' if is_10_layers else '8'}"
    return models_pool[key]


def get_features_from_model(x_sub, model, model_type):
    model.eval()
    # Baseline 需要移除 Linear 层取特征
    original_head = None
    if model_type == 'baseline':
        original_head = model.mlp_head
        # 取 Sequential 的前两层 (LayerNorm + Reduce)
        # 假设 mlp_head 结构为: [LayerNorm, Reduce, Linear]
        model.mlp_head = nn.Sequential(*list(original_head.children())[:-1])

    with torch.no_grad():
        feat = model(x_sub)
        if len(feat.shape) > 2:
            feat = feat.view(feat.size(0), -1)

    # 恢复模型结构
    if model_type == 'baseline':
        model.mlp_head = original_head

    return feat.cpu().numpy()


def collect_global_features(all_data, subjects, models_pool, weight_folder, model_type, sub_to_id):
    """
    收集特征，同时返回 表情标签(y_emo) 和 受试者标签(y_sub)
    """
    all_features = []
    all_emo_labels = []
    all_sub_labels = []

    print(f"Collecting features for {model_type.upper()} from {weight_folder}...")

    if not os.path.exists(weight_folder):
        print(f"Warning: Folder {weight_folder} does not exist.")
        return None, None, None

    for sub in subjects:
        sub_test_data = [d for d in all_data if d['subject'] == sub and d['split'] == 'test']
        if len(sub_test_data) == 0: continue

        # 确保数据格式正确
        try:
            x_sub = torch.stack([d['data'] for d in sub_test_data]).to(device)
        except Exception:
            # 处理可能是 numpy 数组的情况
            x_sub = torch.stack(
                [torch.from_numpy(d['data']) if isinstance(d['data'], np.ndarray) else d['data'] for d in
                 sub_test_data]).to(device)

        y_sub_emo = [d['label'] for d in sub_test_data]

        weight_path = os.path.join(weight_folder, f"{sub}.pth")
        if not os.path.exists(weight_path): continue

        try:
            checkpoint = torch.load(weight_path, map_location=device)

            if model_type == 'baseline':
                state_dict = checkpoint
            else:
                state_dict = checkpoint['model']

            model = detect_depth_and_get_model(state_dict, models_pool, model_type)
            model.load_state_dict(state_dict)

            feats = get_features_from_model(x_sub, model, model_type)

            all_features.append(feats)
            all_emo_labels.extend(y_sub_emo)
            # 记录受试者ID
            all_sub_labels.extend([sub_to_id[sub]] * len(feats))

        except Exception as e:
            # print(f"Error {sub}: {e}")
            continue

    if len(all_features) == 0: return None, None, None
    return np.concatenate(all_features), np.array(all_emo_labels), np.array(all_sub_labels)


# --- 核心绘图修改 ---

def set_tight_bounds(X_emb, padding_factor=0.05):
    """
    根据数据的 1% 和 99% 分位数设置坐标轴范围，
    从而切除极远的离群点，减少留白，使主体更密集。
    """
    x_min, x_max = np.percentile(X_emb[:, 0], [1, 99])
    y_min, y_max = np.percentile(X_emb[:, 1], [1, 99])

    x_range = x_max - x_min
    y_range = y_max - y_min

    plt.xlim(x_min - x_range * padding_factor, x_max + x_range * padding_factor)
    plt.ylim(y_min - y_range * padding_factor, y_max + y_range * padding_factor)


def plot_tsne_emotion(X_emb, y, title, save_name):
    """画按表情分类的图 (用于论文 Figure 3)"""

    # 设置字体 (Times New Roman)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    plt.figure(figsize=(10, 10))  # 保持正方形

    # 论文配色：Negative(蓝), Positive(绿), Surprise(红)
    colors = ['#377eb8', '#4daf4a', '#e41a1c']
    labels_name = ['Negative', 'Positive', 'Surprise']

    for i in range(3):
        indices = np.where(y == i)
        plt.scatter(X_emb[indices, 0], X_emb[indices, 1],
                    c=colors[i], label=labels_name[i],
                    # === 修改处：加大点的大小，提高不透明度，移除白边 ===
                    s=80,  # 增大点的大小 (原40 -> 80)
                    alpha=0.85,  # 提高不透明度，让颜色更实 (原0.7 -> 0.85)
                    edgecolors='none'  # 移除白边，让点更像一个整体
                    )

    # === 修改处：设置紧凑的边界，去除空白 ===
    set_tight_bounds(X_emb)

    plt.xticks([])
    plt.yticks([])

    # 调整图例
    plt.legend(fontsize=18, loc='upper right', framealpha=0.9, frameon=True, edgecolor='gray')

    plt.title(title, fontsize=24, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Saved {save_name}")


def plot_tsne_subject(X_emb, sub_y, title, save_name):
    """画按受试者分类的图 (用于分析 Domain Gap)"""

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    plt.figure(figsize=(12, 10))

    # 同样加大点并使其密集
    plt.scatter(X_emb[:, 0], X_emb[:, 1],
                c=sub_y, cmap='jet',
                s=60, alpha=0.75, edgecolors='none')  # 加大点

    # 设置紧凑边界
    set_tight_bounds(X_emb)

    plt.colorbar(label='Subject ID')
    plt.title(title, fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Saved {save_name}")


def main():
    seed = 3407  # 使用你选定的最佳种子

    baseline_folder = f'baseline_weights_{seed}'
    anchor_folder = f'Anchor_weights_{seed}'

    print("Loading data...")
    if not os.path.exists('processed_data.pt'):
        print("processed_data.pt not found. Please ensure the data file is present.")
        return

    # 增加 map_location 确保 CPU/GPU 兼容
    all_data = torch.load('processed_data.pt', map_location=device)

    # 建立受试者映射
    all_subjects = sorted(list(set([d['subject'] for d in all_data])))
    sub_to_id = {sub: i for i, sub in enumerate(all_subjects)}

    models_pool = get_model_instances()

    # =======================================================
    # 1. 生成 Baseline 全局图 (Figure 3a)
    # =======================================================
    feats_b, labels_b, _ = collect_global_features(all_data, all_subjects, models_pool, baseline_folder, 'baseline',
                                                   sub_to_id)
    if feats_b is not None:
        print(f"Running Baseline T-SNE ({len(feats_b)} samples)...")
        # 增加 perplexity 可以让聚类稍微紧凑一点，这里设为 40
        tsne_b = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=seed, perplexity=40).fit_transform(
            feats_b)
        plot_tsne_emotion(tsne_b, labels_b, "", "paper_fig_a_baseline.pdf")

    # =======================================================
    # 2. 生成 Ours 全局图 (Figure 3b) & Subject ID 图
    # =======================================================
    feats_a, labels_a, subs_a = collect_global_features(all_data, all_subjects, models_pool, anchor_folder, 'anchor',
                                                        sub_to_id)
    if feats_a is not None:
        print(f"Running Anchor T-SNE ({len(feats_a)} samples)...")
        # 增加 perplexity 到 40
        tsne_a = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=seed, perplexity=40).fit_transform(
            feats_a)

        # 图 3(b): 按表情上色
        plot_tsne_emotion(tsne_a, labels_a, "", "paper_fig_b_ours.pdf")

        # 分析图: 按人上色
        plot_tsne_subject(tsne_a, subs_a, "", "tsne_anchor_subject_3407.pdf")


if __name__ == '__main__':
    main()
# 文件名: preprocess_local.py
# 作用: [CPU任务] 负责读取图片、MTCNN检测、裁剪、拼接，最后打包成 Tensor。
# 目的: 生成 processed_data.pt，上传到服务器后可直接加载进内存。

import os
import numpy as np
import cv2
import pandas
import torch
import re
from facenet_pytorch import MTCNN


# --- 1. 辅助函数 (保持原逻辑) ---
def to_canonical_key(filename):
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]
    name = name.lower().strip()
    name = re.sub(r'_s\d+_', '_', name)
    name = name.replace(' ', '')
    return name


def get_face_coordinates(base_root):
    # 读取 CSV (路径需根据你本地实际情况，假设在当前目录下)
    csv_path = os.path.join(base_root, 'combined_3_class2_for_optical_flow.csv')
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return {}

    df = pandas.read_csv(csv_path)
    base_data_src = os.path.join(base_root, 'datasets', 'combined_datasets_whole')

    # 预处理通常在 CPU 或 本地 GPU 上跑
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Preprocessing using device: {device}")

    mtcnn = MTCNN(margin=0, image_size=28, select_largest=True, post_process=False, device=device)

    # 默认坐标 (与原代码一致)
    default_landmarks_float = np.array([[[9.528073, 11.062551], [21.396168, 10.919773], [15.380184, 17.380562],
                                         [10.255435, 22.121233], [20.583706, 22.25584]]])
    default_landmarks = default_landmarks_float[0].astype(int)

    face_block_coordinates = {}
    print("Step 1: Scanning CSV and detecting faces...")

    for i in range(len(df)):
        csv_imagename = str(df['imagename'][i])
        img_path = os.path.join(base_data_src, csv_imagename)
        canonical_key = to_canonical_key(csv_imagename)

        # 路径回退查找逻辑 (与原代码一致)
        if not os.path.exists(img_path):
            csv_sub = str(df['sub'][i])
            csv_filename = str(df['filename_o'][i])
            guess_name = f"{csv_sub}_{csv_filename}"
            if os.path.exists(os.path.join(base_data_src, guess_name + '.jpg')):
                img_path = os.path.join(base_data_src, guess_name + '.jpg')
            elif os.path.exists(os.path.join(base_data_src, guess_name + '.png')):
                img_path = os.path.join(base_data_src, guess_name + '.png')
            else:
                continue

        # 读取与检测
        img = cv2.imread(img_path)
        landmarks = default_landmarks

        if img is not None:
            face_apex = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            try:
                _, _, batch_landmarks = mtcnn.detect(face_apex, landmarks=True)
                if batch_landmarks is not None:
                    landmarks = np.clip(batch_landmarks[0], 7, 21).astype(int)
            except:
                pass

        face_block_coordinates[canonical_key] = landmarks
        if i % 200 == 0: print(f"  Processed {i} faces...", end='\r')

    print(f"\nFace coords ready: {len(face_block_coordinates)}")
    return face_block_coordinates


def process_optical_flow(base_root, coords_dict):
    flow_path = os.path.join(base_root, 'datasets', 'STSNet_whole_norm_u_v_os')
    print("Step 2: Processing Optical Flow Images...")

    if not os.path.exists(flow_path):
        print(f"Error: {flow_path} not found")
        return {}

    processed_flows = {}
    # 关键：使用 sorted 保证顺序与原代码的文件系统遍历顺序可能的一致性（虽然这里是用 key 索引，但保持习惯）
    flow_imgs = sorted(os.listdir(flow_path))

    for idx, n_img in enumerate(flow_imgs):
        key = to_canonical_key(n_img)
        if key not in coords_dict: continue

        img = cv2.imread(os.path.join(flow_path, n_img))
        if img is None: continue

        c = coords_dict[key]
        parts = []
        for k in range(5):
            cx, cy = c[k][0], c[k][1]
            crop = img[cx - 7:cx + 7, cy - 7:cy + 7]
            # 尺寸修正
            if crop.shape[0] != 14 or crop.shape[1] != 14:
                crop = cv2.resize(crop, (14, 14))
            parts.append(crop)
        processed_flows[key] = parts

        if idx % 500 == 0: print(f"  Processed {idx} flow images...", end='\r')

    print(f"\nFlow blocks ready: {len(processed_flows)}")
    return processed_flows


def main():
    base_root = '.'  # 假设脚本就在项目根目录

    # 1. 获取坐标
    coords = get_face_coordinates(base_root)

    # 2. 处理光流块
    flows = process_optical_flow(base_root, coords)

    # 3. 组装最终数据集 (模拟原代码的 Loop 结构)
    print("Step 3: Assembling Final Dataset...")
    main_path = os.path.join(base_root, 'datasets', 'three_norm_u_v_os')

    # 关键：sorted 确保 Subject 顺序与原代码一致
    subName = sorted(os.listdir(main_path))

    final_data_list = []

    for n_sub in subName:
        # 遍历 train 和 test
        for split_dir in ['u_train', 'u_test']:
            split_path = os.path.join(main_path, n_sub, split_dir)
            if not os.path.exists(split_path): continue

            # 遍历表情类别 (确保 sorted)
            expressions = sorted(os.listdir(split_path))
            for expr in expressions:
                img_folder = os.path.join(split_path, expr)
                if not os.path.isdir(img_folder): continue

                try:
                    label_int = int(expr)
                except:
                    continue

                # 遍历图片 (确保 sorted，保证进入 DataLoader 的顺序是确定的)
                imgs = sorted(os.listdir(img_folder))
                for fname in imgs:
                    key = to_canonical_key(fname)
                    if key not in flows: continue

                    parts = flows[key]

                    # --- 拼图 (严格复刻原代码逻辑) ---
                    l_eye_lips = cv2.hconcat([parts[0], parts[1]])
                    r_eye_lips = cv2.hconcat([parts[3], parts[4]])
                    full_img = cv2.vconcat([l_eye_lips, r_eye_lips])

                    # --- 格式转换 ---
                    # 原代码: np.array -> tensor -> permute(0,3,1,2)
                    # 这里: 单张图 (H,W,C) -> permute(2,0,1) -> (C,H,W)
                    # 结果在数学上是全等的
                    img_tensor = torch.from_numpy(full_img).float().permute(2, 0, 1)

                    # 存入列表
                    final_data_list.append({
                        'subject': n_sub,
                        'split': 'train' if split_dir == 'u_train' else 'test',
                        'label': label_int,
                        'data': img_tensor
                    })

    print(f"Total samples collected: {len(final_data_list)}")

    # 4. 保存文件
    save_name = 'processed_data.pt'
    print(f"Saving to {save_name} ...")
    torch.save(final_data_list, save_name)
    print("Done! Upload 'processed_data.pt' to your GPU server.")


if __name__ == '__main__':
    main()
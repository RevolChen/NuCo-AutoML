import pandas as pd
import os

base_dir = os.path.join('..', 'data', 'PAP', 'raw_data')
train_path = os.path.join(base_dir, 'train', 'train.csv')

def prepare_full_data():
    breed_path = os.path.join(base_dir, 'breed_labels.csv')
    color_path = os.path.join(base_dir, 'color_labels.csv')
    state_path = os.path.join(base_dir, 'state_labels.csv')

    output_path = os.path.join('..', 'data', 'PAP', 'data_final.csv')

    print("正在读取 CSV 文件...")
    # 读取主文件
    df = pd.read_csv(train_path, encoding='utf-8')

    # 读取映射文件
    breed_df = pd.read_csv(breed_path, encoding='utf-8')
    color_df = pd.read_csv(color_path, encoding='utf-8')
    state_df = pd.read_csv(state_path, encoding='utf-8')

    breed_map = dict(zip(breed_df['BreedID'], breed_df['BreedName']))
    color_map = dict(zip(color_df['ColorID'], color_df['ColorName']))
    state_map = dict(zip(state_df['StateID'], state_df['StateName']))

    print("正在执行 ID -> 文本 映射...")

    # 映射 Breed1 和 Breed2
    # fillna(0) 是为了防止有些 ID 是空值，astype(int) 确保是整数索引
    # map 之后，如果 ID 不在字典里（比如 0），会变成 NaN，我们用 "Unknown" 填充
    df['Breed1'] = df['Breed1'].map(breed_map).fillna('Unknown')
    df['Breed2'] = df['Breed2'].map(breed_map).fillna('Unknown')

    # 映射 Color1, Color2, Color3, State
    df['Color1'] = df['Color1'].map(color_map).fillna('Unknown')
    df['Color2'] = df['Color2'].map(color_map).fillna('Unknown')
    df['Color3'] = df['Color3'].map(color_map).fillna('Unknown')
    df['State'] = df['State'].map(state_map).fillna('Unknown')

    # Name 填充 "Unknown"
    df['Name'] = df['Name'].fillna('Unknown')

    # ================= 处理图片路径 (这也是 AutoM³L 需要的) =================
    print("正在生成图片路径...")
    img_dir = os.path.abspath(os.path.join(base_dir, 'train_images'))

    def get_image_paths(row):
        pet_id = row['PetID']
        # 容错处理：确保 PhotoAmt 是数字，如果是空值或无法转换则默认为 0
        try:
            amt = int(row['PhotoAmt'])
        except (ValueError, TypeError):
            amt = 0

        if amt <= 0:
            return None

        found_paths = []
        # 遍历从 1 到 PhotoAmt 的所有序号
        for i in range(1, amt + 1):
            # 构建文件名：PetID-序号.jpg
            img_name = f"{pet_id}-{i}.jpg"
            full_path = os.path.join(img_dir, img_name)

            # 再次确认文件确实存在（防止 PhotoAmt 记录有误）
            if os.path.exists(full_path):
                found_paths.append(full_path)

        # 如果找到了图片
        if found_paths:
            # 方案 A: 如果你想保留所有图片，用分号拼接
            # return ";".join(found_paths)
            # 方案 B: 如果你只想保留第一张主图（AutoM³L 这种框架通常只处理单图）
            return found_paths[0]
        else:
            return None

    df['Image_Path'] = df.apply(get_image_paths, axis=1)

    # 将 'AdoptionSpeed' 移动到最后一列
    target_col = 'AdoptionSpeed'
    if target_col in df.columns:
        # 获取除了目标列以外的所有列名
        feature_cols = [c for c in df.columns if c != target_col]
        # 重新组合：特征在前，目标在后
        new_order = feature_cols + [target_col]
        df = df[new_order]


    # 处理 Description 空值
    df['Description'] = df['Description'].fillna('')

    # 保存
    print(f"正在保存最终数据到: {output_path}")
    df.to_csv(output_path, index=False, encoding='utf-8')

    print("\n" + "=" * 20 + " 数据预览 " + "=" * 20)
    print(df.head(10).to_string())

if __name__ == "__main__":
    prepare_full_data()
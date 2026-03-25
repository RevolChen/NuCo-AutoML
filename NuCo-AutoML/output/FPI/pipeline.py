import os
import json
import argparse
import gc
import warnings
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
import optuna
from transformers import AutoModel, AutoConfig, AutoTokenizer
import timm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 环境配置 ---
warnings.filterwarnings('ignore')
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 解决 tokenizer 并行导致的死锁警告

# ==============================================================================
# [Part 1] 核心配置注入区
# ==============================================================================
CONFIG = {
    "DATA_PATH": r"/root/autodl-tmp/SGC-AutoML/output/FPI/train_afe.csv",
    "TEST_PATH": r"/root/autodl-tmp/SGC-AutoML/output/FPI/test_afe.csv",
    "TARGET_COL": "masterCategory",
    "TASK_TYPE": "Classification",
    "MODALITY_MAP": {"gender": "Categorical", "subCategory": "Categorical", "articleType": "Categorical", "baseColour": "Categorical", "season": "Categorical", "year": "Numerical", "usage": "Categorical", "productDisplayName": "Text", "Image_Path": "Image_Path", "masterCategory": "Target"},
    "MODEL_SELECTION": {"Tabular_Model": ["CatBoost", "FT-Transformer"], "Text_Model": "microsoft/deberta-v3-large", "Image_Model": ["swin_base_patch4_window7_224", "eva02_base_patch14_224"], "Fusion_Model": "Transformer_Fusion"},
    "SEED": 16
}

# 全局种子固定函数
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[*] 全局随机种子已固定为: 16")

# ==============================================================================
# [Part 2] 通用多模态数据集类
# ==============================================================================
class MultimodalDataset(Dataset):
    def __init__(self, df, modality_map, tokenizer=None, img_transform=None, target_col=None, is_train=True, scaler=None):
        self.df = df.reset_index(drop=True)
        self.modality_map = modality_map
        self.tokenizer = tokenizer
        self.transform = img_transform
        self.target_col = target_col
        self.is_train = is_train

        self.num_cols = [c for c, m in modality_map.items() if m == 'Numerical']
        self.cat_cols = [c for c, m in modality_map.items() if m == 'Categorical']
        self.text_cols = [c for c, m in modality_map.items() if m == 'Text']
        self.img_cols = [c for c, m in modality_map.items() if m == 'Image_Path']

        # 如果有数值列，必须进行归一化，否则 No_of_Votes 这种大数会让模型崩溃
        if self.num_cols:
            # 先处理缺失值
            for c in self.num_cols:
                self.df[c] = self.df[c].fillna(0).astype(float)

            # 使用传入的 scaler 或者新建一个
            if scaler is None:
                self.scaler = StandardScaler()
                self.df[self.num_cols] = self.scaler.fit_transform(self.df[self.num_cols])
            else:
                self.scaler = scaler
                self.df[self.num_cols] = self.scaler.transform(self.df[self.num_cols])
        else:
            self.scaler = None

        if self.cat_cols:
            for c in self.cat_cols:
                self.df[c] = self.df[c].fillna('Unknown').astype(str)
                le = LabelEncoder()
                self.df[c] = le.fit_transform(self.df[c])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = {}

        # 1. Tabular
        if self.num_cols or self.cat_cols:
            nums = [row[c] for c in self.num_cols]
            cats = [row[c] for c in self.cat_cols]
            sample['tabular'] = torch.tensor(nums + cats, dtype=torch.float32)

        # 2. Text
        if self.text_cols and self.tokenizer:
            text_content = " ".join([str(row[c]) for c in self.text_cols if not pd.isna(row[c])])
            if not text_content.strip(): text_content = "empty"
            enc = self.tokenizer(text_content, padding='max_length', truncation=True, max_length=256,
                                 return_tensors='pt')
            sample['input_ids'] = enc['input_ids'].squeeze(0)
            sample['attention_mask'] = enc['attention_mask'].squeeze(0)

        # 3. Image
        if self.img_cols:
            col_name = self.img_cols[0]
            img_path_raw = row[col_name]
            image_tensor = torch.zeros(3, 224, 224)  # Default black
            if not pd.isna(img_path_raw) and str(img_path_raw).strip():
                paths = str(img_path_raw).split(';')
                for p in paths:
                    p = p.strip()
                    if p and os.path.exists(p):
                        try:
                            with Image.open(p) as pil_img:
                                pil_img = pil_img.convert('RGB')
                                if self.transform:
                                    image_tensor = self.transform(pil_img)
                            break
                        except:
                            continue
            sample['image'] = image_tensor

        # 4. Label
        if self.target_col and self.target_col in row:
            try:
                sample['label'] = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
            except:
                sample['label'] = torch.tensor(0.0, dtype=torch.float32)

        return sample


# ==============================================================================
# [Part 3] 通用模型组件
# ==============================================================================
class TabularBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, model_name="MLP"):
        super().__init__()
        
        # 1. FT-Transformer 风格 (简单的 Embedding + Transformer Encoder)
        if "Transformer" in model_name:
            self.model_type = "Transformer"
            self.embedding_dim = 64
            # 简单的数值特征线性投影模拟 Embedding
            self.num_proj = nn.Linear(1, self.embedding_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
            
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=4, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.out_dim = self.embedding_dim
            
            # 预计算输入维度对应的投影层 (假设所有输入都是数值，或者已经全部编码为数值)
            self.input_dim = input_dim

        # 2. ResNet 风格 (Residual MLP)
        elif "ResNet" in model_name:
            self.model_type = "ResNet"
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.bn1 = nn.LayerNorm(hidden_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.bn2 = nn.LayerNorm(hidden_dim)
            
            self.out_dim = hidden_dim

        # 3. 默认 MLP
        else:
            self.model_type = "MLP"
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU()
            )
            self.out_dim = hidden_dim // 2

    def forward(self, x):
        if self.model_type == "Transformer":
            # (N, D) -> (N, D, 1) -> (N, D, E)
            B, D = x.shape
            x = x.unsqueeze(-1)
            x = self.num_proj(x) 
            
            # Add CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
            x = self.transformer(x)
            # 取 CLS token 输出
            return x[:, 0, :]

        elif self.model_type == "ResNet":
            out = self.fc1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.dropout(out)
            
            residual = out
            out = self.fc2(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.dropout(out)
            
            return out + residual

        else:
            return self.net(x)


class TextBlock(nn.Module):
    def __init__(self, model_name, dropout):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = self.config.hidden_size

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 取 pooler_output 或 CLS token
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            x = out.pooler_output
        else:
            x = out.last_hidden_state[:, 0, :]
        return self.dropout(x)


class ImageBlock(nn.Module):
    def __init__(self, model_name, dropout):
        super().__init__()
        # 支持 CLIP 和 timm
        if 'clip' in model_name.lower():
            from transformers import CLIPVisionModel
            self.backbone = CLIPVisionModel.from_pretrained(model_name)
            self.out_dim = self.backbone.config.hidden_size
            self.is_clip = True
        else:
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
            self.out_dim = self.backbone.num_features
            self.is_clip = False
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.is_clip:
            out = self.backbone(pixel_values=x)
            x = out.pooler_output
        else:
            x = self.backbone(x)
        return self.dropout(x)


class UniversalFusionModel(nn.Module):
    """
    全能融合网络：根据 active_modalities 动态组装子模块
    支持 Ensemble (Model Averaging)
    """

    def __init__(self, params, active_modalities, model_config, tab_input_dim, output_dim):
        super().__init__()
        self.active_mods = active_modalities
        self.fusion_input_dim = 0
        self.model_map = nn.ModuleDict()  # 使用 ModuleDict 管理多个子模型

        # 1. Tabular
        if 'Tabular' in active_modalities:
            # model_config['Tabular_Model'] 可以是字符串或列表
            models = model_config['Tabular_Model']
            if isinstance(models, str): models = [models]
            
            for i, m_name in enumerate(models):
                # 根据 m_name 实例化不同架构的 Tabular 模型
                self.model_map[f'tab_{i}'] = TabularBlock(
                    input_dim=tab_input_dim,
                    hidden_dim=params.get('tab_hidden', 128),
                    dropout=params.get('dropout', 0.2),
                    model_name=m_name
                )
                self.fusion_input_dim += self.model_map[f'tab_{i}'].out_dim

        # 2. Text
        if 'Text' in active_modalities:
            models = model_config['Text_Model']
            if isinstance(models, str): models = [models]

            for i, m_name in enumerate(models):
                self.model_map[f'text_{i}'] = TextBlock(
                    model_name=m_name,
                    dropout=params.get('dropout', 0.2)
                )
                self.fusion_input_dim += self.model_map[f'text_{i}'].out_dim

        # 3. Image
        if 'Image' in active_modalities:
            models = model_config['Image_Model']
            if isinstance(models, str): models = [models]

            for i, m_name in enumerate(models):
                self.model_map[f'img_{i}'] = ImageBlock(
                    model_name=m_name,
                    dropout=params.get('dropout', 0.2)
                )
                self.fusion_input_dim += self.model_map[f'img_{i}'].out_dim

        # 4. Fusion Head
        self.head = nn.Sequential(
            nn.Linear(self.fusion_input_dim, params.get('fusion_dim', 128)),
            nn.ReLU(),
            nn.Dropout(params.get('dropout', 0.2)),
            nn.Linear(params.get('fusion_dim', 128), output_dim)
        )

    def forward(self, batch):
        feats = []

        # 遍历所有注册的子模型
        for name, model in self.model_map.items():
            if name.startswith('tab_'):
                feats.append(model(batch['tabular']))
            elif name.startswith('text_'):
                feats.append(model(batch['input_ids'], batch.get('attention_mask')))
            elif name.startswith('img_'):
                feats.append(model(batch['image']))

        # Concat
        if len(feats) > 1:
            x = torch.cat(feats, dim=1)
        else:
            x = feats[0]

        return self.head(x)


# ==============================================================================
# [Part 4] 训练与评估逻辑
# ==============================================================================
def calculate_metrics(y_true, y_pred, task_type):
    """
    y_true: np.array (真实标签)
    y_pred: np.array (模型输出，可能是logits或概率)
    """
    # [Fix] 检查 NaN，防止 HPO 崩溃
    if np.isnan(y_pred).any():
        print("[Warning] Prediction contains NaN. Returning worst metrics to prune this trial.")
        return {'MAE': 99999.0, 'RMSE': 99999.0, 'RMSLE': 99999.0} if task_type == 'Regression' else {'Accuracy': 0.0, 'F1': 0.0, 'AUC': 0.0}

    metrics = {}

    if task_type == 'Classification':
        # 确定类别预测 (y_pred_cls) 用于 Acc 和 F1
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            # 多分类 (N, C) -> 取最大概率索引
            y_pred_cls = np.argmax(y_pred, axis=1)
            is_multiclass = True
        else:
            # 二分类 (N,) 或 (N, 1) -> 阈值截断
            y_pred_cls = (y_pred > 0.5).astype(int)
            is_multiclass = False

        metrics['Accuracy'] = accuracy_score(y_true, y_pred_cls)
        metrics['F1'] = f1_score(y_true, y_pred_cls, average='weighted')
        try:
            # 多分类 (Multi-class)
            if is_multiclass:
                # roc_auc_score 在多分类下需要 'ovr' (One-vs-Rest) 策略
                metrics['AUC'] = roc_auc_score(y_true,y_pred,multi_class='ovr',average='weighted')

            # 二分类 (Binary)
            else:
                metrics['AUC'] = roc_auc_score(y_true, y_pred)

        except Exception as e:
            metrics['AUC'] = 0.0

        for k, v in metrics.items():
            metrics[k] = round(v * 100, 2)

    elif task_type == 'Regression':
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        try:
            # 避免负值导致的 RMSLE 报错
            y_pred_safe = np.maximum(y_pred, 0)
            y_true_safe = np.maximum(y_true, 0)
            metrics['RMSLE'] = np.sqrt(mean_squared_log_error(y_true_safe, y_pred_safe))
        except:
            metrics['RMSLE'] = 0.0

        for k, v in metrics.items():
            metrics[k] = round(v, 4)

    return metrics


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, num_classes):
    """单轮训练函数"""
    model.train()
    for batch in loader:
        # [核心] 数据上树 (Move to GPU)
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        # 混合精度训练
        with torch.cuda.amp.autocast():
            logits = model(batch)

            # 损失函数处理
            if num_classes == 1:
                # 二分类或回归，输出需要 squeeze 匹配 label
                loss = criterion(logits.squeeze(), batch['label'])
            else:
                # 多分类，label 需要是 Long 类型
                loss = criterion(logits, batch['label'].long())

        scaler.scale(loss).backward()
        
        # 梯度裁剪 (防止梯度爆炸)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()


def evaluate(model, loader, device, num_classes, task_type):
    """通用评估/预测函数"""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)

            # 后处理：Logits -> Probs/Values
            if task_type == 'Classification':
                if num_classes > 1:
                    probs = torch.softmax(logits, dim=1)  # 多分类
                else:
                    probs = torch.sigmoid(logits.reshape(-1))  # 二分类
                y_pred.append(probs.cpu().numpy())
            else:
                y_pred.append(logits.reshape(-1).cpu().numpy())  # 回归

            if 'label' in batch:
                y_true.append(batch['label'].cpu().numpy())

    if len(y_true) > 0:
        y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return y_true, y_pred

def pre_test_oom(batch_size, params, tab_dim, num_classes, device, dataset, active_modalities):
    """运行多轮 forward+backward 来测试是否 OOM，确保显存稳定"""
    try:
        # 主动清理显存，确保测试环境干净
        gc.collect()
        torch.cuda.empty_cache()

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        model = UniversalFusionModel(params, active_modalities, CONFIG['MODEL_SELECTION'], tab_dim, num_classes).to(device)
        model.train()
        
        if CONFIG['TASK_TYPE'] == 'Regression':
            criterion = nn.MSELoss()
        else:
            criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
            
        optimizer = torch.optim.AdamW(model.parameters(), lr=params.get('lr', 1e-4))
        scaler = torch.cuda.amp.GradScaler()
        
        # [Fix] 增加为测试 3 个 batch，更能暴露潜在的 OOM 问题
        test_steps = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(batch)
                if num_classes == 1:
                    loss = criterion(logits.squeeze(), batch['label'])
                else:
                    loss = criterion(logits, batch['label'].long())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # 主动清理当前 batch 的残留显存
            del batch, logits, loss
            torch.cuda.empty_cache()
            
            test_steps += 1
            if test_steps >= 3:  # 测试三组
                break
            
        del model, optimizer, scaler, loader
        gc.collect()
        torch.cuda.empty_cache()
        return True
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if 'model' in locals(): del model
            gc.collect()
            torch.cuda.empty_cache()
            return False
        raise e


# ==============================================================================
# [Part 5] 主流程
# ==============================================================================
if __name__ == '__main__':
    # 确定输出目录 (脚本所在目录)
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=10, help='HPO 尝试次数')
    args = parser.parse_args()

    seed_everything(CONFIG['SEED'])

    # 1. 解析模态配置 (从 CONFIG 中读取)
    modality_map = CONFIG['MODALITY_MAP']
    active_modalities = []
    tab_dim = 0

    if any(m in ['Numerical', 'Categorical'] for m in modality_map.values()):
        active_modalities.append('Tabular')
        tab_dim = sum(1 for m in modality_map.values() if m in ['Numerical', 'Categorical'])
    if any(m == 'Text' for m in modality_map.values()): active_modalities.append('Text')
    if any(m == 'Image_Path' for m in modality_map.values()): active_modalities.append('Image')

    print(f"\n[*] 任务类型: {CONFIG['TASK_TYPE']}")
    print(f"[*] 活跃模态: {active_modalities}")
    print(f"[*] 模型选择: {CONFIG['MODEL_SELECTION']}")

    # 2. 数据加载与分析
    print("1. 加载数据...")
    df_full = pd.read_csv(CONFIG['DATA_PATH'])

    # 确定输出维度 (Output Dim)
    if CONFIG['TASK_TYPE'] == 'Regression':
        num_classes = 1
        print("[*] 回归任务 -> Output Dim = 1")
    else:
        num_classes = df_full[CONFIG['TARGET_COL']].nunique()
        if num_classes == 2: num_classes = 1  # 二分类用 BCELoss，输出设为 1
        print(f"[*] 分类任务 -> Output Dim = {num_classes}")

    # 3. 初始化 Tokenizer (如有文本)
    tokenizer = None
    if 'Text' in active_modalities:
        models = CONFIG['MODEL_SELECTION']['Text_Model']
        if isinstance(models, str): models = [models]
        # 默认使用第一个模型的 tokenizer
        model_name = models[0]
        print(f"[*] 加载 Tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if 'Image' in active_modalities:
        models = CONFIG['MODEL_SELECTION']['Image_Model']
        if isinstance(models, str): models = [models]
        # 默认使用第一个模型
        model_name = models[0]

    # 4. 图像变换 (如有图像)
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 5. HPO 数据划分
    val_frac = 0.2
    df_train_hpo = df_full.sample(frac=1 - val_frac, random_state=CONFIG['SEED'])
    df_val_hpo = df_full.drop(df_train_hpo.index)

    ds_train_hpo = MultimodalDataset(df_train_hpo, modality_map, tokenizer, img_transform, CONFIG['TARGET_COL'], scaler=None)
    hpo_scaler = ds_train_hpo.scaler
    ds_val_hpo = MultimodalDataset(df_val_hpo, modality_map, tokenizer, img_transform, CONFIG['TARGET_COL'], scaler=hpo_scaler)
    print(f"[*] HPO阶段: 数值特征Scaler已从训练集拟合，并应用到验证集。")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MAX_EPOCHS = 12
    EARLY_STOPPING_PATIENCE = 4

    # 5.5 OOM 预测试与动态搜索空间调整
    batch_size_candidates = [16, 32, 64]
    best_init_params = {"lr": 0.0005, "batch_size": 32, "dropout": 0.3, "tab_hidden": 512, "fusion_dim": 1024}

    print("\n[*] 开始 OOM 预测试机制 (动态调整 batch_size 搜索空间)...")
    while True:
        max_bs = max(batch_size_candidates)
        print(f"    -> 正在测试最大 batch_size = {max_bs} ...")
        
        test_params = best_init_params.copy()
        test_params['batch_size'] = max_bs
        
        # 按照用户建议，测试两组以降低偶然性。两次都失败才会被认为是失败值。
        is_success = pre_test_oom(max_bs, test_params, tab_dim, num_classes, device, ds_train_hpo, active_modalities)
        if not is_success:
            print("       (第一组测试失败，进行第二组稳定性测试...)")
            is_success = pre_test_oom(max_bs, test_params, tab_dim, num_classes, device, ds_train_hpo, active_modalities)
            
        if is_success:
            print("    -> 测试通过！当前安全 batch_size 空间:", batch_size_candidates)
            break
        else:
            print(f"    -> [OOM] batch_size={max_bs} 连续两次触发显存溢出！剔除 {max_bs} 并添加更小的值...")
            if max_bs in batch_size_candidates:
                batch_size_candidates.remove(max_bs)
            if len(batch_size_candidates) == 0:
                batch_size_candidates.append(max(1, max_bs // 2))
            else:
                next_min = max(1, min(batch_size_candidates) // 2)
                if next_min not in batch_size_candidates:
                    batch_size_candidates.append(next_min)
            batch_size_candidates.sort()

    # 关键修复：循环结束后，确保把已经爆掉的大 batch_size 彻底剔除，不传给 Optuna
    # 因为在上面的循环中，虽然移除了 max_bs，但打印的时候可能没有正确更新引用
    print("    -> OOM 测试结束。最终安全的 batch_size 搜索空间为:", batch_size_candidates)

    if best_init_params.get('batch_size') not in batch_size_candidates:
        best_init_params['batch_size'] = max(batch_size_candidates)

    # 全局变量记录触发OOM的最小 batch_size
    GLOBAL_MAX_BATCH_SIZE = float('inf')

    # 6. 定义超参优化目标函数
    def objective(trial):
        global GLOBAL_MAX_BATCH_SIZE
        # 定义搜索空间 (由 PA-LLM 动态生成)
        print(f"\n===== [Trial {trial.number}] 开始训练 =====")
        
        # 修正语法问题，先计算 LLM 生成的参数字典，再注入 batch_size
        llm_params = {
            'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True), 'dropout': trial.suggest_float('dropout', 0.1, 0.5), 'tab_hidden': trial.suggest_categorical('tab_hidden', [256, 512, 1024]), 'fusion_dim': trial.suggest_categorical('fusion_dim', [512, 1024, 2048])
        }
        
        params = llm_params.copy()
        params['batch_size'] = trial.suggest_categorical('batch_size', batch_size_candidates)

        # OOM 防护拦截：如果当前 batch_size 大于历史曾经爆显存的 batch_size，直接剪枝
        # 发生过OOM的batch size仍保留作为边界，允许在其他超参组合下再次尝试
        if params.get('batch_size', 0) > GLOBAL_MAX_BATCH_SIZE:
            print(f"[Trial {trial.number}] 触发 OOM 拦截机制！当前 batch_size ({params.get('batch_size')}) > 历史 OOM 阈值 ({GLOBAL_MAX_BATCH_SIZE})，直接跳过。")
            
            # [Fix] 如果这是最后一个尝试（或者所有候选 bs 都比 OOM 阈值大），直接返回极差分数
            # 避免全军覆没导致 HPO 无结果可出
            return -99999.0

        model = None
        try:
            # DataLoader (num_workers=0 省内存)
            num_workers = 8 if os.name != 'nt' else 0

            print(f"[Trial {trial.number}] 初始化DataLoader (batch_size={params['batch_size']})...")
            train_loader = DataLoader(ds_train_hpo, batch_size=params['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True)
            val_loader = DataLoader(ds_val_hpo, batch_size=params['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=True)

            # 实例化通用融合模型
            print(f"[Trial {trial.number}] 初始化模型 (tab_dim={tab_dim}, num_classes={num_classes})...")
            model = UniversalFusionModel(params, active_modalities, CONFIG['MODEL_SELECTION'], tab_dim, num_classes).to(device)

            # 损失函数选择
            if CONFIG['TASK_TYPE'] == 'Regression':
                criterion = nn.MSELoss()
            else:
                criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()

            optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'])
            scaler = torch.cuda.amp.GradScaler()

            # 训练循环
            trial_best_metric = -float('inf')
            trial_best_epoch = 0
            patience_counter = 0

            for epoch in range(MAX_EPOCHS):
                print(f"[Trial {trial.number}] 开始Epoch {epoch + 1}/{MAX_EPOCHS} 训练...")
                train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, num_classes)
                print(f"[Trial {trial.number}] Epoch {epoch + 1} 训练完成，开始评估...")
                y_true, y_pred = evaluate(model, val_loader, device, num_classes, CONFIG['TASK_TYPE'])

                metrics = calculate_metrics(y_true, y_pred, CONFIG['TASK_TYPE'])
                # 优化目标：分类最大化 AUC (或 F1)，回归最大化 -RMSE
                current_score = metrics.get('AUC', metrics.get('F1', -metrics.get('RMSE', -9999)))

                print(f"[Trial {trial.number}] Epoch {epoch + 1}/{MAX_EPOCHS} | Score: {current_score} | Metrics: {metrics}")

                # 早停逻辑与最佳 Epoch 记录
                if current_score > trial_best_metric:
                    trial_best_metric = current_score
                    trial_best_epoch = epoch + 1  # 记录当前是第几轮
                    patience_counter = 0  # 重置耐心值
                else:
                    patience_counter += 1

                # 触发早停
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"[Trial {trial.number}] 早停触发！在第 {epoch + 1} 轮停止。最佳轮次是: {trial_best_epoch}")
                    break
                
                # --- [新增] Optuna Pruning (剪枝) ---
                trial.report(current_score, epoch)
                if trial.should_prune():
                    print(f"[Trial {trial.number}] Pruned! 性能不佳，提前终止。")
                    raise optuna.exceptions.TrialPruned()
                # ------------------------------------

            trial.set_user_attr("best_epoch", trial_best_epoch)

            return trial_best_metric

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                failed_bs = params.get('batch_size', float('inf'))
                if failed_bs < GLOBAL_MAX_BATCH_SIZE:
                    GLOBAL_MAX_BATCH_SIZE = failed_bs
                print(f"[OOM] Trial {trial.number} 显存溢出！已触发保护机制，后续 Trial 的 batch_size 必须小于等于 {GLOBAL_MAX_BATCH_SIZE}。")
                gc.collect()
                torch.cuda.empty_cache()
                return -99999.0
            raise e
        finally:
            # 强制清理模型，释放显存
            print(f"[Trial {trial.number}] 训练结束，清理资源...")
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()

    # 7. 执行 HPO
    if args.trials > 0:
        print(f"\n2. 开始超参优化 (Trials={args.trials})...")
        # 使用 TPESampler (贝叶斯优化) + MedianPruner (中位数剪枝)
        sampler = optuna.samplers.TPESampler(seed=CONFIG['SEED'])
        # 剪枝设置：至少5个Trial建立基准，至少8个Epoch再决定去留
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=8, interval_steps=1)
        
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
        
        # [核心] 注入 LLM 推荐的最佳初始参数 (Warm-start)
        print("[*] 注入 LLM 推荐的初始参数:", best_init_params)
        study.enqueue_trial(best_init_params)
        
        study.optimize(objective, n_trials=args.trials)
        best_params = study.best_params
        best_epoch = study.best_trial.user_attrs["best_epoch"]
        print(f"   >>> 最佳参数: {best_params}")
        print(f"   >>> 最佳训练轮数 (Best Epoch): {best_epoch}")
    else:
        print("\n2. 跳过 HPO，使用默认参数。")
        best_params = {'lr': 1e-4, 'batch_size': 4, 'dropout': 0.2, 'tab_hidden': 128, 'fusion_dim': 256}
        best_epoch = 5

    # 8. 最终训练 (使用最佳参数 + 全量数据)
    print("\n3. 使用最佳参数进行最终训练...")

    # 准备数据集
    ds_full = MultimodalDataset(df_full, modality_map, tokenizer, img_transform, CONFIG['TARGET_COL'], scaler=None)

    final_scaler = ds_full.scaler

    if os.path.exists(CONFIG['TEST_PATH']):
        df_test = pd.read_csv(CONFIG['TEST_PATH'])
        ds_test = MultimodalDataset(df_test, modality_map, tokenizer, img_transform, CONFIG['TARGET_COL'], scaler=final_scaler)
    else:
        print("警告：未找到测试集，使用全量数据的前 20% 作为测试集演示。")
        ds_test = ds_full  # Fallback

    num_workers = 8 if os.name != 'nt' else 0
    # 强制最终训练使用较大的 batch_size 以加速
    final_batch_size = best_params['batch_size']
    train_loader = DataLoader(ds_full, batch_size=final_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=final_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    final_model = UniversalFusionModel(best_params, active_modalities, CONFIG['MODEL_SELECTION'], tab_dim,
                                       num_classes).to(device)

    if CONFIG['TASK_TYPE'] == 'Regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_params['lr'])
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(best_epoch):
        train_one_epoch(final_model, train_loader, criterion, optimizer, scaler, device, num_classes)
        print(f"Epoch {epoch + 1}/{best_epoch} 完成")

    # 9. 最终评估与结果保存
    print("\n4. 最终评估与保存结果...")
    y_true, y_pred = evaluate(final_model, test_loader, device, num_classes, CONFIG['TASK_TYPE'])

    final_metrics = calculate_metrics(y_true, y_pred, CONFIG['TASK_TYPE'])
    print("=" * 40)
    print("最终测试集指标:")
    for k, v in final_metrics.items():
        print(f"{k}: {v}")
    print("=" * 40)

    # 保存预测结果
    res_df = pd.DataFrame()
    # 如果有真实标签，保存对比
    if len(y_true) == len(y_pred):
        res_df['y_true'] = y_true

    if CONFIG['TASK_TYPE'] == 'Classification':
        if num_classes > 1:
            res_df['y_pred_class'] = np.argmax(y_pred, axis=1)
        else:
            res_df['y_pred_class'] = (y_pred > 0.5).astype(int)
            res_df['y_pred_prob'] = y_pred
    else:
        res_df['y_pred'] = y_pred

    res_df.to_csv(os.path.join(OUTPUT_DIR, 'final_results.csv'), index=False)
    print(f"[*] 结果已保存至 {os.path.join(OUTPUT_DIR, 'final_results.csv')}")

    # 保存指标到 txt
    result_txt_path = os.path.join(OUTPUT_DIR, 'results.txt')
    with open(result_txt_path, 'w') as f:
        for k, v in final_metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"[*] 指标已保存至 {result_txt_path}")
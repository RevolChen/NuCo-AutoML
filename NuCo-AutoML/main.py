import os
import sys
import subprocess
import json
import argparse
import config.settings as config
from utils.data_utils import load_data, separate_target, get_dataset_meta_features
from modules.mi_llm import MILLM
from modules.afe_llm import AFELLM
from modules.ms_llm import MSLLM
from modules.pa_llm import PALLM
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# # 强制 UTF-8 输出
# sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def run_modality_inference_module(args):
    """
    模态识别大模型模块的主执行逻辑
    """
    # ================= 动态配置更新 =================
    if args.model:
        config.MODEL_NAME = args.model

    data_path = args.train_data_path
    target_col = args.target
    mi_output_path = args.mi_output

    print("[*][*][*] [MI-LLM] 任务开始。[*][*][*]")

    # ================= 数据加载与预处理 =================
    print(f"[*] [MI-LLM] 正在读取数据: {data_path}")
    if not os.path.exists(data_path):
        print(f"错误：找不到数据文件 {data_path}")
        return

    df = load_data(data_path)

    # 分离目标列
    print(f"[*] MI-LLM 分离目标列: {target_col}")
    try:
        X, y = separate_target(df, target_col)
    except ValueError as e:
        print(f"错误: {e}")
        return

    print(f"[*] MI-LLM 剩余特征列: {len(X.columns)} 个")

    # ================= 调用 MI-LLM 模块 =================
    # 初始化模块，传入命令行指定的模型名称
    mi_agent = MILLM(model_name=args.model)

    # 执行推理，传入命令行指定的迭代次数(样本数)
    inferred_modalities = mi_agent.infer(X, n_samples=args.iterations)

    if not inferred_modalities:
        print("MI-LLM 推理失败，程序终止。")
        return

    # ================= 结果整合 =================
    # 将 Target 列手动加回结果
    final_modalities = inferred_modalities.copy()
    final_modalities[target_col] = "Target"  # 明确标记目标列

    # ================= 输出与保存 =================
    print("\n" + "=" * 30)
    print(f"MI-LLM 模块结果 (Model: {args.model})")

    import pprint
    pprint.pprint(final_modalities)

    # 确保保存目录存在
    os.makedirs(os.path.dirname(mi_output_path), exist_ok=True)

    with open(mi_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_modalities, f, indent=4, ensure_ascii=False)

    print(f"\n[*] MI-LLM 配置已保存至: {mi_output_path}")
    print("[*][*][*] [MI-LLM] 任务完成。[*][*][*]\n")


def run_auto_feature_engineering_module(args):
    """
    自动特征工程模块的主执行逻辑
    依赖: MI-LLM 生成的 json 配置文件
    """
    # 动态配置更新
    if args.model:
        config.MODEL_NAME = args.model

    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    target_col = args.target
    # 输入：需要读取 MI 模块生成的 JSON
    mi_output_path = args.mi_output
    # 输出：AFE 策略以及清洗后的 CSV 保存路径
    afe_output_path = args.afe_output
    afe_train_output_path = args.afe_train_output
    afe_test_output_path = args.afe_test_output

    print("[*][*][*] [AFE-LLM] 任务开始。[*][*][*]")

    if not os.path.exists(mi_output_path):
        print(f"错误: 找不到 MI-LLM 模块的输出文件: {mi_output_path}")
        print("请先运行 MI-LLM 模块生成模态配置文件。")
        return

    print(f"[*] [AFE-LLM] 正在加载数据与模态配置...")

    df_train = load_data(train_data_path)
    df_test = load_data(test_data_path)

    # 加载 MI 生成的模态映射
    with open(mi_output_path, 'r', encoding='utf-8') as f:
        modality_map = json.load(f)

    # 调用 AFE-LLM 模块
    afe_agent = AFELLM(model_name=args.model)

    # 生成清洗计划
    cleaning_plan = afe_agent.generate_plan(df_train, modality_map, target_col)

    if not cleaning_plan:
        print("AFE-LLM 未能生成有效计划，跳过执行。")
        return

    print("\n[AFE-LLM 清洗策略]:")
    print(json.dumps(cleaning_plan, indent=2, ensure_ascii=False))

    # 保存 AFE 策略
    os.makedirs(os.path.dirname(afe_output_path), exist_ok=True)
    with open(afe_output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaning_plan, f, indent=2, ensure_ascii=False)
    print(f"[*] AFE-LLM 清洗规则已保存至: {afe_output_path}")

    # 执行清洗计划
    df_train_cleaned = afe_agent.execute_plan(df_train, cleaning_plan, target_col)
    df_test_cleaned = afe_agent.execute_plan(df_test, cleaning_plan, target_col)

    # 保存结果
    os.makedirs(os.path.dirname(afe_train_output_path), exist_ok=True)
    df_train_cleaned.to_csv(afe_train_output_path, index=False, encoding='utf-8')
    print(f"\n[*] AFE-LLM 处理完成 (训练集):")
    print(f"    原始形状: {df_train.shape} -> 清洗后: {df_train_cleaned.shape}")
    print(f"    保存路径: {afe_train_output_path}")

    df_test_cleaned.to_csv(afe_test_output_path, index=False, encoding='utf-8')
    print(f"[*] AFE-LLM 处理完成 (测试集):")
    print(f"    原始形状: {df_test.shape} -> 清洗后: {df_test_cleaned.shape}")
    print(f"    保存路径: {afe_test_output_path}")

    print("[*][*][*] [AFE-LLM] 任务完成。[*][*][*]\n")


def run_safe_module(args):
    """
    SAFE (formerly CAAFE) 模块的主执行逻辑
    依赖: AFE-LLM 清洗后的数据，以及 MI-LLM 生成的模态映射
    """
    print("[*][*][*] [SAFE] 任务开始。[*][*][*]")
    
    import sys
    import pandas as pd
    safe_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../SAFE'))
    if safe_path not in sys.path:
        sys.path.append(safe_path)
    
    try:
        from test_ji import generate_feat_effect
    except ImportError as e:
        print(f"导入 SAFE 模块失败: {e}")
        return

    train_data_path = args.afe_train_output
    test_data_path = args.afe_test_output
    mi_output_path = args.mi_output
    target_col = args.target
    
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        print(f"错误: 找不到 AFE-LLM 处理后的数据。请先运行 AFE-LLM 模块。")
        return
        
    df_train = load_data(train_data_path)
    df_test = load_data(test_data_path)
    
    # Load modalities
    with open(mi_output_path, 'r', encoding='utf-8') as f:
        modality_map = json.load(f)
        
    # Get numerical columns
    num_cols = [col for col, mod in modality_map.items() if mod == "Numerical" and col in df_train.columns]
    
    if not num_cols:
        print("[*] [SAFE] 没有找到数值型列，跳过特征生成。")
        return
        
    print(f"[*] [SAFE] 识别到 {len(num_cols)} 个数值型列，开始生成特征...")
    
    # Extract numerical + target data
    df_train_num = df_train[num_cols + [target_col]].copy()
    df_test_num = df_test[num_cols + [target_col]].copy()
    
    # Determine task type based on target column
    is_classification = df_train[target_col].nunique() < 20 or not pd.api.types.is_numeric_dtype(df_train[target_col])
    task_type = 'classification' if is_classification else 'regression'
    
    print(f"[*] [SAFE] 推断任务类型为: {task_type}")
    
    df_train_num_feat, df_test_num_feat = generate_feat_effect(
        df_train=df_train_num,
        df_test=df_test_num,
        llm_model=args.model,
        iterations=args.safe_iterations,
        target_column_name=target_col,
        dataset_description="Dataset",
        task=task_type
    )
    
    # Get new features
    new_cols = [col for col in df_train_num_feat.columns if col not in df_train_num.columns and col != target_col]
    
    if not new_cols:
        print("[*] [SAFE] 未生成新特征。")
        print("[*][*][*] [SAFE] 任务完成。[*][*][*]\n")
        return
        
    print(f"[*] [SAFE] 成功生成 {len(new_cols)} 个新特征: {new_cols}")
    
    # Insert new features into original dataframe
    # "将生成的特征放在数据的Image_Path列的前面（如果有的话，如果没有就放在最后一列的前面）"
    image_path_cols = [col for col, mod in modality_map.items() if mod == "Image_Path" and col in df_train.columns]
    
    if image_path_cols:
        insert_idx = df_train.columns.get_loc(image_path_cols[0])
    else:
        insert_idx = len(df_train.columns) - 1 # before the last column
        
    for i, col in enumerate(new_cols):
        df_train.insert(insert_idx + i, col, df_train_num_feat[col])
        df_test.insert(insert_idx + i, col, df_test_num_feat[col])
        
    # Update modality map
    for col in new_cols:
        modality_map[col] = "Numerical"
        
    # Save updated data
    df_train.to_csv(train_data_path, index=False, encoding='utf-8')
    df_test.to_csv(test_data_path, index=False, encoding='utf-8')
    
    # Save updated modality map
    with open(mi_output_path, 'w', encoding='utf-8') as f:
        json.dump(modality_map, f, indent=4, ensure_ascii=False)
        
    print(f"[*] [SAFE] 任务完成，数据与模态配置已更新。")
    print("[*][*][*] [SAFE] 任务完成。[*][*][*]\n")


def run_model_selection_module(args):
    """
    模型选择模块的主执行逻辑
    依赖: MI-LLM 生成的 json
    输出: model_config.json
    """
    if args.model:
        config.MODEL_NAME = args.model

    mi_output_path = args.mi_output
    afe_plan_path = args.afe_output
    ms_output_path = args.ms_output
    afe_train_output_path = args.afe_train_output
    target_col = args.target
    preference = args.preference  # 用户偏好

    print("[*][*][*] [MS-LLM] 任务开始。[*][*][*]")

    # 加载 MI 原始模态
    if not os.path.exists(mi_output_path):
        print(f"错误: 找不到 MI-LLM 输出文件: {mi_output_path}")
        return

    with open(mi_output_path, 'r', encoding='utf-8') as f:
        full_modality_map = json.load(f)

    # 加载 AFE 清洗计划并过滤被删除的列
    filtered_modality_map = full_modality_map.copy()

    if os.path.exists(afe_plan_path):
        print(f"[*] 检测到 AFE-LLM 清洗计划: {afe_plan_path}")
        try:
            with open(afe_plan_path, 'r', encoding='utf-8') as f:
                afe_plan = json.load(f)

            # 获取被删除的列名列表
            drop_columns = afe_plan.get("drop_columns", [])

            if drop_columns:
                print(f"[*] 正在过滤被 AFE-LLM 删除的列 ({len(drop_columns)} 个)...")
                for col in drop_columns:
                    if col in filtered_modality_map:
                        del filtered_modality_map[col]
            else:
                print("[*] AFE-LLM 未删除任何列，使用全量模态。")

        except Exception as e:
            print(f"警告: 读取 AFE-LLM 计划失败 ({e})，将使用原始模态进行模型选择。")
    else:
        print("[!] 警告: 未找到 AFE-LLM 清洗计划文件。建议先运行 AFE-LLM 模块，否则 MS-LLM 可能会为已删除的列分配模型。")

    print(f"[*] [MS-LLM] 正在计算数据集元特征以辅助决策...")

    df_full = load_data(afe_train_output_path)
    meta_features = get_dataset_meta_features(df_full, target_col, filtered_modality_map)
    print(f"    数据集概览: {json.dumps(meta_features, ensure_ascii=False)}")

    # 调用 MS-LLM
    ms_agent = MSLLM(model_name=args.model)

    # 执行选择
    selected_models = ms_agent.select_models(modality_input=filtered_modality_map, meta_features=meta_features, preference=preference)

    if not selected_models:
        print("MS-LLM 未能生成模型配置。")
        return

    # 输出与保存
    print("\n[MS-LLM 模型选择结果]:")
    import pprint
    pprint.pprint(selected_models)

    os.makedirs(os.path.dirname(ms_output_path), exist_ok=True)
    with open(ms_output_path, 'w', encoding='utf-8') as f:
        json.dump(selected_models, f, indent=4, ensure_ascii=False)

    print(f"\n[*] 模型配置已保存至: {ms_output_path}")
    print("[*][*][*] [MS-LLM] 任务完成。[*][*][*]\n")


def run_pipeline_assembly_module(args):
    """
    流水线组装模块的主执行逻辑 (Template Filling 模式)
    依赖:
    1. AFE 处理后的数据 (Train/Test)
    2. MS 生成的模型配置 (model_config.json)
    3. MI 生成并经 AFE 过滤后的模态映射
    4. 预定义的 pipeline_skeleton.py 模板
    输出: pipeline.py (保存到 args.pa_output)
    """
    if args.model:
        config.MODEL_NAME = args.model

    # 输入路径
    afe_train_path = args.afe_train_output
    afe_test_path = args.afe_test_output
    ms_output_path = args.ms_output
    mi_output_path = args.mi_output
    afe_plan_path = args.afe_output

    # 输出路径
    code_output_path = args.pa_output

    print("[*][*][*] [PA-LLM] 任务开始。[*][*][*]")

    # 检查依赖
    if not os.path.exists(afe_train_path):
        print(f"错误: 找不到 AFE-LLM 训练数据: {afe_train_path}")
        return False
    if not os.path.exists(ms_output_path):
        print(f"错误: 找不到 MS-LLM 模型配置: {ms_output_path}")
        return False

    # 准备上下文数据，加载模型配置
    with open(ms_output_path, 'r', encoding='utf-8') as f:
        model_config = json.load(f)

    # 加载模态映射 (并应用 AFE 的删除规则)
    with open(mi_output_path, 'r', encoding='utf-8') as f:
        full_modality_map = json.load(f)

    filtered_modality_map = full_modality_map.copy()
    if os.path.exists(afe_plan_path):
        with open(afe_plan_path, 'r', encoding='utf-8') as f:
            afe_plan = json.load(f)
        for col in afe_plan.get("drop_columns", []):
            if col in filtered_modality_map:
                del filtered_modality_map[col]

    # 调用 PA-LLM
    pa_agent = PALLM(model_name=args.model)

    # 计算元特征
    print(f"[*] [PA-LLM] 正在计算元特征...")
    df_train_afe = load_data(afe_train_path)
    meta_features = get_dataset_meta_features(df_train_afe, args.target, filtered_modality_map)

    # 生成代码
    final_script_path = pa_agent.generate_code(
        data_path=afe_train_path,
        test_path=afe_test_path,
        modality_map=filtered_modality_map,
        model_config=model_config,
        target_col=args.target,
        meta_features=meta_features,
        output_path=code_output_path,
        seed=args.seed
    )

    print(f"\n[*] [PA-LLM] 训练预测脚本已生成: {final_script_path}")
    print("[*][*][*] [PA-LLM] 任务完成。[*][*][*]\n")

if __name__ == "__main__":
    # 定义命令行参数解析器
    parser = argparse.ArgumentParser()

    # 数据集名称
    parser.add_argument('--dataset', default='MMSD', type=str, help='数据集名称 (用于输出目录)')

    # 核心参数
    parser.add_argument('--model', default='gpt-4o',type=str,help='指定使用的大模型版本 (例如: gpt-4o, gpt-3.5-turbo)')
    parser.add_argument('--iterations', default=10,type=int,help='MI-LLM模块上下文学习的样本数量')
    parser.add_argument('--safe_iterations', default=5,type=int,help='SAFE特征生成迭代次数')
    parser.add_argument('--target', default='AdoptionSpeed',type=str,help='数据集的目标列名称 (Label Column)')
    parser.add_argument('--preference', default='Best Accuracy, GPU Available', type=str, help='模型选择偏好 (e.g., "Fastest Inference" or "High Accuracy")')
    parser.add_argument('--seed', default=42, type=int, help='全局随机种子')

    # 解析参数 (第一阶段，为了获取 dataset 参数)
    args, unknown = parser.parse_known_args()
    
    # 根据数据集名称动态设置默认路径
    DEFAULT_DATASET_NAME = args.dataset

    # 数据路径参数
    parser.add_argument('--train_data_path', default=os.path.join('data', DEFAULT_DATASET_NAME, 'train_split.csv'),type=str,help='训练集路径')
    parser.add_argument('--test_data_path', default=os.path.join('data', DEFAULT_DATASET_NAME, 'test_split.csv'), type=str, help='测试集路径')
    parser.add_argument('--mi_output', default=os.path.join('output', f'{DEFAULT_DATASET_NAME}', 'dataset_state.json'), type=str,help='MI模块输出的JSON路径')
    parser.add_argument('--afe_output',default=os.path.join('output', f'{DEFAULT_DATASET_NAME}', 'afe_plan.json'), type=str,help='AFE模块输出的JSON路径')
    parser.add_argument('--afe_train_output', default=os.path.join('output', f'{DEFAULT_DATASET_NAME}', 'train_afe.csv'), type=str,help='AFE模块输出路径 (Train)')
    parser.add_argument('--afe_test_output', default=os.path.join('output', f'{DEFAULT_DATASET_NAME}', 'test_afe.csv'), type=str, help='AFE模块输出路径 (Test)')
    parser.add_argument('--ms_output', default=os.path.join('output', f'{DEFAULT_DATASET_NAME}', 'model_config.json'), type=str, help='MS模块输出的模型配置路径')
    parser.add_argument('--pa_output', default=os.path.join('output', f'{DEFAULT_DATASET_NAME}', 'pipeline.py'), type=str, help='PA模块生成的Python脚本路径')

    # 再次解析参数 (覆盖默认值)
    args = parser.parse_args()

    # 调用主功能函数
    run_modality_inference_module(args)
    run_auto_feature_engineering_module(args)
    run_safe_module(args)
    run_model_selection_module(args)
    run_pipeline_assembly_module(args)

    pipeline_script_path = args.pa_output

    if os.path.exists(pipeline_script_path):
        print("\n" + "=" * 50)
        print(f"[*] 自动执行生成的 Pipeline: {pipeline_script_path}")
        print(f"[*] 全局随机种子 (Seed): {args.seed}")
        print("=" * 50 + "\n")

        # 显式设置 CUDA_VISIBLE_DEVICES 环境变量 (假设服务器有较好的 GPU 编号为 0)
        # 如果需要更智能的检测，可以引入 torch.cuda.is_available() 等
        env = os.environ.copy()
        # 默认尝试使用第一个 GPU，用户也可以在外部设置
        if "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = "0"

        try:
            # 构造执行命令
            cmd = [
                sys.executable,
                '-u',
                pipeline_script_path
            ]

            # 执行命令 (传入 env)，并将输出重定向到文件和屏幕
            # [修改] 将日志文件命名为 run.log，避免与 pipeline.py 生成的 results.txt 冲突
            log_file_path = os.path.join(os.path.dirname(pipeline_script_path), 'run.log')

            # [关键修改] 使用 'a' (append) 模式打开文件，或者 'w' (write) 模式。
            # 这里我们使用 subprocess.run 并将 stdout 直接重定向到文件对象
            # 这样由操作系统负责写入，避免 Python 层的编解码问题
            
            print(f"[*] 开始执行 Pipeline，日志将保存至: {log_file_path}")
            
            # 使用 io.open 确保在 Python 2/3 兼容性，并且强制 line buffering
            with open(log_file_path, 'w', encoding='utf-8', buffering=1) as f_log:
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,  # 让 Python 处理解码
                    encoding='utf-8', # 显式指定 UTF-8
                    errors='replace', # 忽略错误
                    bufsize=1  # 行缓冲
                )
                
                # 实时读取
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    
                    if line:
                        # 1. 打印到屏幕
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        
                        # 2. 写入文件
                        f_log.write(line)
                        f_log.flush()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

            print("\n" + "=" * 50)
            print("[*] 恭喜！全流程执行成功。")
            print("=" * 50)

        except subprocess.CalledProcessError as e:
            print(f"\n[!] Pipeline 执行过程中发生错误 (Exit Code: {e.returncode})")
        except KeyboardInterrupt:
            print("\n[!] 用户手动中断了 Pipeline 执行。")
        except Exception as e:
            print(f"\n[!] 无法启动 Pipeline 脚本: {e}")
    else:
        print(f"\n[!] 警告: 未找到生成的代码文件 ({pipeline_script_path})，跳过执行步骤。")
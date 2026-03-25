## 具体步骤

1. 环境打包为gzh.tar.gz

2. 初始配置：

   （1）openai_key处理：

   ​        在autom3l/config/settings目录下的“OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_key")”填写你自己的openai_key

   （2）数据集预处理：

   1.在autom3l/data路径下下载你的数据集，并且将多个模态的数据处理到统一的一个表格数据中，对于图像模态需要新增列名Image_Path，存放的是数据记录对应图像的本地绝对路径，并将目标列置于最后一列

   2.最终处理后的表格命名为data_final.csv，例如：“autom3l/data/PAP/data_final.csv”

   3.将data_final.csv按照80%：20%的比例划分为训练和测试集，命名为train_split.csv和test_split.csv，例如：“autom3l/data/PAP/train_split.csv”和“autom3l/data/PAP/test_split.csv”

   （3）main.py中可调节参数：

              1. DEFAULT_DATASET_NAME修改为数据集名称，例如：‘PAP’
              2. --model修改为具体调用大模型类型，例如：‘gpt-4o’
           3. --target修改为数据集目标列，例如：‘AdoptionSpeed’
           4. --seed修改为本次实验随机种子数，例如：‘42’
           5. 其他参数可保持原状

3. 运行程序

   运行程序可有两种方法：

   （1）第一种是直接一口气运行，跑完整个pipeline，运行结束后可直接得到本轮结果

```sh
cd autom3l
python main.py
```

​       （2）第二种是分两步，先运行main.py的模块，这些是大模型决策加上流水线代码组装，不包含模型训练过程，在main.py中把main函数中的“run_pipeline_assembly_module(args)”下方代码都注释掉，然后运行

```
python main.py
```

​                上边模块会生成autom3l/output/数据集名/pipeline.py，例如：“autom3l/output/PAP/pipeline.py”，这部分就是多模态模型训练及预测过程，输入命令可得到本轮结果

```
cd autom3l/output/数据集名
python pipeline.py
```

4. 模块介绍

```
-autom3l

 -config(资源配置)
  -init.py
  -model_zoo.py(预定义模型库，包括表格、文本、图像、多模态模型集的描述)
  -setting.py(配置参数)
  
 -data(数据集存放)
 
 -dataprocess(数据集预处理脚本存放)
 
 -modules
  -init.py
  -afe_llm.py(特征工程模块，大模型对数据集进行噪声列剔除，缺失值填补，异常值纠正,最终输出为autom3l/output/afe_plan.json、autom3l/output/train_afe.csv和autom3l/output/test_afe.csv)
  -mi_llm.py(模态识别模块，大模型识别表格数据中各列的模态，最终输出为autom3l/output/dataset_state.json)
  -ms_llm.py(模型选择模块，大模型根据为数据集中各模态在模型库中挑选模型，最终输出为autom3l/output/model_config.json)
  -pa_llm.py(流水线组装模块，将前边流程大模型的决策填充进多模态融合代码，最终输出为autom3l/output/pipeline.py)
  
 -templates(预定义的多模态融合代码存放)
  -init.py
  -pipeline_skeleton.py(多模态融合代码模板)
  
 -utils(工具代码)
  -init.py
  -data_utils.py(存放一些模块可能会用到的函数)
 
 main.py(主函数入口，运行整个程序)
```
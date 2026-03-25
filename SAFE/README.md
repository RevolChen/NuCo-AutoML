## 具体步骤

1. 环境打包为caafe.tar.gz

2. 具体配置：

   （1）openai_key处理：

   ​        在SAFE/caafe/caafe.py目录下的大概第157行，填写你自己的api_key

   ```
   client = OpenAI(
       api_key="xxx",
       base_url="https://api.openai-proxy.org/v1"  # 如是代理或本地服务，自行修改
   )
   ```

​      （2）数据都存放在SAFE/tests/data_ji里

​      （3）主函数入口为SAFE/test_ji.py，需要修改参数--default_seed（随机种子）；--model（调用的大模型）；--iterations（特征生成迭代轮次）；--task（数据集的具体任务类型）；

​        通过下方代码 for ds_name in ['','']: 在''里填写具体数据集名称，例如for ds_name in ['boston','seeds']:。

​      （4）最终输出 "tests/data_ji/seed" + seed + "/" + task + "/outputs_CAAFE"+dataset+"/_original_CAAFE_train.csv"和

"tests/data_ji/seed" + seed + "/" + task + "/outputs_CAAFE"+dataset+"/_original_CAAFE_test.csv"，分别为CAAFE特征工程后的训练集和测试集，例如 tests/data_ji/seed42/regression/outputs_CAAFE/boston_original_CAAFE_train.csv和tests/data_ji/seed42/regression/outputs_CAAFE/boston_original_CAAFE_test.csv

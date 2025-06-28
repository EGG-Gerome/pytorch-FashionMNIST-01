### 🚀 运行方式

以下指令均在终端中依次执行即可：

1. 创建名为 `pytorch01` 的虚拟环境，指定 Python 版本为 3.10

   ```bash
   conda create -n pytorch01 python=3.10
   ```

2. 激活环境

   ```bash
   conda activate pytorch01
   ```

3. 安装依赖项

   ```bash
   pip install -r requirements.txt
   ```

4. 运行主程序

   ```bash
   python main.py
   ```

---

### 📘 使用说明

1. **首次运行时**将自动创建 `data/` 文件夹并下载 `FashionMNIST` 数据集。
2. 训练与测试日志将输出至终端，并同时写入 `log/` 文件夹中的日志文件（如 `log0.txt`、`log1.txt` 等，序号自动递增）。
3. 准确率（accuracy）和损失值（loss）最优的两个模型将分别保存在 `checkpoints/` 文件夹中，文件名包含相关指标与训练轮次。
4. 超参数配置文件位于 `config.py`，可根据需求自行修改（如 `learning_rate`、`epochs`、`device` 等）。

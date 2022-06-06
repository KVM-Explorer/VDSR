## Task 

- [x] 网络构建
- [x] 优化器及损失函数实现
- [x] 程序预处理
- [x] 模型加载和参数保存
- [ ] 模型性能优化

## 项目介绍

项目采用卷积层和Relu层搭建简单的參差网络模型，并在此基础上辅助图像处理算法实现VDSR模型，受限于算力和手动实现无法达到较好的性能因而对原有模型进行了简化，减少了的层数。

## 项目结构

```shell
.
├── data
│   ├── preprocess        # 预处理好的数据
│   ├── test_data         # 测试图片
│   ├── train_data        # 低分辨率训练集
│   ├── train_label       # 高分辨率GroundTruth
├── log
│   └── 开发日志.md
├── main.py
├── model                       # 模型文件
├── README.md
└── utils
    ├── BatchProcess.py         # 批训练和测试
    ├── DataLoader.py           # 数据集加载
    ├── Layer.py                # 网络单层结构实现
    ├── Net.py                  # 模型网络搭建
    ├── Optimizer.py            # 优化器和损失函数
    └── PreProcess.py           # 预处理和部分工具
```

## 注意事项

- 部分内容采用了GPU,因而需要采用GPU版的pytorch
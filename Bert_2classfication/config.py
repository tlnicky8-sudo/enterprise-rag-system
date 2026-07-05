from pathlib import Path

import torch


class Config:
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent

    # 数据路径：train 训练，test 边训边评并保存最优模型
    train_datapath = str(base_dir / "train_data" / "train.jsonl")
    test_datapath = str(base_dir / "train_data" / "test.jsonl")

    # 每 N 个训练 batch 在完整 test 集上评估一次
    eval_every_n_batches = 100

    # 预训练 BERT
    bert_model_path = str(project_root / "models" / "bert-base-chinese")

    # 训练超参
    max_length = 128
    batch_size = 16
    epochs = 5
    learning_rate = 5e-5
    num_labels = 2
    freeze_backbone = True  # True 时只训练线性分类头

    # 模型保存路径
    save_model = str(project_root / "models" / "bert_outputs" / "best_intent_classifier.pt")

    # 设备
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # 标签映射
    label_to_id = {
        "通用知识": 0,
        "专业咨询": 1,
        "0": 0,
        "1": 1,
        0: 0,
        1: 1,
    }
    id_to_label = {
        0: "通用知识",
        1: "专业咨询",
    }

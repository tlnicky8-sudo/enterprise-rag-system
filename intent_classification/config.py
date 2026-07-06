from pathlib import Path

import torch


def _is_valid_local_bert(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").exists():
        return False
    return (path / "vocab.txt").exists() or (path / "tokenizer.json").exists()


def _load_project_bert_base_path(project_root: Path) -> Path | None:
    """读取主项目 config.ini 中的 bert_base_path（与运行时意图分类一致）。"""
    try:
        import sys

        root = str(project_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        from base.config import Config as ProjectConfig

        return Path(ProjectConfig().BERT_BASE_PATH)
    except Exception:
        return None


class Config:
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent

    # 数据路径：train 训练，test 边训边评并保存最优模型
    train_datapath = str(base_dir / "train_data" / "train.jsonl")
    test_datapath = str(base_dir / "train_data" / "test.jsonl")

    # 每 N 个训练 batch 在完整 test 集上评估一次
    eval_every_n_batches = 100

    # 预训练 BERT：优先 config.ini → 项目内 models/ → HuggingFace Hub
    bert_local_path = project_root / "models" / "bert-base-chinese"
    bert_hub_id = "bert-base-chinese"

    # 训练超参
    max_length = 128
    batch_size = 16
    epochs = 5
    learning_rate = 5e-5
    num_labels = 2
    freeze_backbone = True  # True 时只训练线性分类头

    # 模型保存路径
    save_model = str(project_root / "models" / "bert_outputs" / "best_intent_classifier.pt")
    save_quantized_model = str(
        project_root / "models" / "bert_outputs" / "best_intent_classifier_int8.pt"
    )

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

    @classmethod
    def resolve_bert_pretrained(cls) -> str:
        """返回可用的 BERT 预训练路径。"""
        candidates = [
            _load_project_bert_base_path(cls.project_root),
            cls.bert_local_path,
        ]
        for path in candidates:
            if path and _is_valid_local_bert(path):
                return str(path)
        return cls.bert_hub_id

    @classmethod
    def bert_pretrained_description(cls) -> str:
        source = cls.resolve_bert_pretrained()
        if source == cls.bert_hub_id:
            configured = _load_project_bert_base_path(cls.project_root)
            hint = configured or cls.bert_local_path
            return f"HuggingFace Hub: {cls.bert_hub_id} (本地未找到: {hint})"
        return f"local: {source}"

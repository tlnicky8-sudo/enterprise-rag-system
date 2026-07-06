import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizer

from config import Config

config = Config()
_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        source = config.resolve_bert_pretrained()
        print(f"[BERT] 加载 tokenizer: {config.bert_pretrained_description()}")
        _tokenizer = BertTokenizer.from_pretrained(source)
    return _tokenizer


def normalize_label(label):
    if label in config.label_to_id:
        return config.label_to_id[label]
    raise ValueError(f"不支持的标签: {label!r}，请使用 通用知识/专业咨询 或 0/1")


def _parse_json_record(item, source):
    if not isinstance(item, dict):
        raise ValueError(f"{source} 记录必须是对象，实际类型: {type(item)!r}")

    query = item.get("query") or item.get("text") or item.get("question")
    label = item.get("label") if "label" in item else item.get("intent")
    if not query or label is None:
        raise ValueError(f"{source} 缺少 query/text/question 或 label/intent: {item}")

    return str(query).strip(), normalize_label(label)


def load_raw_file(datapath):
    """读取 JSONL / JSON / TSV，返回 [(query, label_id), ...]。"""
    path = Path(datapath)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")

    result_list = []

    if path.suffix.lower() in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8") as file:
            if path.suffix.lower() == ".json":
                payload = json.load(file)
                lines = payload if isinstance(payload, list) else payload.get("data", [])
                for item in tqdm(lines, desc=f"处理 {path.name}"):
                    query, label = _parse_json_record(item, path.name)
                    result_list.append((query, label))
            else:
                for line_no, line in enumerate(tqdm(file, desc=f"处理 {path.name}"), start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        print(f"{path.name}:{line_no} JSON 解析失败: {exc}")
                        continue
                    try:
                        query, label = _parse_json_record(item, f"{path.name}:{line_no}")
                    except ValueError as exc:
                        print(exc)
                        continue
                    result_list.append((query, label))
        return result_list

    with path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    for line_no, line in enumerate(tqdm(lines, desc=f"处理 {path.name}"), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            print(f"{path.name}:{line_no} 格式错误，期望 query\\tlabel，实际: {line}")
            continue
        query, label = parts
        try:
            result_list.append((query.strip(), normalize_label(label.strip())))
        except ValueError as exc:
            print(f"{path.name}:{line_no} {exc}")
            continue

    return result_list


def load_train_test_lists():
    """train 用于训练，test 用于边训边评并保存最优模型。"""
    train_list = load_raw_file(config.train_datapath)
    test_list = load_raw_file(config.test_datapath)
    return train_list, test_list


class IntentDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.sample_len = len(self.data_list)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, index):
        index = min(max(index, 0), self.sample_len - 1)
        query, label = self.data_list[index]
        return query, label


def collate_fn(batch_data):
    queries, labels = zip(*batch_data)

    encoded = get_tokenizer()(
        list(queries),
        padding="max_length",
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = torch.tensor(labels, dtype=torch.long)
    return input_ids, attention_mask, labels


def build_dataloader_from_list(data_list, shuffle=False):
    if not data_list:
        raise ValueError("数据列表为空，无法构建 DataLoader")

    dataset = IntentDataset(data_list)
    return DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
    )


def build_dataloader(datapath, shuffle=False):
    data_list = load_raw_file(datapath)
    if not data_list:
        raise ValueError(f"未从 {datapath} 读到有效样本")
    return build_dataloader_from_list(data_list, shuffle=shuffle)


if __name__ == "__main__":
    train_list, test_list = load_train_test_lists()
    print(f"训练集样本数: {len(train_list)}")
    print(f"测试集样本数: {len(test_list)}")
    print(f"样例: {train_list[:3]}")

    for input_ids, attention_mask, labels in build_dataloader_from_list(test_list, shuffle=False):
        print("input_ids-->", input_ids.shape)
        print("attention_mask-->", attention_mask.shape)
        print("labels-->", labels)
        print("label names-->", [config.id_to_label[int(x)] for x in labels.tolist()])
        break

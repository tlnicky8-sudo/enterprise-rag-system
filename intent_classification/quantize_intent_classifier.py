"""量化已训练的意图分类器（动态 INT8），减小体积并加快 CPU 推理。

用法:
    cd intent_classification
    python quantize_intent_classifier.py

默认读取 config.save_model（best_intent_classifier.pt），
输出到同目录 best_intent_classifier_int8.pt。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bert_model import BertClassifierModel
from config import Config
from data_preprocessing import build_dataloader_from_list, load_train_test_lists

config = Config()
QUANT_CPU = "cpu"


def _setup_quant_engine() -> None:
    """为当前平台选择可用的 PyTorch 量化后端。"""
    if hasattr(torch.backends, "quantized"):
        for engine in ("qnnpack", "fbgemm", "onednn"):
            if engine in torch.backends.quantized.supported_engines:
                torch.backends.quantized.engine = engine
                return
    raise RuntimeError(
        "当前 PyTorch 未启用量化引擎，无法执行 INT8 量化。"
        "请确认安装的是带 QNNPACK 的 PyTorch（Mac/ARM 常用 qnnpack）。"
    )


def _checkpoint_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def load_float_model(checkpoint: Path, device: str) -> BertClassifierModel:
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"未找到训练好的模型: {checkpoint}\n"
            "请先运行: python train_intent_classifier.py"
        )

    model = BertClassifierModel().to(device)
    try:
        state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate(model, test_dataloader, sample_count: int, device: str) -> dict:
    preds: list[int] = []
    labels: list[int] = []

    with torch.no_grad():
        for input_ids, attention_mask, batch_labels in tqdm(
            test_dataloader, desc="评估", leave=False
        ):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids, attention_mask)
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            labels.extend(batch_labels.tolist())

    if len(labels) != sample_count:
        raise RuntimeError(f"测试样本数不一致: 期望 {sample_count}, 实际 {len(labels)}")

    return {
        "f1": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds),
    }


def benchmark_latency(model, test_dataloader, device: str, rounds: int = 3) -> float:
    """返回单条样本平均推理耗时（毫秒）。"""
    model.eval()
    batches = list(test_dataloader)
    if not batches:
        return 0.0

    input_ids, attention_mask, _ = batches[0]
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    batch_size = input_ids.size(0)

    with torch.no_grad():
        for _ in range(2):
            model(input_ids, attention_mask)

        start = time.perf_counter()
        for _ in range(rounds):
            for ids, mask, _ in batches:
                model(ids.to(device), mask.to(device))
        elapsed = time.perf_counter() - start

    total_samples = batch_size * rounds * len(batches)
    return elapsed / total_samples * 1000


def quantize_model(model: BertClassifierModel) -> BertClassifierModel:
    """对 Linear 层做动态 INT8 量化（适合 CPU 推理）。"""
    _setup_quant_engine()
    model_cpu = model.to(QUANT_CPU).eval()
    quantized = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return quantized


def save_quantized_model(model, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "torch_dynamic_int8",
        "quantize_targets": ["Linear"],
        "num_labels": config.num_labels,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, output_path)


def load_quantized_model(checkpoint: Path, device: str = QUANT_CPU) -> BertClassifierModel:
    """从量化 checkpoint 恢复模型（需在同一脚本/环境中使用）。"""
    _setup_quant_engine()
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)

    if isinstance(payload, dict) and payload.get("format") == "torch_dynamic_int8":
        model = quantize_model(BertClassifierModel().to(device))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    raise ValueError(f"不支持的量化模型格式: {checkpoint}")


def parse_args():
    parser = argparse.ArgumentParser(description="量化意图分类 BERT 模型（动态 INT8）")
    parser.add_argument(
        "--input",
        default=config.save_model,
        help="浮点模型 checkpoint（默认 config.save_model）",
    )
    parser.add_argument(
        "--output",
        default=str(Path(config.save_model).with_name("best_intent_classifier_int8.pt")),
        help="量化模型输出路径",
    )
    parser.add_argument(
        "--device",
        default=config.device,
        help="浮点模型评估设备（量化后固定 CPU）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    print("=" * 50)
    print("  意图分类模型量化 (动态 INT8)")
    print("=" * 50)
    print(f"  输入: {input_path}")
    print(f"  输出: {output_path}")
    print(f"  BERT: {config.bert_pretrained_description()}")

    _, test_list = load_train_test_lists()
    test_dataloader = build_dataloader_from_list(test_list, shuffle=False)
    test_count = len(test_list)

    float_model = load_float_model(input_path, args.device)
    float_metrics = evaluate(float_model, test_dataloader, test_count, args.device)
    float_latency = benchmark_latency(float_model, test_dataloader, args.device)

    quantized_model = quantize_model(float_model)
    quant_metrics = evaluate(quantized_model, test_dataloader, test_count, QUANT_CPU)
    quant_latency = benchmark_latency(quantized_model, test_dataloader, QUANT_CPU)

    save_quantized_model(quantized_model, output_path)

    input_mb = _checkpoint_size_mb(input_path)
    output_mb = _checkpoint_size_mb(output_path)

    print("\n[浮点模型]")
    print(f"  F1={float_metrics['f1']:.4f}  Acc={float_metrics['accuracy']:.4f}")
    print(f"  延迟≈{float_latency:.2f} ms/条  体积={input_mb:.1f} MB")

    print("\n[INT8 量化模型]")
    print(f"  F1={quant_metrics['f1']:.4f}  Acc={quant_metrics['accuracy']:.4f}")
    print(f"  延迟≈{quant_latency:.2f} ms/条  体积={output_mb:.1f} MB")

    print("\n[对比]")
    if input_mb > 0:
        print(f"  体积减少: {input_mb - output_mb:.1f} MB ({(1 - output_mb / input_mb) * 100:.1f}%)")
    print(f"  F1 变化: {quant_metrics['f1'] - float_metrics['f1']:+.4f}")

    print(f"\n量化完成 -> {output_path}")
    print("说明: INT8 模型适合 CPU 推理；Mac MPS/CUDA 请继续用浮点 checkpoint。")


if __name__ == "__main__":
    main()

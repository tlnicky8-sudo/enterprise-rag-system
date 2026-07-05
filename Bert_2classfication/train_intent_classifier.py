from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from bert_model import BertClassifierModel
from config import Config
from data_preprocessing import build_dataloader_from_list, load_train_test_lists

config = Config()


def eval_model(model, test_dataloader, sample_count):
    """在完整 test 集上评估，不反向传播、不更新参数。"""
    all_pred_result = []
    all_true_result = []

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="测试中", leave=False):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            labels = labels.to(config.device)

            pred_output = model(input_ids, attention_mask)
            pred_index = torch.argmax(pred_output, dim=-1)

            all_pred_result.extend(pred_index.cpu().tolist())
            all_true_result.extend(labels.cpu().tolist())

    if was_training:
        model.train()

    if len(all_true_result) != sample_count:
        raise RuntimeError(
            f"测试样本数不一致: 期望 {sample_count}, 实际 {len(all_true_result)}"
        )

    f1score = f1_score(all_true_result, all_pred_result, average="macro")
    accuracy = accuracy_score(all_true_result, all_pred_result)
    precision = precision_score(all_true_result, all_pred_result, average="macro", zero_division=0)
    recall = recall_score(all_true_result, all_pred_result, average="macro", zero_division=0)
    return f1score, accuracy, precision, recall


def train_and_eval():
    train_list, test_list = load_train_test_lists()
    train_dataloader = build_dataloader_from_list(train_list, shuffle=True)
    test_dataloader = build_dataloader_from_list(test_list, shuffle=False)
    test_sample_count = len(test_list)

    model = BertClassifierModel().to(device=config.device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        params=[param for param in model.parameters() if param.requires_grad],
        lr=config.learning_rate,
    )

    Path(config.save_model).parent.mkdir(parents=True, exist_ok=True)
    best_f1score = 0.0

    print(f"device={config.device}")
    print(f"train={config.train_datapath} ({len(train_list)} 条)")
    print(f"test={config.test_datapath} ({test_sample_count} 条)")
    print(f"save_model={config.save_model}")
    print(f"freeze_backbone={config.freeze_backbone}")
    print(f"eval_every_n_batches={config.eval_every_n_batches}")

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0

        progress = tqdm(train_dataloader, desc=f"Epoch {epoch}/{config.epochs}")
        for i, batch in enumerate(progress, start=1):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            labels = labels.to(config.device)

            pred_output = model(input_ids, attention_mask)
            loss_value = loss_fn(pred_output, labels)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            total_loss += loss_value.item()
            progress.set_postfix(loss=f"{total_loss / i:.4f}")

            should_eval = (
                i % config.eval_every_n_batches == 0
                or i == len(train_dataloader)
            )
            if not should_eval:
                continue

            f1score, accuracy, precision, recall = eval_model(
                model,
                test_dataloader,
                test_sample_count,
            )
            print(
                f"Epoch {epoch} Batch {i}: "
                f"test_f1={f1score:.4f}, acc={accuracy:.4f}, "
                f"precision={precision:.4f}, recall={recall:.4f} "
                f"(test_n={test_sample_count})"
            )

            if f1score > best_f1score:
                torch.save(model.state_dict(), config.save_model)
                best_f1score = f1score
                print(f"保存最优模型，best_test_f1={best_f1score:.4f} -> {config.save_model}")

        avg_loss = total_loss / max(len(train_dataloader), 1)
        print(f"Epoch {epoch} 平均损失: {avg_loss:.4f}")

    print(f"训练完成，最优 test F1: {best_f1score:.4f}")
    print(f"模型路径: {config.save_model}")


if __name__ == "__main__":
    train_and_eval()

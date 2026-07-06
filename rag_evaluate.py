"""RAGAS 离线评估脚本。"""
import json
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from base import Config

conf = Config()
DATA_PATH = conf.PROJECT_ROOT / "data" / "assessment_data" / "rag_evaluate_data.json"
OUTPUT_PATH = conf.PROJECT_ROOT / "data" / "assessment_data" / "ragas_evaluation_results.csv"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"评估数据不存在: {DATA_PATH}")

    with DATA_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    dataset = Dataset.from_dict(
        {
            "question": [item["question"] for item in data],
            "answer": [item["answer"] for item in data],
            "contexts": [item["context"] for item in data],
            "ground_truth": [item["ground_truth"] for item in data],
        }
    )

    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("API-KEY")
    llm = ChatTongyi(model="qwen-max", api_key=dashscope_api_key)
    embeddings = DashScopeEmbeddings(
        dashscope_api_key=dashscope_api_key,
        model="text-embedding-v3",
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    print("RAGAS 评估结果：")
    print(result)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(OUTPUT_PATH, index=False)
    print(f"结果已保存: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

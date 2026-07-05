"""法律语料六步入库：解析 -> 清洗 -> 切块 -> 增强 -> 索引 -> 血缘治理。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from base import Config, logger
from core.ingest import IngestPipeline
from core.vector_store import VectorStore

conf = Config()


def check_collection_has_data(vs):
    """快速检查集合中是否已有数据。"""
    try:
        results = vs.client.query(
            collection_name=vs.collection_name,
            filter='id != ""',
            output_fields=["id"],
            limit=1,
        )
        return len(results) > 0
    except Exception as exc:
        logger.debug(f"检查已有数据时出现预期异常: {exc}")
        return False


def resolve_source_dir(source):
    """基于项目根目录解析语料目录，避免依赖当前工作目录。"""
    return Path(conf.PROJECT_ROOT) / "data" / f"{source}_data"


def main(skip_if_exists=False, dry_run=False, enhance=False, doc_version="1"):
    print("\n" + "=" * 50)
    print("  法律语料六步入库")
    print("=" * 50)
    print("  ① 解析  ② 清洗  ③ 切块  ④ 增强  ⑤ 索引  ⑥ 血缘")

    vs = None
    has_data = False
    if dry_run:
        print("\n[准备] dry-run 模式：不连接 Milvus，不写中间文件")
    else:
        # 入库不需要加载 reranker
        print("\n[准备] 连接 Milvus...")
        try:
            vs = VectorStore(
                collection_name=conf.MILVUS_COLLECTION_NAME,
                host=conf.MILVUS_HOST,
                port=conf.MILVUS_PORT,
                database=conf.MILVUS_DATABASE_NAME,
                load_reranker=False,
            )
            print(f"  已连接到 {conf.MILVUS_HOST}:{conf.MILVUS_PORT}")
            print(f"  数据库: {conf.MILVUS_DATABASE_NAME}")
            print(f"  集合: {conf.MILVUS_COLLECTION_NAME}")
        except Exception as exc:
            print(f"  错误：无法连接 Milvus - {exc}")
            print("  请确保 Milvus 服务已启动，且 BGE-M3 模型路径正确")
            logger.error(f"初始化 VectorStore 失败: {exc}")
            return False

        print("\n[准备] 检查已有数据...")
        has_data = check_collection_has_data(vs)
        if has_data and skip_if_exists:
            print(f"  集合 '{conf.MILVUS_COLLECTION_NAME}' 中已有数据，跳过处理")
            print("\n" + "=" * 50)
            print("  数据就绪，可以直接启动 Web 服务")
            print("=" * 50 + "\n")
            return True
        if has_data:
            print("  集合已有数据，将基于内容 MD5 执行 upsert（相同内容不会重复）")
        else:
            print("  集合为空，开始写入")

    print("\n[入库] 开始六步流水线...")
    total = 0
    reports = []
    missing_dirs = []
    pipeline = IngestPipeline(vs, conf=conf)

    for source in conf.VALID_SOURCES:
        dir_path = resolve_source_dir(source)
        if not dir_path.exists():
            print(f"  警告：目录 {dir_path} 不存在，跳过")
            missing_dirs.append(str(dir_path))
            continue

        print(f"  处理目录: {dir_path}")
        try:
            report = pipeline.run_directory(
                dir_path,
                source=source,
                dry_run=dry_run,
                enhance=enhance,
                doc_version=doc_version,
            )
            reports.append(report)
            written = int(report.get("chunks_indexed", 0))
            total += written
            print(
                f"  完成 source={source} docs={report.get('documents', 0)} "
                f"chunks={written} report=ingest_report_{report.get('run_id')}.json"
            )
        except Exception as exc:
            print(f"  处理失败: {exc}")
            logger.error(f"处理 {dir_path} 失败: {exc}")

    print(f"\n  总计写入/更新 {total} 个文档块到向量库")
    if missing_dirs and total == 0:
        print("  错误：所有语料目录都不存在或没有有效内容")
        print("  请将文档放到 data/{source}_data/，例如 data/labor_law_data/")
        return False

    print("\n" + "=" * 50)
    if total > 0 or has_data or dry_run:
        print("  数据处理完成！")
        if dry_run:
            print("  dry-run 模式未实际写入 Milvus")
        print("=" * 50 + "\n")
        return True

    print("  数据处理失败：没有写入任何文档块")
    print("=" * 50 + "\n")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest legal corpus into Milvus")
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="若集合中已有数据则跳过处理",
    )
    parser.add_argument("--dry-run", action="store_true", help="只跑流程，不写入 Milvus 和中间文件")
    parser.add_argument("--enhance", action="store_true", help="启用关键词和可能问题增强")
    parser.add_argument("--doc-version", default="1", help="入库文档版本号")
    args = parser.parse_args()

    success = main(
        skip_if_exists=args.skip_if_exists,
        dry_run=args.dry_run,
        enhance=args.enhance,
        doc_version=args.doc_version,
    )
    if not success:
        print("\n按任意键退出...")
        try:
            input()
        except EOFError:
            pass
        sys.exit(1)
    sys.exit(0)

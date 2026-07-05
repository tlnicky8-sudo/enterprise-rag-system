"""FAQ 入库：JSONL/JSON -> Redis。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from base import Config, logger
from core.faq.loader import load_faq_pairs, to_question_answer_pairs


def resolve_data_path(conf, source_path=None):
    if source_path:
        path = Path(source_path)
        if not path.is_absolute():
            path = conf.PROJECT_ROOT / path
        return path
    return conf.PROJECT_ROOT / "data" / "faq_data" / "faq_pairs.jsonl"


def main(source_path=None, domain=None, dry_run=False, replace=False):
    conf = Config()
    domain = domain or conf.DOMAIN_NAME
    data_path = resolve_data_path(conf, source_path)

    print("\n" + "=" * 50)
    print("  FAQ 数据入库 (Redis)")
    print("=" * 50)
    print(f"  数据文件: {data_path}")
    print(f"  领域: {domain}")

    try:
        pairs, skipped = load_faq_pairs(data_path, default_subject=domain)
    except Exception as exc:
        print(f"\n  错误：读取 FAQ 数据失败 - {exc}")
        logger.error(f"Load FAQ data failed: {exc}")
        return False

    print(f"\n[解析] 有效 FAQ 条数: {len(pairs)}")
    if skipped:
        print(f"  跳过无效/重复记录: {skipped}")
    if not pairs:
        print("  错误：没有可入库的 FAQ 数据")
        return False

    if dry_run:
        print("\n[dry-run] 未写入 Redis")
        print("=" * 50 + "\n")
        return True

    from core.faq_service import FAQService

    service = FAQService(conf)
    if not service.enabled:
        print("\n  错误：无法连接 Redis 或初始化 FAQ 服务")
        return False

    try:
        if replace:
            cleared = service.clear()
            print(f"\n[Redis] 已清空旧 FAQ 数据: {cleared} 条")

        print("[Redis] 写入问答对并预热语义缓存...")
        qa_pairs = to_question_answer_pairs(pairs)
        preheated = service.preheat_cache(qa_pairs)
        print(f"  已写入/更新: {len(qa_pairs)} 条")
        print(f"  语义缓存预热: {preheated} 条")

        print("\n" + "=" * 50)
        print("  FAQ 入库完成！")
        print("=" * 50 + "\n")
        return True
    except Exception as exc:
        print(f"\n  错误：FAQ 入库失败 - {exc}")
        logger.error(f"FAQ ingest failed: {exc}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import FAQ pairs into Redis")
    parser.add_argument(
        "--source",
        default=None,
        help="FAQ 数据文件路径，默认 data/faq_data/faq_pairs.jsonl",
    )
    parser.add_argument("--domain", default=None, help="领域名称，默认读取 config.ini 的 domain.name")
    parser.add_argument("--dry-run", action="store_true", help="只解析校验，不写入 Redis")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="入库前先清空 Redis 中的旧 FAQ 数据",
    )
    args = parser.parse_args()

    success = main(
        source_path=args.source,
        domain=args.domain,
        dry_run=args.dry_run,
        replace=args.replace,
    )
    if not success:
        sys.exit(1)

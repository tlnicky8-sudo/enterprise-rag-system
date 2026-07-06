"""智能问答系统 - CLI 入口"""
import uuid

from openai import OpenAI

from base import Config, logger
from core.qa_pipeline import QAPipeline
from core.rag_system import RAGSystem
from core.vector_store import VectorStore

conf = Config()


def main():
    try:
        client = OpenAI(api_key=conf.DASHSCOPE_API_KEY, base_url=conf.DASHSCOPE_BASE_URL)
    except Exception as exc:
        logger.error(f"初始化 OpenAI 客户端失败: {exc}")
        print("错误：无法初始化语言模型客户端。")
        return

    try:
        vector_store = VectorStore(
            collection_name=conf.MILVUS_COLLECTION_NAME,
            host=conf.MILVUS_HOST,
            port=conf.MILVUS_PORT,
            database=conf.MILVUS_DATABASE_NAME,
        )
    except Exception as exc:
        logger.error(f"初始化 VectorStore 失败: {exc}")
        print("错误：无法连接到向量数据库。")
        return

    def call_dashscope(prompt):
        try:
            completion = client.chat.completions.create(
                model=conf.LLM_MODEL,
                messages=[
                    {"role": "system", "content": conf.GENERATION_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
            )
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content
            logger.error("LLM API 返回无效响应")
            return "错误: LLM 返回无效响应"
        except Exception as exc:
            logger.error(f"LLM API 调用失败: {exc}")
            return f"错误: 调用 LLM 失败 - {exc}"

    try:
        rag_system = RAGSystem(vector_store, call_dashscope)
        pipeline = QAPipeline(rag_system=rag_system)
    except Exception as exc:
        logger.error(f"初始化问答流水线失败: {exc}")
        print("错误：无法初始化问答流水线。")
        return

    session_id = str(uuid.uuid4())
    valid_sources = conf.VALID_SOURCES
    print("\n欢迎使用企业知识库智能问答系统！")
    print(f"支持的文档类别：{valid_sources}")
    print("输入您的问题，或输入 'exit' 退出。")

    while True:
        query = input("\n请输入您的问题：")
        if query.lower() == "exit":
            print("再见！")
            break

        source_filter_input = input(
            f"请输入文档类别 ({'/'.join(valid_sources)}) (直接回车默认不过滤)："
        ).strip()
        source_filter = source_filter_input if source_filter_input in valid_sources else None
        if source_filter_input and source_filter is None:
            print(f"提示：输入的来源 '{source_filter_input}' 无效，将不过滤。")

        try:
            print("正在生成答案，请稍候...")
            result = pipeline.answer(
                query,
                session_id,
                source_filter=source_filter,
                stream=False,
            )
            print("-" * 30)
            print(f"问题: {query}")
            print(f"来源: {result.source}")
            print(f"回答: {result.answer}")
            print("-" * 30)
        except Exception as exc:
            logger.error(f"处理查询失败: {exc}")
            print("抱歉，处理您的问题时遇到了错误，请稍后重试。")


if __name__ == "__main__":
    main()

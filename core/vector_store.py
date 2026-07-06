from core.cache_policy import normalize_rerank_score
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 相关类，用于操作向量数据库
"""
MilvusClient：连接 Milvus 服务器的客户端。
DataType：定义字段数据类型（如 VARCHAR、FLOAT_VECTOR）。
AnnSearchRequest：封装一个 ANN（近似最近邻）搜索请求（稠密或稀疏）。
WeightedRanker：用于混合搜索时对多个查询结果加权融合。
"""
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象
from langchain_core.documents import Document
# 导入 CrossEncoder，用于重排序和 NLI 判断
from sentence_transformers import CrossEncoder
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值
import hashlib
import json
from base import logger, Config
conf = Config()

# 定义 VectorStore 类，封装向量存储和检索功能
class VectorStore:
    MAX_VARCHAR_LENGTH = 65535
    UPSERT_BATCH_SIZE = 64

    # 初始化方法，设置向量存储的基本参数
    def __init__(self,
                 collection_name=conf.MILVUS_COLLECTION_NAME,
                 host=conf.MILVUS_HOST,
                 port=conf.MILVUS_PORT,
                 database=conf.MILVUS_DATABASE_NAME,
                 load_reranker=True):
        # 设置 Milvus 集合名称
        self.collection_name = collection_name
        # 设置 Milvus 主机地址
        self.host = host
        # 设置 Milvus 端口号
        self.port = port
        # 设置 Milvus 数据库名称
        self.database = database
        # 设置日志记录器
        self.logger = logger
        # 入库场景可不加载重排模型
        self.reranker = CrossEncoder(conf.BGE_RERANKER_PATH) if load_reranker else None
        # 初始化 BGE-M3 嵌入函数，使用 CPU 设备，不启用 FP16(半精度)
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=conf.BGE_M3_PATH,
            use_fp16=False, device="cpu")
        # 获取稠密向量的维度
        self.dense_dim = self.embedding_function.dim["dense"]
        # 初始化 Milvus 客户端，并确保数据库存在
        self.uri = f"http://{self.host}:{self.port}"
        self._ensure_database()
        self.client = MilvusClient(uri=self.uri, db_name=self.database)
        # 调用方法创建或加载 Milvus 集合
        self._create_or_load_collection()

    def _ensure_database(self):
        """若配置的数据库不存在则创建。"""
        root_client = MilvusClient(uri=self.uri)
        try:
            databases = root_client.list_databases()
        except Exception as exc:
            logger.warning(f"无法列出 Milvus 数据库，继续使用默认连接: {exc}")
            return

        if self.database not in databases:
            root_client.create_database(self.database)
            logger.info(f"已创建 Milvus 数据库: {self.database}")

    @staticmethod
    def _truncate(text, max_length=MAX_VARCHAR_LENGTH):
        text = text or ""
        if len(text) <= max_length:
            return text
        return text[:max_length]

    @staticmethod
    def _sparse_to_dict(sparse_row):
        """兼容不同版本 milvus-model 的稀疏向量格式。"""
        if hasattr(sparse_row, "col") and hasattr(sparse_row, "data"):
            indices = sparse_row.col
            values = sparse_row.data
        elif hasattr(sparse_row, "indices") and hasattr(sparse_row, "data"):
            indices = sparse_row.indices
            values = sparse_row.data
        elif isinstance(sparse_row, dict):
            return {int(k): float(v) for k, v in sparse_row.items()}
        else:
            raise TypeError(f"不支持的稀疏向量格式: {type(sparse_row)!r}")
        return {int(idx): float(value) for idx, value in zip(indices, values)}

    # 定义私有方法，创建或加载 Milvus 集合
    def _create_or_load_collection(self):
        # 检查指定集合是否已存在
        if not self.client.has_collection(self.collection_name):
            # 创建集合 Schema，禁用自动 ID，启用动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加来源类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )

            # 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                         index_params=index_params)
            # 记录创建集合的日志
            logger.info(f"已创建集合 {self.collection_name}")
        # 如果集合已存在
        else:
            # 记录加载集合的日志
            logger.info(f"已加载集合 {self.collection_name}")
        # 将集合加载到内存，确保可立即查询
        self.client.load_collection(self.collection_name)

    # 定义方法，向向量存储添加文档
    def add_documents(self, documents):
        documents = [
            doc for doc in documents
            if doc.page_content and str(doc.page_content).strip()
        ]
        if not documents:
            logger.warning("没有可写入的非空文档块")
            return 0

        total_written = 0
        for start in range(0, len(documents), self.UPSERT_BATCH_SIZE):
            batch_docs = documents[start:start + self.UPSERT_BATCH_SIZE]
            texts = [doc.metadata.get("index_text") or doc.page_content for doc in batch_docs]
            embeddings = self.embedding_function(texts)
            data = []

            for i, doc in enumerate(batch_docs):
                text = self._truncate(doc.page_content)
                parent_content = self._truncate(doc.metadata.get("parent_content", text))
                text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                sparse_vector = self._sparse_to_dict(embeddings["sparse"][i])
                row = {
                    "id": text_hash,
                    "text": text,
                    "dense_vector": embeddings["dense"][i],
                    "sparse_vector": sparse_vector,
                    "parent_id": str(doc.metadata.get("parent_id", f"unknown_{text_hash[:8]}")),
                    "parent_content": parent_content,
                    "source": str(doc.metadata.get("source", "unknown"))[:50],
                    "timestamp": str(doc.metadata.get("timestamp", "unknown"))[:50],
                }
                # Dynamic fields: lineage and optional enrichment metadata.
                for key in (
                    "child_id",
                    "doc_id",
                    "source_file",
                    "file_hash",
                    "markdown_path",
                    "doc_type",
                    "section_path",
                    "content_hash",
                    "ingest_time",
                    "keywords",
                    "hypothetical_questions",
                    "chunk_index",
                    "chunk_type",
                    "version",
                ):
                    if key not in doc.metadata:
                        continue
                    value = doc.metadata[key]
                    if isinstance(value, (list, dict)):
                        row[key] = json.dumps(value, ensure_ascii=False)
                    else:
                        row[key] = value
                data.append(row)

            self.client.upsert(collection_name=self.collection_name, data=data)
            total_written += len(data)
            logger.info(f"已插入或更新 {len(data)} 个文档（累计 {total_written}）")

        return total_written

    # 定义方法，执行混合检索并重排序
    def hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None):
        if source_filter and source_filter not in conf.VALID_SOURCES:
            logger.warning("忽略无效来源过滤: %s", source_filter)
            source_filter = None

        # 使用 BGE-M3 嵌入函数生成查询的嵌入
        query_embeddings = self.embedding_function([query])
        # 获取查询的稠密向量
        dense_query_vector = query_embeddings["dense"][0]
        sparse_query_vector = self._sparse_to_dict(query_embeddings["sparse"][0])

        # 初始化过滤表达式，默认不过滤
        filter_expr = f"source == '{source_filter}'" if source_filter else ""
        # 创建稠密向量搜索请求
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": conf.MILVUS_NPROBE}},
            limit=k,
            expr=filter_expr  # 按来源过滤，例如 enterprise
        )
        # 创建稀疏向量搜索请求
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expr
        )

        # 创建加权排序器，稀疏向量权重 0.7，稠密向量权重 1.0
        ranker = WeightedRanker(conf.HYBRID_DENSE_WEIGHT, conf.HYBRID_SPARSE_WEIGHT)
        # 执行混合搜索，返回 Top-K 结果
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=[
                "text",
                "parent_id",
                "parent_content",
                "source",
                "timestamp",
                "source_file",
                "section_path",
                "doc_type",
                "doc_id",
            ]
        )[0]

        # 将搜索结果转换为 Document 对象列表
        sub_chunks = [self._doc_from_hit(hit["entity"]) for hit in results]
        parent_docs = self._get_unique_parent_docs(sub_chunks)
        top_rerank_score = 0.0

        if len(parent_docs) < 2:
            ranked_parent_docs = parent_docs[:conf.CANDIDATE_M]
            if ranked_parent_docs:
                top_rerank_score = conf.SHORT_CIRCUIT_RERANK_SCORE
        elif parent_docs:
            if self.reranker is None:
                self.reranker = CrossEncoder(conf.BGE_RERANKER_PATH)
            pairs = [[query, doc.page_content] for doc in parent_docs]
            scores = self.reranker.predict(pairs)
            top_rerank_score = normalize_rerank_score(max(float(score) for score in scores))
            ranked_parent_docs = [
                doc for _, doc in sorted(zip(scores, parent_docs), key=lambda item: item[0], reverse=True)
            ]
        else:
            ranked_parent_docs = []

        return ranked_parent_docs[:conf.CANDIDATE_M], top_rerank_score

    # 定义私有方法，从子块中提取去重的父文档
    def _get_unique_parent_docs(self, sub_chunks):
        # 初始化集合，用于存储已处理的父块内容（去重）
        parent_contents = set()
        # 初始化列表，用于存储唯一父文档
        unique_docs = []
        # 遍历所有子块
        for chunk in sub_chunks:
            # 获取子块的父块内容，默认为子块内容
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            # 检查父块内容是否非空且未重复
            if parent_content and parent_content not in parent_contents:
                # 创建新的 Document 对象，包含父块内容和元数据
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                # 将父块内容添加到去重集合
                parent_contents.add(parent_content)
        # 返回去重后的父文档列表
        return unique_docs

    # 定义私有方法，从 Milvus 查询结果创建 Document 对象
    def _doc_from_hit(self, hit):
        # 创建并返回 Document 对象，填充内容和元数据
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp"),
                "source_file": hit.get("source_file"),
                "section_path": hit.get("section_path"),
                "doc_type": hit.get("doc_type"),
                "doc_id": hit.get("doc_id"),
            },
        )

if __name__ == "__main__":
    print("请使用 setup_data.py 执行语料入库。")
    print("示例: python setup_data.py")





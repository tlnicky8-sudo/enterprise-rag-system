import time

from base import Config, logger
from core.citations import build_citations_from_docs
from core.llm_response import parse_llm_json_response
from core.llm_utils import iter_llm_output
from core.prompts import RAGPrompts
from core.query_classifier import QueryClassifier
from core.rag_models import GenerationResult, GroundingDecision
from core.strategy_selector import StrategySelector

conf = Config()


class RAGSystem:
    """Unified RAG orchestration used by CLI and Web."""

    def __init__(self, vector_store, llm, stream_llm=None):
        self.vector_store = vector_store
        self.llm = llm
        self.stream_llm = stream_llm or llm
        self.rag_prompt = RAGPrompts.rag_prompt(conf)
        self.query_classifier = QueryClassifier(conf=conf)
        self.strategy_selector = StrategySelector(conf)
        self.max_prompt_length = conf.MAX_PROMPT_LENGTH

    @staticmethod
    def _unpack_search_result(result):
        if isinstance(result, tuple) and len(result) == 2:
            docs, score = result
            return docs, float(score)
        return result, 0.0

    def _call_llm_text(self, prompt):
        raw = self.llm(prompt)
        if hasattr(raw, "__iter__") and not isinstance(raw, str):
            raw = "".join(str(chunk) for chunk in raw)
        return str(raw).strip()

    def _call_llm_json(self, prompt):
        parsed = parse_llm_json_response(self._call_llm_text(prompt))
        return parsed["answer"], parsed["confidence"]

    def _retrieve_with_hyde(self, query, source_filter=None):
        logger.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")
        hyde_prompt_template = RAGPrompts.hyde_prompt()
        try:
            hypo_answer = self._call_llm_text(hyde_prompt_template.format(query=query))
            logger.info(f"HyDE 生成的假设答案: '{hypo_answer[:100]}'")
            return self._unpack_search_result(
                self.vector_store.hybrid_search_with_rerank(
                    hypo_answer,
                    k=conf.RETRIEVAL_K,
                    source_filter=source_filter,
                )
            )
        except Exception as exc:
            logger.error(f"HyDE 策略执行失败: {exc}")
            return [], 0.0

    def _retrieve_with_subqueries(self, query, source_filter=None):
        logger.info(f"使用子查询策略进行检索 (查询: '{query}')")
        subquery_prompt_template = RAGPrompts.subquery_prompt()
        try:
            subqueries_text = self._call_llm_text(subquery_prompt_template.format(query=query))
            subqueries = [item.strip() for item in subqueries_text.split("\n") if item.strip()]
            logger.info(f"生成的子查询: {subqueries}")
            if not subqueries:
                logger.warning("未能生成有效的子查询")
                return [], 0.0

            all_docs = []
            top_score = 0.0
            for sub_q in subqueries[: conf.SUBQUERY_MAX_COUNT]:
                docs, score = self._unpack_search_result(
                    self.vector_store.hybrid_search_with_rerank(
                        sub_q,
                        k=conf.RETRIEVAL_K,
                        source_filter=source_filter,
                    )
                )
                all_docs.extend(docs)
                top_score = max(top_score, score)
                logger.info(f"子查询 '{sub_q}' 检索到 {len(docs)} 个文档")

            unique_docs_dict = {doc.page_content: doc for doc in all_docs}
            return list(unique_docs_dict.values()), top_score
        except Exception as exc:
            logger.error(f"子查询策略执行失败: {exc}")
            return [], 0.0

    def _retrieve_with_backtracking(self, query, source_filter=None):
        logger.info(f"使用回溯问题策略进行检索 (查询: '{query}')")
        backtrack_prompt_template = RAGPrompts.backtracking_prompt()
        try:
            simplified_query = self._call_llm_text(backtrack_prompt_template.format(query=query))
            logger.info(f"生成的回溯问题: '{simplified_query}'")
            return self._unpack_search_result(
                self.vector_store.hybrid_search_with_rerank(
                    simplified_query,
                    k=conf.RETRIEVAL_K,
                    source_filter=source_filter,
                )
            )
        except Exception as exc:
            logger.error(f"回溯问题策略执行失败: {exc}")
            return [], 0.0

    def retrieve_and_merge(self, query, source_filter=None, strategy=None):
        if not strategy:
            if conf.ENABLE_STRATEGY_SELECTOR:
                strategy = self.strategy_selector.select_strategy(query)
            else:
                strategy = conf.DEFAULT_STRATEGY

        if strategy == "回溯问题检索":
            ranked_docs, rerank_score = self._retrieve_with_backtracking(
                query,
                source_filter=source_filter,
            )
        elif strategy == "子查询检索":
            ranked_docs, rerank_score = self._retrieve_with_subqueries(
                query,
                source_filter=source_filter,
            )
        elif strategy == "假设问题检索":
            ranked_docs, rerank_score = self._retrieve_with_hyde(
                query,
                source_filter=source_filter,
            )
        else:
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            ranked_docs, rerank_score = self._unpack_search_result(
                self.vector_store.hybrid_search_with_rerank(
                    query,
                    k=conf.RETRIEVAL_K,
                    source_filter=source_filter,
                )
            )

        logger.info(f"策略 '{strategy}' 检索到 {len(ranked_docs)} 个候选文档")
        final_context_docs = ranked_docs[: conf.CANDIDATE_M]
        logger.info(f"最终选取 {len(final_context_docs)} 个文档作为上下文")
        return final_context_docs, rerank_score

    def _build_prompt(self, query, context="", history=""):
        return self.rag_prompt.format(
            context=context[: conf.MAX_CONTEXT_CHARS],
            question=query,
            phone=conf.CUSTOMER_SERVICE_PHONE,
            history=history,
        )

    @staticmethod
    def _build_context_with_citations(context_docs, citations):
        blocks = []
        for doc, citation in zip(context_docs, citations):
            blocks.append(f"[{citation['id']}] {doc.page_content}")
        return "\n\n".join(blocks)

    def _make_refusal(self, reason):
        return f"信息不足，无法基于已检索到的法律条文可靠回答。{reason}如需进一步确认，请联系人工客服，电话：{conf.CUSTOMER_SERVICE_PHONE}。"

    def _grounding_decision(self, source, has_context, rerank_score, citations):
        if source != "rag":
            return GroundingDecision(should_answer=True, grounded=False)
        if not conf.ENABLE_GROUNDING_GATE:
            return GroundingDecision(should_answer=True, grounded=bool(citations))
        if not conf.REQUIRE_CONTEXT_FOR_LEGAL_QA:
            return GroundingDecision(should_answer=True, grounded=bool(citations))
        if not has_context or not citations:
            return GroundingDecision(
                should_answer=False,
                grounded=False,
                refusal_reason="未检索到可引用的法律依据。",
            )
        min_score = conf.MIN_RERANK_SCORE
        if rerank_score < min_score:
            return GroundingDecision(
                should_answer=False,
                grounded=False,
                refusal_reason=f"检索相关性不足（{rerank_score:.2f} < {min_score:.2f}）。",
            )
        return GroundingDecision(should_answer=True, grounded=True)

    def _resolve_context(self, query, source_filter=None):
        query_category = self.query_classifier.predict_category(query)
        logger.info(f"查询分类结果：{query_category} (查询: '{query}')")

        if query_category == "通用知识":
            logger.info("查询为通用知识，跳过知识库检索")
            return "", "direct_llm", 0.0, False, []

        strategy = (
            self.strategy_selector.select_strategy(query)
            if conf.ENABLE_STRATEGY_SELECTOR
            else conf.DEFAULT_STRATEGY
        )
        context_docs, rerank_score = self.retrieve_and_merge(
            query,
            source_filter=source_filter,
            strategy=strategy,
        )
        if context_docs:
            citations = build_citations_from_docs(context_docs)
            context = self._build_context_with_citations(context_docs, citations)
            logger.info(f"构建上下文完成，包含 {len(context_docs)} 个文档块")
            return context, "rag", rerank_score, True, citations

        logger.info("未检索到相关文档，上下文为空")
        return "", "rag", 0.0, False, []

    def generate_answer(self, query, source_filter=None, history=""):
        start_time = time.time()
        logger.info(f"开始处理查询: '{query}', 来源过滤: {source_filter}")

        context, source, rerank_score, has_context, citations = self._resolve_context(
            query,
            source_filter=source_filter,
        )

        grounding = self._grounding_decision(source, has_context, rerank_score, citations)
        if not grounding.should_answer:
            answer = self._make_refusal(grounding.refusal_reason)
            processing_time = time.time() - start_time
            logger.info(
                "查询拒答 (耗时: %.2fs, source=%s, rerank=%.3f, reason=%s)",
                processing_time,
                source,
                rerank_score,
                grounding.refusal_reason,
            )
            return GenerationResult(
                answer=answer,
                source=source,
                rerank_score=rerank_score,
                has_context=has_context,
                citations=citations,
                grounded=False,
                refusal_reason=grounding.refusal_reason,
            )

        prompt_input = self._build_prompt(query, context=context, history=history)

        answer = ""
        llm_confidence = 0.0
        try:
            answer, llm_confidence = self._call_llm_json(prompt_input)
        except Exception as exc:
            logger.error(f"调用 LLM 生成答案失败: {exc}")
            answer = f"抱歉，处理您的问题时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"

        processing_time = time.time() - start_time
        logger.info(
            f"查询处理完成 (耗时: {processing_time:.2f}s, source={source}, "
            f"rerank={rerank_score:.3f}, confidence={llm_confidence:.3f})"
        )
        return GenerationResult(
            answer=answer,
            source=source,
            llm_confidence=llm_confidence,
            rerank_score=rerank_score,
            has_context=has_context,
            citations=citations,
            grounded=grounding.grounded,
            refusal_reason="",
        )

    def generate_answer_stream(self, query, source_filter=None, history=""):
        result = self.generate_answer(query, source_filter=source_filter, history=history)
        return iter_llm_output(result.answer), result

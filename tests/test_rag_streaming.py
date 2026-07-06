from core.rag_system import RAGSystem


def test_iter_answer_from_json_stream_extracts_split_answer():
    chunks = ['{"answer":"年假', "通过 OA ", '申请","confidence":0.9}']

    assert "".join(RAGSystem._iter_answer_from_json_stream(chunks)) == "年假通过 OA 申请"


def test_iter_answer_from_json_stream_handles_escapes():
    chunks = ['{"answer":"第一行\\n第二行 \\"重点\\"","confidence":0.8}']

    assert "".join(RAGSystem._iter_answer_from_json_stream(chunks)) == '第一行\n第二行 "重点"'

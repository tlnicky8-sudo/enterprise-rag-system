from collections.abc import Iterator


def collect_llm_output(result) -> str:
    """Collect a complete LLM response from a string or token iterator."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, Iterator):
        return "".join(str(chunk) for chunk in result)
    return str(result)


def iter_llm_output(result):
    """Normalize an LLM response into a token iterator."""
    if result is None:
        return iter(())
    if isinstance(result, str):
        return iter((result,))
    return iter(result)

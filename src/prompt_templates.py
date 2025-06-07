# === Prompt Templates ===

def template_triples(question: str, paths: list, options: list) -> str:
    """
    Template that formats knowledge as (head → tail) triples for each reasoning path.

    Args:
        question (str): The question to be answered.
        paths (list): List of paths, each a list of concepts [e1, ..., en].
        options (list): List of answer options.

    Returns:
        str: The formatted prompt string.
    """
    # Convert each path to a list of textual (h → t) triples
    lines = []
    for path in paths:
        triples = [f"({path[i]} → {path[i+1]})" for i in range(len(path) - 1)]
        lines.append("\n".join(triples))

    # Format answer options
    opt_lines = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]

    # Compose final prompt
    prompt = (
        f"Question: {question}\n"
        + "Options:\n" + "\n".join(opt_lines) + "\n"
        + "Knowledge:\n" + "\n\n".join(lines)
    )
    return prompt


def template_sentence(question: str, paths: list, options: list) -> str:
    """
    Template that formats reasoning paths as flat natural language sentences.

    Args:
        question (str): The input question.
        paths (list): List of concept paths.
        options (list): Answer choices.

    Returns:
        str: The composed prompt.
    """
    sentences = [" → ".join(path) for path in paths]
    opt_lines = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]

    prompt = (
        f"Question: {question}\n"
        + "Options:\n" + "\n".join(opt_lines) + "\n"
        + "Context:\n" + "\n\n".join(sentences)
    )
    return prompt


def template_graph_summary(question: str, paths: list, options: list) -> str:
    """
    Template that summarizes reasoning paths with a high-level overview.

    Args:
        question (str): Question text.
        paths (list): List of paths [e1, ..., en].
        options (list): List of candidate answers.

    Returns:
        str: The formatted summary-style prompt.
    """
    summaries = [
        f"The concept '{path[0]}' leads through {len(path)} steps to '{path[-1]}'"
        for path in paths
    ]
    opt_lines = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]

    prompt = (
        f"Question: {question}\n"
        + "Options:\n" + "\n".join(opt_lines) + "\n"
        + "Summary:\n" + "\n\n".join(summaries)
    )
    return prompt


# === Register all templates in a dictionary ===
TEMPLATE_FUNCS = {
    "triples": template_triples,
    "sentence": template_sentence,
    "summary": template_graph_summary
}

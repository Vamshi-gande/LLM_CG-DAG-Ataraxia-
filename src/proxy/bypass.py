from src.graph.graph import Graph


def should_bypass(
    graph: Graph,
    turn_count: int,
    min_nodes: int = 20,
    min_turns: int = 10,
) -> bool:
    """
    Determine whether to bypass the full graph pipeline.

    Bypass when BOTH conditions are true (AND logic - not OR):
        graph.node_count() < min_nodes
        AND turn_count < min_turns

    During bypass:
        - Full pipeline (embedding -> propagation -> DAG -> context) is skipped
        - Recent context is passed directly to Ollama unchanged
        - Graph updater still runs - graph builds up during bypass
        - This provides a clean A/B baseline: bypass = LLM without middleware

    Returns True if bypass mode is active, False if full pipeline should run.
    """
    return graph.node_count() < min_nodes and turn_count < min_turns
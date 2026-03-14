"""Custom evaluator that checks if the agent routed the query to the correct specialist."""


class RoutingAccuracyEvaluator:
    """Checks whether the agent's query_type classification matches the expected label.

    This is a deterministic (non-LLM) evaluator and requires no model configuration.

    Input parameters (passed via evaluate() column_mapping):
        expected (str): The ground-truth query type from the evaluation dataset.
        actual   (str): The query_type returned by the agent.

    Returns:
        routing_correct (bool): True if classification matches.
        routing_score   (float): 1.0 if correct, 0.0 otherwise.
    """

    id = "routing_accuracy"

    def __call__(self, *, expected: str, actual: str, **kwargs) -> dict:
        correct = expected.strip().lower() == actual.strip().lower()
        return {
            "routing_correct": correct,
            "routing_score": 1.0 if correct else 0.0,
        }

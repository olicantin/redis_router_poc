# from articles import load_dataset
# from customer_baseline import classify_queries_with_stats
# from redis_routing_optimization import route_with_llm_fallback_queries_with_stats


# articles = load_dataset("bbc-news-articles-labeled/BBC News Test.csv")[0:500]

# result_baseline = classify_queries_with_stats(articles)
# result_optimized = route_with_llm_fallback_queries_with_stats(articles)

from articles import load_dataset
from customer_baseline import classify_queries_with_stats
from redis_routing_optimization import route_with_llm_fallback_queries_with_stats

LABELS = ['politics', 'sport', 'tech', 'business', 'entertainment']

def extract_label(prediction_object):
    """Return a lowercase label string from a prediction object."""
    if isinstance(prediction_object, str):
        return prediction_object.strip().lower()
    if isinstance(prediction_object, dict):
        for key in ("label", "category", "pred", "prediction"):
            if key in prediction_object:
                return str(prediction_object[key]).strip().lower()
    return None


def normalize_result(raw_result):
    """Accept (responses, cost, exec_time, accuracy) or (cost, exec_time, accuracy) or dict."""
    if isinstance(raw_result, dict):
        return (
            raw_result.get("responses"),
            raw_result.get("costs"),
            raw_result.get("exec_time"),
            raw_result.get("accuracy"),
        )
    if isinstance(raw_result, (list, tuple)):
        if len(raw_result) == 4:
            return raw_result
        if len(raw_result) == 3:
            return None, raw_result[0], raw_result[1], raw_result[2]
    return None, None, None, None


def compute_total_seconds(execution_time_value):
    """Sum list/tuple of seconds or pass through a single float seconds value."""
    if isinstance(execution_time_value, (list, tuple)):
        return sum(execution_time_value)
    return execution_time_value


def summarize_run(method_name, raw_result):
    responses, total_cost_usd, execution_time_value, accuracy = normalize_result(raw_result)
    total_seconds = compute_total_seconds(execution_time_value) or 0.0

    if accuracy is None and responses is not None:
        predicted_labels = [extract_label(item) for item in responses]
        accuracy = sum(t == p for t, p in zip(LABELS, predicted_labels)) / len(LABELS)

    average_seconds = (total_seconds / len(LABELS)) if LABELS else 0.0

    print(
        f"{method_name:10} | accuracy={accuracy if accuracy is not None else float('nan'):.3f} "
        f"| cost=${(total_cost_usd or 0):.6f} | total_seconds={total_seconds:.3f} "
        f"| average_seconds={average_seconds:.3f}"
    )
    return responses


def main():
    articles = load_dataset("bbc-news-articles-labeled/BBC News Test.csv")[:1000]

    baseline_result = classify_queries_with_stats(articles)
    optimized_result = route_with_llm_fallback_queries_with_stats(articles)

    print("\n== Summary ==")
    baseline_responses = summarize_run("baseline", baseline_result)
    optimized_responses = summarize_run("optimized", optimized_result)

    try:
        from sklearn.metrics import classification_report
        if baseline_responses:
            print(
                "\nBaseline report:\n",
                classification_report(LABELS, [extract_label(x) for x in baseline_responses], digits=3),
            )
        if optimized_responses:
            print(
                "\nOptimized report:\n",
                classification_report(LABELS, [extract_label(x) for x in optimized_responses], digits=3),
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()

from articles import load_dataset
from customer_baseline import classify_queries_with_stats, LABELS
from redis_routing_optimization import route_with_llm_fallback_queries_with_stats
from sklearn.metrics import classification_report, confusion_matrix


def extract_label(prediction_object):
    """Return a lowercase label string from a prediction object."""
    if isinstance(prediction_object, str):
        return prediction_object.strip().lower()

    if isinstance(prediction_object, dict):
        for key in ("label", "category", "pred", "prediction"):
            if key in prediction_object:
                return str(prediction_object[key]).strip().lower()

   # handle (ArticleId, label) tuples
    if isinstance(prediction_object, (tuple, list)) and len(prediction_object) >= 2:
       return str(prediction_object[1]).strip().lower()
   # handle HF chat objects
    try:
       choices = getattr(prediction_object, "choices", None)
       if choices:
           import json as _json
           content = choices[0].message.content
           return (_json.loads(content) or {}).get("label", "").strip().lower()
    except Exception:
        pass
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


def summarize_run(method_name, raw_result, articles):
    responses, total_cost_usd, execution_time_value, accuracy = normalize_result(raw_result)
    # exec_time returned by your functions is already an average per query
    average_seconds = execution_time_value or 0.0

    # recompute accuracy robustly if not provided
    if responses is not None:
        predicted_labels = [extract_label(item) for item in responses]
        labels = [(a.get("Category","") or "").strip().lower() for a in articles]
        # align lengths defensively
        n = min(len(labels), len(predicted_labels))
        if n:
            accuracy = sum(labels[i] == (predicted_labels[i] or "") for i in range(n)) / n
        else:
            accuracy = float("nan")
        f"{method_name:10} | accuracy={(accuracy if accuracy is not None else float('nan')):.3f} "
        f"| cost=${(total_cost_usd or 0):.6f} | avg_seconds={average_seconds:.3f}"
    return responses

def print_summary(baseline_result, optimized_result, articles):
    # Unpacking Results
    baseline_responses, baseline_costs, baseline_exec_time, baseline_accuracy = baseline_result
    optimized_responses, optimized_costs, optimized_exec_time, optimized_accuracy = optimized_result

    # Just making sure we have values for everything
    baseline_costs = baseline_costs or 0.0 # cost/query
    optimized_costs = optimized_costs or 0.0
    baseline_exec_time = baseline_exec_time or 0.0   # avg seconds/query
    optimized_exec_time = optimized_exec_time or 0.0
    baseline_accuracy = baseline_accuracy if baseline_accuracy is not None else float("nan") # good response/total query
    optimized_accuracy = optimized_accuracy if optimized_accuracy is not None else float("nan")

    print("\n== Summary ==")
    print(
        f"Baseline                     | accuracy={baseline_accuracy:.3f} "
        f"| cost=${baseline_costs:.6f} | avg_seconds={baseline_exec_time:.3f}"
    )
    print(
        f"REDIS Semantic Router + LLM  | accuracy={optimized_accuracy:.3f} "
        f"| cost=${optimized_costs:.6f} | avg_seconds={optimized_exec_time:.3f}"
    )

    # Optional: projected cost per 1k items
    n_items = len(articles)
    if n_items:
        for query_nb in [100000,]:
            scale = query_nb / n_items
            print(
                f"Projected cost per ~100k        | baseline=${baseline_costs * scale:.2f} "
                f"| router+LLM=${optimized_costs * scale:.2f}"
            )


def main():
    # Load your labeled test dataset
    articles = load_dataset("bbc-news-articles-labeled/BBC News Train.csv")[:1000] # Limit to 1000

    # Run both pipelines
    baseline_results = classify_queries_with_stats(articles)
    optimized_results = route_with_llm_fallback_queries_with_stats(articles)

    # Print the accuracy, cost and execution time + projected costs
    print_summary(baseline_results, optimized_results, articles)

    # === Detailed classification reports (sklearn) ===
    from sklearn.metrics import classification_report, confusion_matrix
    from collections import Counter

    # Labels
    labels = [a.get("Category", "")for a in articles]

    # Extract labels from results
    pred_baseline_labels = [extract_label(x) for x in baseline_results[0]]
    pred_optimized_labels = [extract_label(x) for x in optimized_results[0]]

    # Align lengths just in case
    n_b = min(len(labels), len(pred_baseline_labels))
    n_o = min(len(labels), len(pred_optimized_labels))
    baseline_reference = labels[:n_b]
    optimized_reference = labels[:n_o]
    baseline_prediction = pred_baseline_labels[:n_b]
    optimized_prediction = pred_optimized_labels[:n_o]

    # Extract predictions
    baseline_prediction  = [(p or "").strip().lower() for p in baseline_prediction]
    optimized_prediction = [(p or "").strip().lower() for p in optimized_prediction]

    # Works only with training data
    print("\n== Baseline ==")
    print(classification_report(baseline_reference, baseline_prediction, labels=LABELS, digits=3, zero_division=0))

    print("\n== REDIS Semantic Router + LLM ==")
    print(classification_report(optimized_reference, optimized_prediction, labels=LABELS, digits=3, zero_division=0))


if __name__ == "__main__":
    main()

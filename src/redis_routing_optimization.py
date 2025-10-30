from articles import load_dataset, load_dataset_per_category
from customer_baseline import format_query, classify, COST_IN, COST_OUT, LABELS

from redisvl.extensions.router import Route, SemanticRouter
from redis_retrieval_optimizer.threshold_optimization import RouterThresholdOptimizer

import json
import os
import random
import time
import warnings


# Redis stuff
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"


def split_dataset_per_category(dataset, ref_frac=0.6, val_frac=0.2):
    """Split raw training dataset into reference/validation/test"""
    refs, val, test = {}, {}, {}

    for category, items in dataset.items():
        items = items[:]
        random.seed(42)
        random.shuffle(items)
        a = int(len(items)*ref_frac)
        b = a + int(len(items)*val_frac)
        refs[category], val[category], test[category] = items[:a], items[a:b], items[b:]
    return refs, val, test


def build_routes(dataset: dict) -> list:
    """Define routes with categories and article"""
    routes = []
    for category, articles in dataset.items():
        route = Route(
            name=category,
            references=[text for _,text in articles],
            distance_threshold=0.5
        )
        routes.append(route)

    return routes

def build_optimizer_data(dataset: dict) -> str:
    """Build training data for optimizer"""
    data = []
    for category, articles in dataset.items():
        for _, article in articles:
         data.append({"query": article, "query_match": category})
    return data


def build_router_and_optimizer():
    """Build the router and optimizer"""
    full_training_dataset = load_dataset_per_category("bbc-news-articles-labeled/BBC News Train.csv")
    reference_dt, validation_dt, test_dt = split_dataset_per_category(full_training_dataset)

    # Initialize the SemanticRouter
    routes = build_routes(reference_dt)
    article_router = SemanticRouter(
        name="article-router",
        routes=routes,
        redis_url="redis://localhost:6379",
        overwrite=False
    )
    training_data = build_optimizer_data(validation_dt)
    optimizer = RouterThresholdOptimizer(article_router, training_data)
    optimizer.optimize()

    return article_router


def route_with_llm_fallback_queries_with_stats(queries: list):
    """Route first; if below threshold, call the baseline LLM."""

    try:
        article_router = build_router_and_optimizer()

        _ = article_router("warmup") if callable(article_router) else None
    except Exception:
        # In case we're running REDIS elsewhere
        full_training_dataset = load_dataset_per_category("bbc-news-articles-labeled/BBC News Train.csv")
        reference_dt, validation_dt, _ = split_dataset_per_category(full_training_dataset)
        routes = build_routes(reference_dt)
        article_router = SemanticRouter(
            name="article-router",
            routes=routes,
            redis_url=f"redis://{os.getenv('REDIS_HOST','localhost')}:{os.getenv('REDIS_PORT','6379')}",
            overwrite=False
        )
        training_data = build_optimizer_data(validation_dt)
        opt = RouterThresholdOptimizer(article_router, training_data)
        opt.optimize()

    costs = 0.0
    times = []
    accurate = 0
    responses = []

    for query in queries:
        start = time.time()

        # 1) Router first
        match = article_router(query["Text"])  # RouteMatch or None
        if match:
           # router handled it → no LLM call/cost
           times.append(time.time() - start)
           pred_label = (getattr(match, "name", "") or "").strip().lower()
           truth = (query.get("Category","") or "").strip().lower()
           responses.append({"ArticleId": query.get("ArticleId"), "label": pred_label})
           accurate += int(pred_label == truth)
           continue

       # 2) Fallback to baseline LLM
        formatted_query = format_query(query["Text"])
        response = classify(formatted_query)
        times.append(time.time() - start)
        pred_label = None
        try:
            if response and getattr(response, "choices", None):
                content = response.choices[0].message.content
                pred_label = (json.loads(content) or {}).get("label")
            elif isinstance(response, dict):
                pred_label = response.get("label")
        except Exception:
            print(f"Bad response: {response}")

        pred_label = (pred_label or "").strip().lower()
        truth = (query.get("Category","") or "").strip().lower()
        responses.append({"ArticleId": query.get("ArticleId"), "label": pred_label})
        accurate += int(pred_label == truth)

        usage = getattr(response, "usage", None)
        if usage:
            costs += (getattr(usage, "prompt_tokens", 0) * COST_IN +
                      getattr(usage, "completion_tokens", 0) * COST_OUT)

    exec_time = (sum(times) / len(times)) if times else 0.0
    accuracy = (accurate / len(queries)) if queries else 0.0
    return responses, costs, exec_time, accuracy


def main() -> int:
    """Main for testing"""
    # Uncomment below for testing
    # test_dataset = load_dataset("bbc-news-articles-labeled/BBC News Train.csv")[:50]
    # results = route_with_llm_fallback_queries_with_stats(test_dataset)
    # print(results)
    return 0

if __name__ == "__main__":
    main()

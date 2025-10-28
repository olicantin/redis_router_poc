import os, json
from huggingface_hub import InferenceClient
from articles import load_dataset
import time

LABELS = ['politics', 'sport', 'tech', 'business', 'entertainment'] + ["unknown"]

HF_TOKEN = os.getenv("HF_TOKEN")
PROVIDER = os.getenv("HF_PROVIDER", "nscale")
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Inference Costs
# $0.09 / 1M input tokens
# $0.29 / 1M output tokens.
COST_IN, COST_OUT = 0.09/1_000_000, 0.29/1_000_000


# Schema for response
SCHEMA = {
  "type": "json_schema",
  "json_schema": {
    "name": "NewsLabel",
    "strict": True,
    "schema": {
      "type": "object",
      "properties": {
        "label": {
          "oneOf": [
            {"type": "string", "enum": LABELS},
            {"type": "null"}
          ]
        }
      },
      "required": ["label"],
      "additionalProperties": False
    }
  }
}

client = InferenceClient(model=MODEL_ID, provider=PROVIDER, token=HF_TOKEN)

def format_query(article: str) -> list:
    return [
        {"role":"system","content":"Classify the article from one label from the enum and return JSON only."},
        {"role":"user","content":"Text:\n" + article}
    ]


def classify(query: str) -> str:
    """Classify 1 article and return"""
    try:
        out = client.chat_completion(messages=query, response_format=SCHEMA,
                                     temperature=0.2, max_tokens=16)
        return out
    except Exception:
        # Fallback if provider doesn’t support structured outputs
        out = client.chat_completion(messages=query, temperature=0.2, max_tokens=16)
        return json.loads(out.choices[0].message["content"])

def classify_queries_with_stats(queries: list) -> (float, float, float): #cost, exec_time, accuracy

    costs = 0.0
    times = []
    accurate = 0
    responses = []

    for query in queries:
        formatted_query = format_query(query["Text"])

        start = time.time()
        response = classify(formatted_query)
        times.append(time.time() - start)

        responses.append(responses)
        try:
            if response and response.choices:
                if json.loads(response.choices[0].message.content)['label'] in LABELS:
                    accurate += 1
        except:
            print(f"Bad response: {response}")

        usage = response.usage
        costs += (getattr(usage, "prompt_tokens", 0) * COST_IN +
                  getattr(usage, "completion_tokens", 0) * COST_OUT)

    exec_time = (sum(times) / len(times)) if times else 0.0
    accuracy = (accurate / len(queries)) if queries else 0.0
    return responses, costs, exec_time, accuracy


def main() -> int:
    """Just the main for testing"""
    articles = load_dataset("bbc-news-articles-labeled/BBC News Test.csv")[:5]
    results =  classify_queries_with_stats(articles)
    print(results)

    return 0


if __name__ == "__main__":
    main()


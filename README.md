# Redis Semantic Router POC

## Basic setup

Run:
```shell
export HF_TOKEN=YOUR_HF_TOKEN
```

Run:
```shell
docker login
docker pull redis:8.0.3
docker run -d --name redis -p 6379:6379 redis/redis:8.0.3
```

Run:
```shell
pip install requirements.txt
```

## Execute Customer Baseline

Uncomment code in the main function and run:
```shell
python customer_baseline.py
```
## Execute Redis Router

Uncomment code in the main function and run
```shell
python redis_routing_optimization.py
```

## Comparative Results

Run:
```shell
python comparison_baseline_vs_router_llm.py
```

### Example Output

After running 100 articles through the baseline, the router + some tests
```shell
== Summary ==
Baseline                     | accuracy=0.780 | cost=$0.004993 | avg_seconds=1.087
REDIS Semantic Router + LLM  | accuracy=0.970 | cost=$0.000000 | avg_seconds=0.042
Projected cost per ~100k        | baseline=$4.99 | router+LLM=$0.00

== Baseline ==
               precision    recall  f1-score   support

     politics      0.565     0.929     0.703        14
        sport      0.964     0.871     0.915        31
         tech      1.000     0.421     0.593        19
     business      0.818     0.783     0.800        23
entertainment      0.632     0.923     0.750        13
      unknown      0.000     0.000     0.000         0

     accuracy                          0.780       100
    macro avg      0.663     0.654     0.627       100
 weighted avg      0.838     0.780     0.776       100


== REDIS Semantic Router + LLM ==
               precision    recall  f1-score   support

     politics      0.933     1.000     0.966        14
        sport      1.000     0.968     0.984        31
         tech      1.000     0.947     0.973        19
     business      0.958     1.000     0.979        23
entertainment      1.000     0.923     0.960        13
      unknown      0.000     0.000     0.000         0

    micro avg      0.980     0.970     0.975       100
    macro avg      0.815     0.806     0.810       100
 weighted avg      0.981     0.970     0.975       100
```

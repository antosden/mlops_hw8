import random
import time

from fastapi import FastAPI
from prometheus_client import Histogram, Counter, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response


app = FastAPI(title="ML Recommendation Service")

REQUEST_COUNT = Counter(
    "request_count_total",
    "Total number of requests",
    ["endpoint"]
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict(slow: bool = False):
    endpoint = "/predict"
    REQUEST_COUNT.labels(endpoint=endpoint).inc()

    with REQUEST_LATENCY.labels(endpoint=endpoint).time():
        if slow:
            # Искусственное замедление для проверки Grafana alert
            time.sleep(2)
        else:
            time.sleep(random.uniform(0.05, 0.3))

        recommendations = [
            {"movie_id": 101, "score": 0.94},
            {"movie_id": 205, "score": 0.88},
            {"movie_id": 317, "score": 0.81}
        ]

        return {
            "user_id": random.randint(1, 1000),
            "recommendations": recommendations
        }


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
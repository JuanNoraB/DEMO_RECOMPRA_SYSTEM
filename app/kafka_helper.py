"""
kafka_helper.py — Helper minimalista para publicar eventos a Kafka.

Uso:
    from kafka_helper import publish
    publish("training.completed", {"precision@3": 0.85, "epochs": 30})

Si Kafka no está disponible, simplemente logea y continúa (no bloquea el pipeline).
"""
from __future__ import annotations

import json
import os
from datetime import datetime


KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "kafka-svc:9092")

_producer = None


def _get_producer():
    global _producer
    if _producer is None:
        from kafka import KafkaProducer
        _producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            request_timeout_ms=5000,
            max_block_ms=5000,
        )
    return _producer


def publish(topic: str, data: dict):
    """Publica un mensaje JSON a un topic de Kafka. No-op si Kafka no está disponible."""
    data["_timestamp"] = datetime.now().isoformat()
    try:
        producer = _get_producer()
        producer.send(topic, value=data)
        producer.flush(timeout=3)
        print(f"[Kafka] → {topic}: {json.dumps(data, default=str)[:200]}")
    except Exception as e:
        print(f"[Kafka] No disponible ({e}). Evento descartado: {topic}")

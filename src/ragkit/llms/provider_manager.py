"""Multi-provider LLM with automatic fallback.

Refactor of the original ``utils/provider_manager.py``. Behavior is
preserved (Groq -> Cerebras -> AWS Bedrock, key rotation, per-model Bedrock
body formatting), but:

- it now implements the :class:`~ragkit.core.interfaces.BaseLLM` interface,
- provider SDKs are imported lazily so the base install stays light,
- model names come from :class:`~ragkit.core.config.LLMConfig`,
- credentials still come from the environment (``.env`` is auto-loaded).
"""

from __future__ import annotations

import json
import os
import time

from dotenv import load_dotenv

from ragkit.core.interfaces import BaseLLM
from ragkit.core.config import LLMConfig
from ragkit.core.registry import LLM_REGISTRY
from ragkit.utils.logger import get_logger

load_dotenv()
_log = get_logger("ragkit.llm")


@LLM_REGISTRY.register("provider_manager")
class ProviderManager(BaseLLM):
    """LLM with Groq -> Cerebras -> Bedrock fallback."""

    BEDROCK_MODELS = [
        "amazon.nova-pro-v1:0",
        "meta.llama3-70b-instruct-v1:0",
        "mistral.mixtral-8x7b-instruct-v0:1",
        "meta.llama3-8b-instruct-v1:0",
        "mistral.mistral-7b-instruct-v0:2",
        "amazon.nova-lite-v1:0",
    ]

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

        self.groq_keys = _split_env("GROQ_API_KEYS")
        self.groq_idx = 0

        self.cerebras_keys = _split_env("CEREBRAS_API_KEYS")
        self.cerebras_idx = 0

        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
        self.aws_region = os.getenv("AWS_REGION", "us-east-1").strip()
        self._bedrock_client = None

    # ── key rotation ──────────────────────────────────────────────────
    def _next_groq(self) -> str:
        key = self.groq_keys[self.groq_idx]
        self.groq_idx = (self.groq_idx + 1) % len(self.groq_keys)
        return key

    def _next_cerebras(self) -> str:
        key = self.cerebras_keys[self.cerebras_idx]
        self.cerebras_idx = (self.cerebras_idx + 1) % len(self.cerebras_keys)
        return key

    def _get_bedrock_client(self):
        if self._bedrock_client is None:
            import boto3
            self._bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.aws_region,
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
            )
        return self._bedrock_client

    # ── providers ─────────────────────────────────────────────────────
    def _try_groq(self, prompt: str, temperature: float) -> str | None:
        if not self.groq_keys:
            return None
        for _ in range(len(self.groq_keys)):
            try:
                from groq import Groq
                client = Groq(api_key=self._next_groq())
                resp = client.chat.completions.create(
                    model=self.config.groq_model,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content
            except Exception as e:  # noqa: BLE001
                _log.warning("Groq failed: %s", str(e)[:120])
        return None

    def _try_cerebras(self, prompt: str) -> str | None:
        if not self.cerebras_keys:
            return None
        for _ in range(len(self.cerebras_keys)):
            try:
                from cerebras.cloud.sdk import Cerebras
                client = Cerebras(api_key=self._next_cerebras())
                resp = client.chat.completions.create(
                    model=self.config.cerebras_model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content
            except Exception as e:  # noqa: BLE001
                _log.warning("Cerebras failed: %s", str(e)[:120])
        return None

    def _try_bedrock(self, prompt: str, temperature: float) -> str | None:
        if not self.aws_access_key or not self.aws_secret_key:
            return None
        try:
            import boto3  # noqa: F401
        except ImportError:
            _log.warning("boto3 not installed; skipping Bedrock.")
            return None

        client = self._get_bedrock_client()
        for model_id in self.BEDROCK_MODELS:
            try:
                result = self._invoke_bedrock(client, model_id, prompt, temperature)
                if result:
                    _log.info("Bedrock OK: %s", model_id)
                    return result
            except Exception as e:  # noqa: BLE001
                _log.warning("Bedrock FAIL %s: %s", model_id, str(e)[:100])
                time.sleep(0.5)
        return None

    def _invoke_bedrock(self, client, model_id, prompt, temperature) -> str | None:
        if "amazon.nova" in model_id:
            body = json.dumps({
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {
                    "max_new_tokens": 1024,
                    "temperature": max(temperature, 1e-6),
                },
            })
            resp = client.invoke_model(modelId=model_id, body=body,
                                       contentType="application/json",
                                       accept="application/json")
            result = json.loads(resp["body"].read())
            return result["output"]["message"]["content"][0]["text"]

        if "meta.llama" in model_id:
            body = json.dumps({
                "prompt": (
                    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
                    f"{prompt}<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>\n"
                ),
                "max_gen_len": 1024,
                "temperature": max(temperature, 1e-6),
            })
            resp = client.invoke_model(modelId=model_id, body=body,
                                       contentType="application/json",
                                       accept="application/json")
            result = json.loads(resp["body"].read())
            return result.get("generation", "")

        if "mistral" in model_id:
            body = json.dumps({
                "prompt": f"<s>[INST]{prompt}[/INST]",
                "max_tokens": 1024,
                "temperature": max(temperature, 1e-6),
            })
            resp = client.invoke_model(modelId=model_id, body=body,
                                       contentType="application/json",
                                       accept="application/json")
            result = json.loads(resp["body"].read())
            return result["outputs"][0]["text"]

        raise ValueError(f"Unknown model family: {model_id}")

    # ── public interface ──────────────────────────────────────────────
    def generate(self, prompt: str, temperature: float | None = None) -> str:
        temp = self.config.temperature if temperature is None else temperature

        for attempt in (
            lambda: self._try_groq(prompt, temp),
            lambda: self._try_cerebras(prompt),
            lambda: self._try_bedrock(prompt, temp),
        ):
            result = attempt()
            if result:
                return result

        raise RuntimeError(
            "All LLM providers exhausted (Groq, Cerebras, AWS Bedrock). "
            "Set GROQ_API_KEYS / CEREBRAS_API_KEYS / AWS_* in your environment "
            "or pass a custom llm to the pipeline."
        )

    def has_credentials(self) -> bool:
        """True if at least one provider is configured."""
        return bool(
            self.groq_keys
            or self.cerebras_keys
            or (self.aws_access_key and self.aws_secret_key)
        )


def _split_env(var: str) -> list[str]:
    return [k.strip() for k in os.getenv(var, "").split(",") if k.strip()]

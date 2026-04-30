"""Benchmark embedding models on MTEB Korean retrieval tasks.

Usage:
    uv run evaluate.py \
        --models nlpai-lab/KURE-v1,BAAI/bge-m3 \
        --tasks LawIRKo,SQuADKorV1Retrieval \
        --gpu 3

Models and tasks are passed as comma-separated lists. Upstage API models are
specified as `upstage/solar-embedding-1-large` and require UPSTAGE_API_KEY in .env.
"""
from __future__ import annotations

import argparse
import logging
import os
from multiprocessing import Pool, current_process
from typing import Any

import numpy as np
import requests
import torch
import torch.multiprocessing as mp
from dotenv import load_dotenv
from setproctitle import setproctitle

import mteb
from mteb import MTEB, get_tasks
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.instruct_wrapper import instruct_wrapper
from mteb.models.model_meta import ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.types import PromptType
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Upstage Solar embedding API encoder
# ---------------------------------------------------------------------------
UPSTAGE_API_URL = "https://api.upstage.ai/v1/solar/embeddings"
UPSTAGE_QUERY_MODEL = "solar-embedding-1-large-query"
UPSTAGE_PASSAGE_MODEL = "solar-embedding-1-large-passage"


class UpstageSolarEncoder(AbsEncoder):
    """Encoder calling Upstage Solar embeddings API.

    Routes by prompt_type: query texts use the `-query` model and document texts
    use the `-passage` model. Embedding dim is 4096, similarity is cosine.
    """

    # Upstage limits per request (per console docs):
    #   - max 100 texts per request
    #   - max 204,800 total tokens per request
    #   - solar-embedding-1-large context window ≈ 4,000 tokens
    # We use char-level proxies for tokens (Korean ≈ 1 char/token,
    # Latin scripts ≈ 4 chars/token) and stay well below limits.
    DEFAULT_MAX_CHARS_PER_TEXT = 8000  # ≈ 4k Korean tokens (safety margin)
    DEFAULT_MAX_CHARS_PER_BATCH = 100_000  # ≈ 100k Korean tokens / 400k Latin
    DEFAULT_MAX_TEXTS_PER_BATCH = 100

    def __init__(
        self,
        model_name: str = "upstage/solar-embedding-1-large",
        revision: str | None = None,
        *,
        device: str | None = None,
        api_key: str | None = None,
        max_texts_per_batch: int = DEFAULT_MAX_TEXTS_PER_BATCH,
        max_chars_per_batch: int = DEFAULT_MAX_CHARS_PER_BATCH,
        max_chars_per_text: int = DEFAULT_MAX_CHARS_PER_TEXT,
        timeout: int = 120,
        max_retries: int = 5,
        **kwargs: Any,
    ) -> None:
        api_key = (
            api_key
            or os.environ.get("UPSTAGE_API_KEY")
            or os.environ.get("UPSTAGE_API")
        )
        if not api_key:
            raise RuntimeError(
                "UPSTAGE_API_KEY (or UPSTAGE_API) is not set. "
                "Add it to .env to evaluate Upstage models."
            )
        self.api_key = api_key
        self.max_texts_per_batch = max_texts_per_batch
        self.max_chars_per_batch = max_chars_per_batch
        self.max_chars_per_text = max_chars_per_text
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites=dict(
                name=model_name,
                revision=revision or "no_revision_available",
                languages=["kor-Hang", "eng-Latn"],
                max_tokens=4000,
                embed_dim=4096,
                framework=["API"],
                similarity_fn_name=ScoringFunction.COSINE,
                use_instructions=False,
                open_weights=False,
                license="not specified",
                reference="https://console.upstage.ai/docs/capabilities/embeddings",
            )
        )

    @staticmethod
    def _safe_text(t: str | None) -> str:
        # Upstage rejects empty inputs; mteb may pass empty docs from corpora.
        return t if t else " "

    def _truncate(self, text: str) -> str:
        text = self._safe_text(text)
        if len(text) > self.max_chars_per_text:
            return text[: self.max_chars_per_text]
        return text

    def _pack_batches(self, texts: list[str]) -> list[list[int]]:
        """Group text indices into batches respecting both count and char-budget limits."""
        batches: list[list[int]] = []
        cur: list[int] = []
        cur_chars = 0
        for i, t in enumerate(texts):
            n = len(t)
            if cur and (
                len(cur) >= self.max_texts_per_batch
                or cur_chars + n > self.max_chars_per_batch
            ):
                batches.append(cur)
                cur, cur_chars = [], 0
            cur.append(i)
            cur_chars += n
        if cur:
            batches.append(cur)
        return batches

    def _post_with_retries(self, payload: dict) -> dict:
        backoff = 1.0
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self.session.post(
                    UPSTAGE_API_URL, json=payload, timeout=self.timeout
                )
                # 4xx other than 429 are not transient -- surface immediately.
                if r.status_code == 400:
                    r.raise_for_status()  # raises HTTPError caller will catch
                r.raise_for_status()
                return r.json()
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                # 400 is a payload error; do not retry, propagate to splitter.
                if status == 400:
                    raise
                last_exc = exc
            except Exception as exc:  # noqa: BLE001 -- network errors etc.
                last_exc = exc
            if attempt == self.max_retries - 1:
                break
            logger.warning(
                f"Upstage API error (attempt {attempt + 1}/{self.max_retries}): "
                f"{last_exc}. Retrying in {backoff:.1f}s..."
            )
            import time

            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
        assert last_exc is not None
        raise last_exc

    def _embed_chunk(self, chunk: list[str], upstream_model: str) -> list[list[float]]:
        """Embed a chunk; on HTTP 400 split-and-retry recursively."""
        try:
            data = self._post_with_retries({"input": chunk, "model": upstream_model})
            embs = sorted(data["data"], key=lambda x: x["index"])
            return [e["embedding"] for e in embs]
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 400:
                if len(chunk) == 1:
                    # Last resort: aggressively truncate the single offender.
                    short = chunk[0][: max(self.max_chars_per_text // 4, 500)]
                    if len(short) < len(chunk[0]):
                        logger.warning(
                            f"[Upstage] 400 on single text len={len(chunk[0])}. "
                            f"Aggressively truncating to len={len(short)} and retrying."
                        )
                        data = self._post_with_retries(
                            {"input": [short], "model": upstream_model}
                        )
                        embs = sorted(data["data"], key=lambda x: x["index"])
                        return [e["embedding"] for e in embs]
                    body = exc.response.text[:300] if exc.response is not None else ""
                    logger.error(
                        f"[Upstage] 400 on minimal text (len={len(chunk[0])}): {body}"
                    )
                    raise
                # Split in half and recurse.
                mid = len(chunk) // 2
                logger.warning(
                    f"[Upstage] 400 on batch of {len(chunk)} texts. Splitting to "
                    f"{mid}+{len(chunk) - mid} and retrying."
                )
                return self._embed_chunk(chunk[:mid], upstream_model) + self._embed_chunk(
                    chunk[mid:], upstream_model
                )
            raise

    def _embed(self, texts: list[str], upstream_model: str) -> np.ndarray:
        truncated = [self._truncate(t) for t in texts]
        batches = self._pack_batches(truncated)
        out: list[list[float]] = [None] * len(truncated)  # type: ignore[list-item]
        for indices in batches:
            chunk = [truncated[i] for i in indices]
            embs = self._embed_chunk(chunk, upstream_model)
            for j, idx in enumerate(indices):
                out[idx] = embs[j]
        return np.asarray(out, dtype=np.float32)

    def encode(
        self,
        inputs,
        *,
        task_metadata=None,
        hf_split=None,
        hf_subset=None,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        upstream_model = (
            UPSTAGE_QUERY_MODEL if prompt_type == PromptType.query else UPSTAGE_PASSAGE_MODEL
        )
        texts: list[str] = []
        for batch in inputs:
            if isinstance(batch, dict) and "text" in batch:
                texts.extend(batch["text"])
            else:
                texts.extend(batch)
        logger.info(
            f"[Upstage] task={getattr(task_metadata, 'name', None)} "
            f"prompt_type={prompt_type} model={upstream_model} n={len(texts)}"
        )
        return self._embed(texts, upstream_model)


# ---------------------------------------------------------------------------
# Per-model loader: rely on each model's built-in prompts when available
# ---------------------------------------------------------------------------
def build_model(model_name: str):
    """Return an MTEB-compatible encoder for the given model id.

    Models with built-in `prompts` (set in their config_sentence_transformers.json)
    are loaded as-is — `model_prompts=None` lets mteb's wrapper pick up those
    built-in prompts automatically. For models that need explicit overrides
    (instruction-tuned LLMs, gemma2 instruct), apply targeted handling.
    """
    name_lower = model_name.lower()

    # upstage API models
    if name_lower.startswith("upstage/"):
        logger.info(f"Loading Upstage API encoder: {model_name}")
        return UpstageSolarEncoder(model_name=model_name)

    # model2vec static embeddings
    if "m2v" in name_lower:
        logger.info(f"Loading model2vec model: {model_name}")
        static_embedding = StaticEmbedding.from_model2vec(model_name)
        st_model = SentenceTransformer(
            modules=[static_embedding],
            model_kwargs={"attn_implementation": "sdpa"},
        )
        return SentenceTransformerEncoderWrapper(model=st_model)

    # gemma2 instruct: needs custom instruct wrapper
    if model_name == "BAAI/bge-multilingual-gemma2":
        logger.info(f"Loading instruct wrapper for: {model_name}")
        instruction_template = "<instruct>{instruction}\n<query>"
        return instruct_wrapper(
            model_name_or_path=model_name,
            mode="embedding",
            instruction_template=instruction_template,
            attn="cccc",
            pooling_method="lasttoken",
            torch_dtype=torch.float16,
            normalized=True,
        )

    # qwen3-embedding / PwC: explicit instruct prefix on queries only
    if "qwen" in name_lower or model_name == "SamilPwC-AXNode-GenAI/PwC-Embedding_expr":
        logger.info(f"Loading instruct-prefix model: {model_name}")
        task_description = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
        model_prompts = {"query": f"Instruct: {task_description}\nQuery:"}
        wrapper = SentenceTransformerEncoderWrapper(
            model=model_name,
            model_prompts=model_prompts,
            model_kwargs={"attn_implementation": "sdpa"},
            trust_remote_code=True,
        )
        if model_name == "SamilPwC-AXNode-GenAI/PwC-Embedding_expr":
            wrapper.model.max_seq_length = 512
        else:
            wrapper.model.max_seq_length = 8192
        return wrapper

    # e5 family (multilingual-e5-*, KoE5, etc.) -- their HF configs ship with
    # empty `prompts`, but the models were trained with `query: ` / `passage: `
    # prefixes. We must inject these explicitly.
    if (
        model_name in {"nlpai-lab/KoE5", "KU-HIAI-ONTHEIT/ontheit-large-v1_1"}
        or "e5" in name_lower
        or "intfloat" in name_lower
    ):
        logger.info(f"Loading e5-family model with query/passage prefixes: {model_name}")
        model_prompts = {"query": "query: ", "document": "passage: "}
        return SentenceTransformerEncoderWrapper(
            model=model_name,
            model_prompts=model_prompts,
            model_kwargs={"attn_implementation": "sdpa"},
        )

    # nomic-embed-v2-moe ships with `search_query:` / `search_document:` prefixes.
    if model_name == "nomic-ai/nomic-embed-text-v2-moe":
        logger.info(f"Loading nomic with search_*: prefixes: {model_name}")
        model_prompts = {"query": "search_query: ", "document": "search_document: "}
        return SentenceTransformerEncoderWrapper(
            model=model_name,
            model_prompts=model_prompts,
            model_kwargs={"attn_implementation": "sdpa"},
            trust_remote_code=True,
        )

    # kanana sentence-transformer (not in HF prompts config)
    if "kanana" in name_lower:
        logger.info(f"Loading kanana model: {model_name}")
        model_prompts = {
            "query": "다음은 사용자의 검색 질문입니다. 질문에 답할 수 있는 문서를 찾아주세요.\n질문:",
        }
        return SentenceTransformerEncoderWrapper(
            model=model_name,
            model_prompts=model_prompts,
            model_kwargs={"attn_implementation": "sdpa"},
            trust_remote_code=True,
        )

    # Default: load the model and let mteb auto-pick built-in prompts.
    logger.info(f"Loading sentence-transformer with built-in prompts: {model_name}")
    return SentenceTransformerEncoderWrapper(
        model=model_name,
        model_kwargs={"attn_implementation": "sdpa"},
        trust_remote_code=True,
    )


def pick_batch_size(model_name: str) -> int:
    name_lower = model_name.lower()
    if name_lower.startswith("upstage/"):
        return 100
    if (
        "multilingual-e5" in name_lower
        or "koe5" in name_lower
        or "ontheit" in name_lower
        or "nomic" in name_lower
        or "pwc" in name_lower
    ):
        return 1200
    if "jina" in name_lower:
        return 8
    if "bge-m3" in name_lower or "snowflake" in name_lower:
        return 32
    if "gemma2" in name_lower:
        return 256
    if "salesforce" in name_lower:
        return 128
    return 64


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def evaluate_model(model_name: str, gpu_id: int, tasks: list[str], output_dir: str, quantize: bool):
    if not model_name.startswith("upstage/"):
        # CUDA_VISIBLE_DEVICES is set by the shell wrapper -> remap to local 0.
        if torch.cuda.device_count() > 0:
            torch.cuda.set_device(0)

    model = build_model(model_name)
    setproctitle(f"{model_name}-{gpu_id}")
    logger.info(
        f"Running tasks={tasks} model={model_name} on GPU {gpu_id} "
        f"in process {current_process().name}"
    )

    task_objs = get_tasks(tasks=tasks, languages=["kor"])
    evaluation = MTEB(tasks=task_objs)
    batch_size = pick_batch_size(model_name)
    logger.info(f"batch_size: {batch_size}")

    encode_kwargs: dict[str, Any] = {"batch_size": batch_size}
    if quantize:
        encode_kwargs["precision"] = "binary"

    out = f"{output_dir}/{model_name}" + ("-quantized" if quantize else "")
    evaluation.run(model, output_folder=out, encode_kwargs=encode_kwargs)


def parse_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run MTEB evaluation on selected models and tasks.")
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated list of model IDs (HF, local paths, or upstage/<name>).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Comma-separated list of MTEB task names.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU number to use.")
    parser.add_argument("--quantize", action="store_true", help="Use binary quantization.")
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output folder for results."
    )
    args = parser.parse_args()

    models = parse_csv(args.models)
    tasks = parse_csv(args.tasks)
    if not models:
        raise SystemExit("No models specified.")
    if not tasks:
        raise SystemExit("No tasks specified.")

    logger.info(f"Models: {models}")
    logger.info(f"Tasks:  {tasks}")
    logger.info(f"GPU:    {args.gpu}")

    mp.set_start_method("spawn", force=True)
    with Pool(processes=1) as pool:
        pool.starmap(
            evaluate_model,
            [(m, args.gpu, tasks, args.output_dir, args.quantize) for m in models],
        )


if __name__ == "__main__":
    main()

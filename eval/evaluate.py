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
from sentence_transformers import SentenceTransformer, models as st_modules
from sentence_transformers.models import StaticEmbedding


# Local kozistr models live under HF-hub cache layout in /data. The two
# `*_v1` / `*_v5` snapshots ship without sentence-transformers configs
# (no modules.json / 1_Pooling), so they need an explicit CLS+Normalize
# head. `ko_embed_v2` is a full sentence-transformers tree (CLS + Normalize)
# and is loaded directly from its snapshot path.
KOZISTR_LOCAL_SNAPSHOTS: dict[str, str] = {
    "kozistr/ko_embed_v1": "/data/models--kozistr--ko_embed_v1/snapshots/03d8a7042464022cf8a81f35f7def1b17f6600ed",
    "kozistr/ko_embed_v2": "/data/models--kozistr--ko_embed_v2/snapshots/b9ed28facafc46caec22d1b5d1178dd5235315d4",
    "kozistr/multi-emb-unsup-v5": "/data/models--kozistr--multi-emb-unsup-v5/snapshots/f01cf0647bd0e69f4a55cdbcf713a58cfd11cdd7",
}
KOZISTR_NEEDS_MANUAL_HEAD = {
    "kozistr/ko_embed_v1",
    "kozistr/multi-emb-unsup-v5",
}

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


# On Blackwell (B300) + cuDNN 9.x with torch 2.11/cu128, the cuDNN SDPA backend
# raises "cuDNN Frontend error: No valid execution plans built" at the attention
# forward, breaking every encode (both KaLM-Gemma3 and nemotron-Llama hit it).
# flash-attn isn't installed, so disable only the cuDNN SDPA backend and let
# torch fall back to its built-in flash / mem-efficient SDPA kernels, which work
# on this GPU. Harmless on non-CUDA / API paths.
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)
    logger.info("Disabled cuDNN SDPA backend (B300/cuDNN9 workaround).")


# ---------------------------------------------------------------------------
# mteb BelebeleRetrieval loader patch
# ---------------------------------------------------------------------------
def _patch_belebele_loader() -> None:
    # mteb's BelebeleRetrieval.load_data calls load_dataset(path, revision)
    # without a config name, but mteb/belebele requires a per-language config
    # (e.g. "kor_Hang"). Replace load_data with one that loads each language
    # config separately and keys self.dataset by language code, matching how
    # the rest of the original method consumes self.dataset[lang_code].
    from datasets import load_dataset
    from mteb.tasks.retrieval.multilingual.belebele_retrieval import (
        BelebeleRetrieval,
    )

    _EVAL_SPLIT = "test"

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        needed_langs: set[str] = set()
        for lang_pair in self.hf_subsets:
            for lang in self.metadata.eval_langs[lang_pair]:
                needed_langs.add(lang.replace("-", "_"))
        self.dataset = {}
        for lang in needed_langs:
            self.dataset[lang] = load_dataset(
                self.metadata.dataset["path"],
                lang,
                revision=self.metadata.dataset["revision"],
                split=_EVAL_SPLIT,
            )
        self.queries = {lp: {_EVAL_SPLIT: {}} for lp in self.hf_subsets}
        self.corpus = {lp: {_EVAL_SPLIT: {}} for lp in self.hf_subsets}
        self.relevant_docs = {lp: {_EVAL_SPLIT: {}} for lp in self.hf_subsets}
        for lang_pair in self.hf_subsets:
            langs = self.metadata.eval_langs[lang_pair]
            lang_corpus = langs[0].replace("-", "_")
            lang_question = langs[1].replace("-", "_")
            ds_corpus = self.dataset[lang_corpus]
            ds_question = self.dataset[lang_question]
            question_ids: dict[str, int] = {}
            for row in ds_question:
                q = row["question"]
                if q not in question_ids:
                    question_ids[q] = len(question_ids)
            link_to_context_id: dict[str, str] = {}
            context_idx = 0
            for row in ds_corpus:
                if row["link"] not in link_to_context_id:
                    cid = f"C{context_idx}"
                    link_to_context_id[row["link"]] = cid
                    self.corpus[lang_pair][_EVAL_SPLIT][cid] = {
                        "title": "",
                        "text": row["flores_passage"],
                    }
                    context_idx += 1
            for row in ds_question:
                qid = f"Q{question_ids[row['question']]}"
                self.queries[lang_pair][_EVAL_SPLIT][qid] = row["question"]
                cid = link_to_context_id[row["link"]]
                self.relevant_docs[lang_pair][_EVAL_SPLIT].setdefault(qid, {})[cid] = 1
        self.data_loaded = True

    BelebeleRetrieval.load_data = load_data
    logger.info("Patched mteb.BelebeleRetrieval.load_data to load per-language configs.")


_patch_belebele_loader()


# ---------------------------------------------------------------------------
# transformers AutoProcessor text-tokenizer fallback patch
# ---------------------------------------------------------------------------
def _patch_autoprocessor_text_fallback() -> None:
    # sentence-transformers v5's Transformer module unconditionally calls
    # AutoProcessor.from_pretrained for every model. For text-only LLM embedders
    # built on multimodal-capable architectures (e.g. KaLM-Embedding-Gemma3,
    # whose config model_type is `gemma3_text`), transformers tries to assemble a
    # multimodal processor and fails because the checkpoint ships no
    # preprocessor_config.json:
    #     OSError: Can't load image processor for '<model>' ...
    # Such models only need a tokenizer. Wrap AutoProcessor.from_pretrained so
    # that, on this failure, it falls back to AutoTokenizer. ST treats a
    # PreTrainedTokenizerBase processor as a plain tokenizer (see Transformer
    # .tokenizer property + tokenize()), which is exactly what a text embedder
    # needs. Models with a valid processor are unaffected — the original call
    # succeeds and the fallback never runs.
    from transformers import AutoProcessor, AutoTokenizer

    _orig_from_pretrained = AutoProcessor.from_pretrained

    _TOKENIZER_KWARG_KEYS = {
        "trust_remote_code", "model_max_length", "padding_side", "revision",
        "use_fast", "token", "cache_dir", "subfolder",
    }

    def from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        try:
            return _orig_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        except (OSError, ValueError) as exc:
            tok_kwargs = {k: v for k, v in kwargs.items() if k in _TOKENIZER_KWARG_KEYS}
            logger.warning(
                f"AutoProcessor failed for {pretrained_model_name_or_path} "
                f"({type(exc).__name__}: {exc}); falling back to AutoTokenizer."
            )
            return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **tok_kwargs)

    AutoProcessor.from_pretrained = staticmethod(from_pretrained)
    logger.info("Patched transformers.AutoProcessor.from_pretrained with text-tokenizer fallback.")


_patch_autoprocessor_text_fallback()


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
    # Korean text is roughly 1.5 tokens per character (worst case for syllable
    # heavy Hangul). We pick conservative char budgets so a single text never
    # bumps the 4k-token wall and a packed batch stays under ~150k tokens.
    # Push to the API's per-text cap (4000 tokens). 4000 chars covers most
    # Korean text under that limit; rare-char outliers fall to the 400 handler
    # which progressively shrinks 1/2 -> 1/4 -> ... and retries.
    DEFAULT_MAX_CHARS_PER_TEXT = 4000
    # Larger per-request budget reduces total request count for long-doc tasks
    # like MLDR. Stays well under the API's 204,800-token request cap
    # (~136K Korean chars worst case).
    DEFAULT_MAX_CHARS_PER_BATCH = 80_000
    DEFAULT_MAX_TEXTS_PER_BATCH = 20  # reduced from 50 to ease 429s on MLDR
    # Baseline throttle to keep us under the per-key request rate limit so we
    # don't spend every second eating a 429+1s retry.
    DEFAULT_BASELINE_DELAY_SEC = 2.0

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
        baseline_delay_sec: float = DEFAULT_BASELINE_DELAY_SEC,
        timeout: int = 120,
        max_retries: int = 10,
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
        self.baseline_delay_sec = baseline_delay_sec
        self.timeout = timeout
        self.max_retries = max_retries
        # Track the last request time to enforce baseline_delay_sec between calls.
        import time as _time

        self._time = _time
        self._last_request_at = 0.0
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

    def _throttle(self) -> None:
        """Sleep so consecutive requests stay >= baseline_delay_sec apart."""
        now = self._time.monotonic()
        elapsed = now - self._last_request_at
        if elapsed < self.baseline_delay_sec:
            self._time.sleep(self.baseline_delay_sec - elapsed)
        self._last_request_at = self._time.monotonic()

    def _post_with_retries(self, payload: dict) -> dict:
        backoff = 2.0
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                r = self.session.post(
                    UPSTAGE_API_URL, json=payload, timeout=self.timeout
                )
                if r.status_code == 400:
                    r.raise_for_status()  # propagate to splitter
                r.raise_for_status()
                return r.json()
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status == 400:
                    raise
                # Use Retry-After header when present (RFC 7231).
                if status == 429 and exc.response is not None:
                    ra = exc.response.headers.get("Retry-After")
                    if ra:
                        try:
                            backoff = max(backoff, float(ra))
                        except ValueError:
                            pass
                last_exc = exc
            except Exception as exc:  # noqa: BLE001 -- network errors etc.
                last_exc = exc
            if attempt == self.max_retries - 1:
                break
            logger.warning(
                f"Upstage API error (attempt {attempt + 1}/{self.max_retries}): "
                f"{last_exc}. Retrying in {backoff:.1f}s..."
            )
            self._time.sleep(backoff)
            backoff = min(backoff * 1.5, 60)
        assert last_exc is not None
        raise last_exc

    EMBED_DIM = 4096

    def _zero_vec(self) -> list[float]:
        return [0.0] * self.EMBED_DIM

    def _embed_chunk(self, chunk: list[str], upstream_model: str) -> list[list[float]]:
        """Embed a chunk; on HTTP 400 split-and-retry recursively. Single-text
        failures fall back to a zero vector so the eval never aborts."""
        try:
            data = self._post_with_retries({"input": chunk, "model": upstream_model})
            embs = sorted(data["data"], key=lambda x: x["index"])
            return [e["embedding"] for e in embs]
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 400:
                if len(chunk) == 1:
                    # Progressive shrink: 1/2 -> 1/4 -> 1/8 -> 1/16 of current text.
                    text = chunk[0]
                    for divisor in (2, 4, 8, 16):
                        short_len = max(len(text) // divisor, 200)
                        if short_len >= len(text):
                            continue
                        short = text[:short_len]
                        try:
                            data = self._post_with_retries(
                                {"input": [short], "model": upstream_model}
                            )
                            embs = sorted(data["data"], key=lambda x: x["index"])
                            logger.warning(
                                f"[Upstage] 400 recovered by truncating "
                                f"{len(text)}→{len(short)} chars."
                            )
                            return [e["embedding"] for e in embs]
                        except requests.HTTPError as exc2:
                            if exc2.response is None or exc2.response.status_code != 400:
                                raise
                            continue
                    body = (
                        exc.response.text[:300] if exc.response is not None else ""
                    )
                    logger.error(
                        f"[Upstage] 400 unrecoverable for single text "
                        f"len={len(text)}; substituting zero vector. body={body}"
                    )
                    return [self._zero_vec()]
                # Split in half and recurse.
                mid = len(chunk) // 2
                logger.warning(
                    f"[Upstage] 400 on batch of {len(chunk)} texts. Splitting to "
                    f"{mid}+{len(chunk) - mid} and retrying."
                )
                return self._embed_chunk(chunk[:mid], upstream_model) + self._embed_chunk(
                    chunk[mid:], upstream_model
                )
            # Non-400 HTTPError after retries exhausted: substitute zero vectors
            # for the whole chunk so the eval can finish; log for diagnosis.
            logger.error(
                f"[Upstage] HTTPError after retries for chunk size={len(chunk)}: "
                f"{exc}; substituting zero vectors."
            )
            return [self._zero_vec() for _ in chunk]
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"[Upstage] Unexpected error for chunk size={len(chunk)}: {exc}; "
                f"substituting zero vectors."
            )
            return [self._zero_vec() for _ in chunk]

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

    # Local kozistr snapshots in /data (HF hub cache layout). v2 ships a full
    # sentence-transformers tree; v1/v5 are bare HF checkpoints and need a
    # manually-attached CLS + Normalize head (the trained pooling for both
    # KoSimCSE-style RoBERTa and bge-m3 backbones).
    if model_name in KOZISTR_LOCAL_SNAPSHOTS:
        snapshot = KOZISTR_LOCAL_SNAPSHOTS[model_name]
        if model_name in KOZISTR_NEEDS_MANUAL_HEAD:
            logger.info(
                f"Loading kozistr model with explicit CLS+Normalize head: "
                f"{model_name} (snapshot={snapshot})"
            )
            transformer = st_modules.Transformer(
                snapshot,
                model_args={"attn_implementation": "sdpa"},
            )
            pooling = st_modules.Pooling(
                transformer.get_word_embedding_dimension(),
                pooling_mode="cls",
            )
            normalize = st_modules.Normalize()
            st_model = SentenceTransformer(modules=[transformer, pooling, normalize])
            return SentenceTransformerEncoderWrapper(model=st_model)

        logger.info(f"Loading kozistr sentence-transformers model: {model_name} (snapshot={snapshot})")
        return SentenceTransformerEncoderWrapper(
            model=snapshot,
            model_kwargs={"attn_implementation": "sdpa"},
        )

    # upstage API models
    if name_lower.startswith("upstage/"):
        logger.info(f"Loading Upstage API encoder: {model_name}")
        return UpstageSolarEncoder(model_name=model_name)

    # KaLM-Embedding (Gemma3 backbone): last-token-pooled LLM embedder with an
    # instruct prefix on queries only. Its config ships empty `prompts`, so we
    # inject KaLM's documented default retrieval prompt (queries) and leave
    # documents unprefixed. Built-in lasttoken pooling + Normalize come from its
    # sentence-transformers tree. 12B fp32 is heavy -> load bf16; the model's
    # default max_seq_length is 131072, so we cap it to a retrieval-sane 8192.
    if "kalm" in name_lower:
        logger.info(f"Loading KaLM instruct-prefix model: {model_name}")
        model_prompts = {
            "query": "Instruct: Given a query, retrieve documents that answer the query \nQuery: ",
            "document": "",
        }
        wrapper = SentenceTransformerEncoderWrapper(
            model=model_name,
            model_prompts=model_prompts,
            model_kwargs={"attn_implementation": "sdpa", "torch_dtype": torch.bfloat16},
            trust_remote_code=True,
        )
        # README example uses 512, but we evaluate at 8192 to match nemotron and
        # avoid truncating long-document tasks (e.g. MultiLongDocRetrieval).
        wrapper.model.max_seq_length = 8192
        return wrapper

    # nvidia llama-embed-nemotron-8b: Llama bidirectional embedder (custom remote
    # code). Ships built-in query/document prompts in config_sentence_transformers
    # .json; we pass them explicitly to be safe. Mean pooling + Normalize from its
    # ST tree. Load bf16 and cap the 131072 default max_seq_length to 8192.
    if "nemotron" in name_lower:
        logger.info(f"Loading nemotron embed model: {model_name}")
        model_prompts = {
            "query": "Instruct: Given a question, retrieve passages that answer the question\nQuery: ",
            "document": "",
        }
        wrapper = SentenceTransformerEncoderWrapper(
            model=model_name,
            model_prompts=model_prompts,
            model_kwargs={"attn_implementation": "sdpa", "torch_dtype": torch.bfloat16},
            tokenizer_kwargs={"padding_side": "left"},  # per nemotron README
            trust_remote_code=True,
        )
        # nemotron README leaves max_seq_length at the model default (131072),
        # which is impractical for benchmark corpora; cap at 8192 (covers the
        # Korean retrieval passages comfortably).
        wrapper.model.max_seq_length = 8192
        return wrapper

    # codefuse F2LLM (Qwen3 backbone): last-token-pooled LLM embedder. Native
    # Qwen3 architecture (no remote code). Ships built-in query/document prompts
    # in config_sentence_transformers.json; we pass them explicitly. lasttoken
    # pooling + Normalize come from its ST tree. bf16, cap seq at 8192 to match
    # the other LLM embedders.
    if "f2llm" in name_lower:
        logger.info(f"Loading F2LLM model: {model_name}")
        model_prompts = {
            "query": "Instruct: Given a question, retrieve passages that can help answer the question.\nQuery: ",
            "document": "",
        }
        wrapper = SentenceTransformerEncoderWrapper(
            model=model_name,
            model_prompts=model_prompts,
            model_kwargs={"attn_implementation": "sdpa", "torch_dtype": torch.bfloat16},
        )
        wrapper.model.max_seq_length = 8192
        return wrapper

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

    # Qwen3-VL-Embedding (multimodal generative backbone) used TEXT-ONLY for
    # retrieval. Loads via its sentence-transformers tree + custom qwen3_vl module
    # (trust_remote_code). Per the model card the instruction is applied as a
    # system prompt and is customizable per call; for retrieval we set a retrieval
    # instruction on queries and leave documents on the model's default
    # ("Represent the user's input."). Must precede the generic "qwen" branch.
    if "qwen3-vl" in name_lower:
        logger.info(f"Loading Qwen3-VL-Embedding (text-only) model: {model_name}")
        model_prompts = {
            "query": "Retrieve relevant documents for the query.",
            "document": "Represent the user's input.",
        }
        wrapper = SentenceTransformerEncoderWrapper(
            model=model_name,
            model_prompts=model_prompts,
            model_kwargs={"attn_implementation": "sdpa", "torch_dtype": torch.bfloat16},
            trust_remote_code=True,
        )
        wrapper.model.max_seq_length = 8192
        return wrapper

    # qwen3-embedding / PwC: explicit instruct prefix on queries only
    if "qwen" in name_lower or model_name == "SamilPwC-AXNode-GenAI/PwC-Embedding_expr":
        logger.info(f"Loading instruct-prefix model: {model_name}")
        task_description = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
        model_prompts = {"query": f"Instruct: {task_description}\nQuery:"}
        model_kwargs: dict[str, Any] = {"attn_implementation": "sdpa"}
        # Qwen3-Embedding-8B: 32GB fp32 weights won't fit comfortably alongside
        # 8k-seq activations on an 80GB GPU. Load in bf16.
        if "qwen3-embedding-8b" in name_lower:
            model_kwargs["torch_dtype"] = torch.bfloat16
        wrapper = SentenceTransformerEncoderWrapper(
            model=model_name,
            model_prompts=model_prompts,
            model_kwargs=model_kwargs,
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
    if "kalm" in name_lower:
        # 12B in bf16 with 8k-seq last-token pooling. Small batch keeps long-doc
        # tasks (MLDR/MIRACL) within memory.
        return 4
    if "nemotron" in name_lower:
        # 8B bidirectional in bf16 at 8k-seq.
        return 8
    if "f2llm" in name_lower:
        # 8B Qwen3 last-token embedder in bf16 at 8k-seq.
        return 8
    if "qwen3-embedding-8b" in name_lower:
        # 8B in bf16 leaves ~64GB for activations; batch=2 keeps O(seq^2) attention
        # plus 8k-seq activations under that budget on an 80GB GPU.
        return 2
    if "qwen" in name_lower:
        # Qwen3-Embedding loads with max_seq_length=8192; long-doc tasks (e.g. MLDR)
        # OOM at batch=64. 8 keeps activation memory comfortably under 80GB.
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
    if model_name.startswith("upstage/"):
        # API-based encoder: hide GPUs entirely so mteb's similarity ops use CPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # Re-init torch CUDA visibility (safe even if torch already imported).
    else:
        # CUDA_VISIBLE_DEVICES is set by the shell wrapper -> remap to local 0.
        if torch.cuda.device_count() > 0:
            torch.cuda.set_device(0)

    model = build_model(model_name)
    setproctitle(f"{model_name}-{gpu_id}")
    logger.info(
        f"Running tasks={tasks} model={model_name} on GPU {gpu_id} "
        f"in process {current_process().name}"
    )

    batch_size = pick_batch_size(model_name)
    logger.info(f"batch_size: {batch_size}")

    encode_kwargs: dict[str, Any] = {"batch_size": batch_size}
    if quantize:
        encode_kwargs["precision"] = "binary"

    out = f"{output_dir}/{model_name}" + ("-quantized" if quantize else "")

    # Run tasks one by one so a single task failure (mteb metadata bug, missing
    # config, dataset gone, etc.) does not abort the whole eval.
    for task_name in tasks:
        try:
            task_objs = get_tasks(tasks=[task_name], languages=["kor"])
        except Exception as e:  # noqa: BLE001
            logger.error(f"[task] {task_name}: get_tasks failed: {e}")
            continue
        if not task_objs:
            logger.error(f"[task] {task_name}: no matching task object — skipping.")
            continue
        try:
            evaluation = MTEB(tasks=task_objs)
            evaluation.run(
                model,
                output_folder=out,
                encode_kwargs=encode_kwargs,
                raise_error=False,
            )
            logger.info(f"[task] {task_name}: done.")
        except Exception as e:  # noqa: BLE001
            import traceback as _tb

            logger.error(f"[task] {task_name}: failed and skipped: {e}")
            _tb.print_exc()
            continue


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

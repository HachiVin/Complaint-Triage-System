from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from complaint_triage.baseline import BaselineModel  # noqa: E402
from complaint_triage.data import load_jsonl  # noqa: E402

DEMO = ROOT / "data" / "sample" / "demo_reviews.jsonl"
HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
TEST_DSN = os.getenv("TEST_DATABASE_URL")


@pytest.fixture(scope="session")
def records():
    return load_jsonl(DEMO)


@pytest.fixture(scope="session")
def baseline(records):
    return BaselineModel.train(records, version="test")


class ScriptedChatModel(BaseChatModel):
    """Returns pre-written AIMessages in order, so the agent graph runs without an LLM."""

    responses: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # A fresh copy each time: LangGraph merges messages by id, so reusing one object would overwrite it.
        msg = self.responses[min(self.calls, len(self.responses) - 1)].model_copy(update={"id": None})
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def scripted_llm():
    return ScriptedChatModel


@pytest.fixture(scope="session")
def database():
    """A real PostgreSQL database, wiped and loaded with the demo reviews. Skipped without TEST_DATABASE_URL."""
    if not TEST_DSN:
        pytest.skip("TEST_DATABASE_URL not set")
    from complaint_triage.db import Database

    db = Database(TEST_DSN)
    with db.pool.connection() as conn:
        conn.execute("DROP VIEW IF EXISTS v_reviews; DROP TABLE IF EXISTS corrections, predictions, labels, "
                     "agent_log, reviews CASCADE")
    db.apply_schema()
    yield db
    db.close()


@pytest.fixture(scope="session")
def loaded_db(database, records, baseline):
    from complaint_triage.analyzer import Analyzer

    ids = database.upsert_reviews(records)
    database.save_predictions(ids, Analyzer(baseline).analyze_batch([r.text for r in records]))
    return database


@pytest.fixture(scope="session")
def tiny_bert(tmp_path_factory, records):
    """A tiny randomly initialised BERT with a WordPiece vocab trained on the demo texts. Lets the training
    loop run offline in seconds; the numbers it produces mean nothing."""
    if not HAS_TORCH:
        pytest.skip("torch/transformers not installed")
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    out = tmp_path_factory.mktemp("tiny-bert")
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size=500, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tok.train_from_iterator([r.text for r in records], trainer)
    cls_id, sep_id = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]", special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                   sep_token="[SEP]", mask_token="[MASK]")
    fast.save_pretrained(out)
    config = BertConfig(vocab_size=fast.vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=256)
    BertModel(config).save_pretrained(out)
    return out

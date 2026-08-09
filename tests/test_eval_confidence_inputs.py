import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval_confidence_inputs import infer_link_source, derive_confidence_inputs


def test_infer_link_source_exact_when_full_qualified_name_backticked():
    doc = "`Config.__init__` now requires `api_key`."
    assert infer_link_source("Config.__init__", doc) == "exact"


def test_infer_link_source_leaf_when_only_bare_name_backticked():
    doc = "`close()` will wait for in-flight requests to finish."
    assert infer_link_source("ClientSession.close", doc) == "leaf"


def test_infer_link_source_embedding_when_no_backtick_mention_at_all():
    # The name only appears inside a fenced code block, which
    # doc_parser._extract_mentions does not scan at all (only single
    # backtick inline spans count as a "mention").
    doc = "```python\nConfig(host, api_key=...)\n```"
    assert infer_link_source("Config.__init__", doc) == "embedding"


def test_infer_link_source_embedding_for_plain_prose_without_backticks():
    doc = "GZipMiddleware now supports an extra option for compression."
    assert infer_link_source("GZipMiddleware.__init__", doc) == "embedding"


def test_infer_link_source_top_level_function_leaf_equals_qualified():
    # A top-level function has no class prefix, so its own qualified name
    # IS its leaf name -- a backtick mention of it should count as exact,
    # not merely leaf.
    doc = "`send_email` now accepts a `retries` argument."
    assert infer_link_source("send_email", doc) == "exact"


def test_derive_confidence_inputs_combines_both_fields():
    old = "def fetch(url, retries=3):\n    pass\n"
    new = "def fetch(url, retries=5):\n    pass\n"
    doc = "`fetch` retries up to `retries` times."
    result = derive_confidence_inputs(old, new, doc)
    assert result == {"change_type": "signature", "link_source": "exact"}

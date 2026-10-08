import json
from dataclasses import replace

from localrok.config import PathsConfig, load_config
from localrok.tools import call_tool
from tests.helpers import tmp_conn


def test_call_tool_logs_every_call_and_never_raises(tmp_path):
    tmp_conn(tmp_path).close()
    cfg = replace(load_config(), paths=PathsConfig(source=tmp_path, derived=tmp_path))

    hit = call_tool(cfg, "docs_search", query="J5")
    bad = call_tool(cfg, "docs_frobnicate", query="J5")

    assert "J5 provides the MAINTPORT" in hit
    assert bad.startswith("docs_frobnicate failed:")
    records = [json.loads(line) for line in cfg.paths.tool_log.read_text().splitlines()]
    assert [r["tool"] for r in records] == ["docs_search", "docs_frobnicate"]
    assert records[0]["args"] == {"query": "J5"}
    assert records[0]["result"] == hit
    assert records[0]["chars"] == len(hit)

import json
from types import SimpleNamespace

import pytest

from eval.gold import first_gold_rank, matches_gold, parse_hits, parse_read_labels
from eval.run import _gold_scores, _parse_pi_json, _tool_log_since, main, run_question
from localrok.config import ReadConfig, SearchConfig, load_config
from localrok.read import read
from localrok.search import search
from tests.helpers import tmp_conn


def test_parse_hits_reads_search_output(tmp_path):
    conn = tmp_conn(tmp_path)
    hits = parse_hits(search(conn, "watchdog", SearchConfig()))
    assert hits and all(h.filename.endswith(".pdf") for h in hits)
    gold = [{"doc": "operations-manual", "section": "8.1 Watchdog Recovery", "page": 12}]
    assert first_gold_rank(hits, gold) is not None
    wrong_page = [{"doc": "operations-manual", "section": "8.1 Watchdog Recovery", "page": 99}]
    assert first_gold_rank(hits, wrong_page) is None
    conn.close()


def test_parse_read_labels_skips_error_blocks(tmp_path):
    conn = tmp_conn(tmp_path)
    out = read(
        conn,
        [
            "ops > 8.1 Watchdog Recovery",
            "no-such.pdf > 1 Intro",
            "device-manual-v4.0.0.pdf > p21-22",
        ],
        ReadConfig(),
    )
    labels = parse_read_labels(out)
    assert [h.filename for h in labels] == [
        "operations-manual.pdf",
        "device-manual-v4.0.0.pdf",
    ]
    assert matches_gold(labels[0], {"doc": "operations", "section": "Watchdog Recovery"})
    # A page-range read only proves a gold location when the gold pins a page.
    assert matches_gold(labels[1], {"doc": "device", "section": "7 Electrical", "page": 21})
    assert not matches_gold(labels[1], {"doc": "device", "section": "7 Electrical"})
    conn.close()


def test_parse_pi_json_events():
    events = [
        {"type": "agent_start"},
        {
            "type": "tool_execution_start",
            "toolCallId": "a",
            "toolName": "mcp__localrok__docs_search",
            "args": {"query": "J5"},
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "a",
            "toolName": "mcp__localrok__docs_search",
            "result": {"content": [{"type": "text", "text": "[1] f.pdf | 4.3 X | p3 | fragment"}]},
            "isError": False,
        },
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "toolCall", "name": "mcp__localrok__docs_search", "arguments": {}}
                ],
                "usage": {"input": 100, "output": 10, "cacheRead": 50},
            },
        },
        {"type": "message_end", "message": {"role": "toolResult", "content": []}},
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "..."},
                    {"type": "text", "text": "Answer [1]"},
                ],
                "usage": {"input": 20, "output": 5, "cacheRead": 150},
            },
        },
    ]
    parsed = _parse_pi_json("\n".join(json.dumps(e) for e in events))
    assert parsed["answer"] == "Answer [1]"
    assert [c["name"] for c in parsed["tool_calls"]] == ["mcp__localrok__docs_search"]
    assert parsed["tool_calls"][0]["args"] == {"query": "J5"}
    assert parsed["tool_calls"][0]["result"].startswith("[1] f.pdf")
    assert (parsed["tokens_in"], parsed["tokens_cached"], parsed["tokens_out"]) == (120, 200, 15)


def test_jsonl_preserves_unicode_line_separators():
    text = "Policy\u2028with conditions\u2029and exceptions"
    event = {
        "type": "message_end",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }
    assert _parse_pi_json(json.dumps(event, ensure_ascii=False))["answer"] == text


def test_server_log_preserves_unicode_line_separators(tmp_path):
    record = {"tool": "docs_read", "result": "Policy\u2028with conditions\u2029and exceptions"}
    path = tmp_path / "tool_calls.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
    assert _tool_log_since(path, 0) == [record]


def test_native_mcp_names_score_only_localrok_evidence(tmp_path):
    conn = tmp_conn(tmp_path)
    payload = search(conn, "watchdog", SearchConfig())
    conn.close()
    gold = [{"doc": "operations-manual", "section": "8.1 Watchdog Recovery", "page": 12}]
    calls = [{"name": "mcp__localrok__docs_search", "result": payload}]
    assert _gold_scores(calls, gold)["gold_seen"] is True
    calls[0]["name"] = "mcp__other__docs_search"
    assert _gold_scores(calls, gold)["gold_seen"] is False


@pytest.mark.parametrize("baseline", [False, True])
def test_evaluation_tool_and_extension_isolation(tmp_path, monkeypatch, baseline):
    (tmp_path / "config.toml").write_text("")
    cfg = load_config(root=tmp_path)
    tools = cfg.agent.baseline_tools if baseline else cfg.agent.tools
    captured = []

    def invoke(cmd, **kwargs):
        captured.append(cmd)
        events = [
            {"type": "tool_execution_start", "toolCallId": "a", "toolName": tools[0], "args": {}},
            {
                "type": "tool_execution_end",
                "toolCallId": "a",
                "result": {"content": [{"type": "text", "text": "Evidence"}]},
                "isError": False,
            },
            {
                "type": "message_end",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Answer"}]},
            },
        ]
        return SimpleNamespace(stdout="\n".join(map(json.dumps, events)), stderr="", returncode=0)

    monkeypatch.setattr("eval.run.subprocess.run", invoke)
    question = {"id": "Q1", "text": "Question"}
    result = run_question(cfg, question, tools, baseline=baseline)
    cmd = captured[0]
    assert cmd[cmd.index("--tools") + 1] == ",".join(tools)
    assert "--approve" in cmd and "--offline" in cmd and "--no-extensions" in cmd
    assert "--no-context-files" in cmd
    if baseline:
        assert "--no-mcp" in cmd and "--extension" not in cmd
    else:
        assert cmd[cmd.index("--extension") + 1] == "builtin:mcp"
        assert "--no-mcp" not in cmd
    assert result["tool_calls"][0]["name"] == tools[0]
    assert result["unexpected_tools"] == []

    result = run_question(cfg, question, [], baseline=baseline)
    assert result["unexpected_tools"] == [tools[0]]


def test_pi_failure_aborts_instead_of_reporting_empty_scores(tmp_path, monkeypatch, capsys):
    (tmp_path / "config.toml").write_text("")
    questions = tmp_path / "questions.toml"
    questions.write_text('[[question]]\nid = "Q1"\ntext = "Question"\n')
    monkeypatch.setattr("eval.run.ROOT", tmp_path)
    monkeypatch.setattr(
        "eval.run.run_question",
        lambda *args, **kwargs: {"returncode": 1, "stderr": "Ollama provider missing"},
    )
    assert main(["--skip-preflight", "--questions", str(questions)]) == 2
    assert "Ollama provider missing" in capsys.readouterr().err
    result = next((tmp_path / "eval/results").glob("*/results.json"))
    assert json.loads(result.read_text())["aborted"]["returncode"] == 1

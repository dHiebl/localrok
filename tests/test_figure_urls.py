"""Figure paths carry the source filename, so they contain spaces. Markdown
ends a bare link destination at the first space, so the tools must emit a
finished URL rather than the stored path."""

from __future__ import annotations

from localrok.figures import figure_alt, figure_url

DIR = "DOC-MANUAL-001 V2.0.1-rc6 Device Manual"


def test_spaces_are_encoded_and_the_url_is_root_relative():
    assert figure_url(f"figures/{DIR}/fig-12-14.png") == (
        "/figures/DOC-MANUAL-001%20V2.0.1-rc6%20Device%20Manual/fig-12-14.png"
    )


def test_ampersand_is_encoded():
    url = figure_url("figures/R&D Procedure v1.0 - Example Acceptance Procedure/fig-1-0.png")
    assert url.startswith("/figures/R%26D%20Procedure%20v1.0%20-%20Example%20")
    assert " " not in url and "&" not in url


def test_separators_survive_and_are_not_doubled():
    assert figure_url("/figures/a/b.png") == "/figures/a/b.png"
    assert figure_url(None) is None
    assert figure_url("") is None


def test_alt_text_cannot_close_the_markdown_label_early():
    assert figure_alt("Figure 3: rate [deg/s] over time") == "Figure 3: rate (deg/s) over time"
    assert figure_alt(None) == "figure"

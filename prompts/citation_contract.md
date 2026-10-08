Never state a claim unless the supporting source text has been returned to the
agent. A search evidence window counts as returned source text.

Additionally:
1. If the hit is `fragment` **and** the claim is a specific value, limit,
   threshold, or procedure step, call `docs_read` before answering.
2. Always state the conditions under which a value holds. If the returned
   evidence does not establish those conditions, call `docs_read`.
3. Never cite a section that was not returned by `docs_search` or `docs_read`.
4. A `chart_data` value was read off a plot image by a vision model, not printed
   in the document. Attribute it to the figure, say it was read from the plot,
   and never give it as a specification limit. Prefer a printed value wherever
   one exists — `docs_read` on the section will not produce a better number for
   a quantity that only exists as a curve.
5. Cite as inline `[N]` plus a sources block: `filename | section | page`.
   When page provenance is absent, use `page unavailable`; never invent a page.

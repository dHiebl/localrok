# Document retrieval approach

Treat a document collection as a searchable workspace. Let the assistant inspect
its map, search for promising evidence, read coherent sections, and answer with
verifiable citations. This applies to handbooks, reports, policies, manuals,
notes, presentations, and other documents.

Start with lexical full-text search. Preserve structure, tables, figures, and
source provenance. Keep indexing units small enough for useful ranking and read
units large enough to establish context. Missing evidence is a valid outcome.

Documents remain outside the model context until a tool returns evidence. Add
more complex retrieval only when evaluation of the actual collection establishes
a need. The agent and client should not assume a particular organization,
industry, language, or document naming convention.

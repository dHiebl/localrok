/**
 * Custom AgentTools that call the FastAPI backend REST endpoints,
 * wrapping the existing search/read/outline MCP server functions.
 */

import { Type } from "@earendil-works/pi-ai";
import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";

const API = "/api";

async function post<T>(path: string, body: Record<string, unknown>): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API ${path} failed: ${res.status}`);
  return res.json() as Promise<T>;
}

function textResult(text: string): AgentToolResult<void> {
  return { content: [{ type: "text", text }], details: undefined };
}

export const docsSearchTool: AgentTool = {
  name: "docs_search",
  label: "Search",
  description:
    "Keyword search over the document collection (BM25). " +
    "Returns ranked evidence windows with filename, section_path, page, " +
    "completeness (complete_section or fragment), and any figures in the " +
    "section. Prefer identifiers and exact terms first. A complete_section " +
    "window is citable; a fragment plus a specific value/limit/threshold/" +
    "procedure step requires docs_read.",
  parameters: Type.Object({
    query: Type.String({
      description: "Search terms. Identifiers like J5 are matched as phrases.",
    }),
    doc: Type.Optional(
      Type.String({ description: "Optional filename or unambiguous substring to restrict search." }),
    ),
    section: Type.Optional(
      Type.String({ description: "Optional section_path prefix to restrict search." }),
    ),
    exact: Type.Optional(Type.Boolean({ description: "If true, match the whole query as a phrase." })),
    limit: Type.Optional(Type.Number({ description: "Max hits to return (1–20, default 8)." })),
  }),
  async execute(_id: string, params: any) {
    const { text } = await post<{ text: string }>("/docs_search", params);
    return textResult(text);
  },
};

export const docsReadTool: AgentTool = {
  name: "docs_read",
  label: "Read",
  description:
    "Read one or more sections, page ranges, or whole documents in a single " +
    "call. Batch cross-document comparisons here instead of issuing multiple " +
    "reads. Each target is a section path (optionally prefixed with filename), " +
    "a page range like 'file.pdf > p37-41', or a filename for the whole document.",
  parameters: Type.Object({
    targets: Type.Array(Type.String(), {
      minItems: 1,
      maxItems: 10,
      description: "One to ten read targets (section path, page range, or filename).",
    }),
    doc: Type.Optional(Type.String({ description: "Default document for bare section paths." })),
  }),
  async execute(_id: string, params: any) {
    const { text } = await post<{ text: string }>("/docs_read", params);
    return textResult(text);
  },
};

export const docsOutlineTool: AgentTool = {
  name: "docs_outline",
  label: "Outline",
  description:
    "Heading tree for one document, with page numbers and character counts " +
    "so you can judge read cost before calling docs_read. Use for large " +
    "documents whose corpus-map entry is coarse.",
  parameters: Type.Object({
    doc: Type.String({ description: "Filename or unambiguous substring." }),
    max_depth: Type.Optional(
      Type.Number({ description: "Maximum heading depth to include (default 3)." }),
    ),
  }),
  async execute(_id: string, params: any) {
    const { text } = await post<{ text: string }>("/docs_outline", params);
    return textResult(text);
  },
};

export function getAllTools(): AgentTool[] {
  return [docsSearchTool, docsReadTool, docsOutlineTool];
}

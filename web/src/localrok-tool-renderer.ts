/**
 * Compact, collapsed-by-default renderers for LocalRök retrieval tools.
 */

import type { ToolResultMessage } from "@earendil-works/pi-ai";
import {
  registerToolRenderer,
  renderCollapsibleHeader,
  renderHeader,
  type ToolRenderer,
} from "@earendil-works/pi-web-ui";
import { html } from "lit";
import { createRef, ref } from "lit/directives/ref.js";
import { FileText, List, Search } from "lucide";
import { getAllTools } from "./tools.js";

type LucideIcon = typeof Search;

const TOOL_ICONS: Record<string, LucideIcon> = {
  docs_search: Search,
  docs_read: FileText,
  docs_outline: List,
};

function normalizeParams(params: unknown): Record<string, unknown> | undefined {
  if (!params) return undefined;
  if (typeof params === "object" && !Array.isArray(params)) {
    return params as Record<string, unknown>;
  }
  if (typeof params === "string") {
    try {
      const parsed = JSON.parse(params);
      if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      return undefined;
    }
  }
  return undefined;
}

function truncate(text: string, max = 72): string {
  const oneLine = text.replace(/\s+/g, " ").trim();
  if (oneLine.length <= max) return oneLine;
  return `${oneLine.slice(0, max - 1)}…`;
}

function formatParamsSummary(toolName: string, params: Record<string, unknown> | undefined): string {
  if (!params || Object.keys(params).length === 0) return "";

  const q = (key: string) => {
    const v = params[key];
    return typeof v === "string" && v.trim() ? truncate(v, 48) : undefined;
  };

  switch (toolName) {
    case "docs_search": {
      const query = q("query");
      return query ? `"${query}"` : "";
    }
    case "docs_read": {
      const targets = params.targets;
      if (Array.isArray(targets) && targets.length > 0) {
        const first = truncate(String(targets[0]), 40);
        return targets.length > 1 ? `${first} +${targets.length - 1}` : first;
      }
      return q("doc") ?? "";
    }
    case "docs_outline":
      return q("doc") ?? "";
    default:
      return "";
  }
}

function getResultText(result: ToolResultMessage | undefined): string {
  if (!result?.content?.length) return "";
  return result.content
    .filter((c) => c.type === "text")
    .map((c) => c.text)
    .join("\n");
}

function formatResultHint(result: ToolResultMessage | undefined): string {
  const text = getResultText(result);
  if (!text) return result?.isError ? "failed" : "";
  if (result?.isError) return truncate(text, 40);

  const lines = text.split("\n").filter((l) => l.trim()).length;
  const hits = text.match(/^\[\d+\]/gm)?.length ?? 0;
  if (hits > 0) return `${hits} hit${hits === 1 ? "" : "s"}`;
  if (lines > 1) return `${lines} lines`;
  return truncate(text, 40);
}

function buildHeaderText(
  label: string,
  toolName: string,
  params: Record<string, unknown> | undefined,
  result: ToolResultMessage | undefined,
  state: "inprogress" | "complete" | "error",
): string {
  const argSummary = formatParamsSummary(toolName, params);
  const base = argSummary ? `${label}: ${argSummary}` : label;

  if (state === "inprogress") return base;

  const hint = formatResultHint(result);
  if (!hint) return base;
  return `${base} · ${hint}`;
}

function prettyJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function createCompactRenderer(label: string, toolName: string, icon: LucideIcon): ToolRenderer {
  return {
    render(params: unknown, result: ToolResultMessage | undefined, isStreaming?: boolean) {
      const parsed = normalizeParams(params);
      const state = result ? (result.isError ? "error" : "complete") : isStreaming ? "inprogress" : "inprogress";
      const headerText = buildHeaderText(label, toolName, parsed, result, state);
      const output = getResultText(result);

      if (!parsed && !result) {
        return { content: renderHeader(state, icon, headerText), isCustom: false };
      }

      const contentRef = createRef<HTMLDivElement>();
      const chevronRef = createRef<HTMLSpanElement>();
      const paramsJson = parsed && Object.keys(parsed).length > 0 ? prettyJson(parsed) : "";

      return {
        content: html`
          <div>
            ${renderCollapsibleHeader(state, icon, headerText, contentRef, chevronRef, false)}
            <div ${ref(contentRef)} class="max-h-0 overflow-hidden transition-all duration-300 space-y-3">
              ${
                paramsJson
                  ? html`<div>
                      <div class="text-xs font-medium mb-1 text-muted-foreground">Input</div>
                      <code-block .code=${paramsJson} language="json"></code-block>
                    </div>`
                  : ""
              }
              ${
                output
                  ? html`<div>
                      <div class="text-xs font-medium mb-1 text-muted-foreground">Output</div>
                      <code-block .code=${output} language="text"></code-block>
                    </div>`
                  : html`<div class="text-xs text-muted-foreground">(no output)</div>`
              }
            </div>
          </div>
        `,
        isCustom: false,
      };
    },
  };
}

/** Register collapsed renderers for every LocalRök tool from tools.ts */
export function registerCompactToolRenderers(): void {
  for (const tool of getAllTools()) {
    const icon = TOOL_ICONS[tool.name] ?? Search;
    const label = tool.label ?? tool.name;
    registerToolRenderer(tool.name, createCompactRenderer(label, tool.name, icon));
  }
}

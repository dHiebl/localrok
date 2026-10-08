import "./theme-init.js";
import "@mariozechner/mini-lit/dist/ThemeToggle.js";
import { Agent } from "@earendil-works/pi-agent-core";
import type { Model } from "@earendil-works/pi-ai";
import {
  AppStorage,
  ChatPanel,
  CustomProvidersStore,
  IndexedDBStorageBackend,
  ProviderKeysStore,
  SessionsStore,
  SettingsStore,
  setAppStorage,
  defaultConvertToLlm,
} from "@earendil-works/pi-web-ui";
import "./app.css";
import { registerCompactToolRenderers } from "./localrok-tool-renderer.js";
import { getAllTools } from "./tools.js";

registerCompactToolRenderers();

type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh";

interface BootInfo {
  text: string;
  model: string;
  thinking_level: string;
  context_window: number;
  max_output_tokens: number;
}

interface StatsInfo {
  documents: number;
  sections: number;
  units: number;
  docs: { filename: string }[];
}

const FALLBACK_PROMPT =
  "You are LocalRök. Use docs_search, then docs_read before answering. Cite as [N] plus a Sources block.";

let boot: BootInfo = {
  text: FALLBACK_PROMPT,
  model: "qwen3.5:9b-mlx",
  thinking_level: "off",
  context_window: 32768,
  max_output_tokens: 4096,
};
let stats: StatsInfo = { documents: 0, sections: 0, units: 0, docs: [] };
let modelStatus: { status: string; detail?: string } = {
  status: "unknown",
  detail: "Could not check Ollama.",
};

try {
  const [promptRes, statsRes, modelRes] = await Promise.all([
    fetch("/api/system-prompt"),
    fetch("/api/stats"),
    fetch("/api/model"),
  ]);
  if (promptRes.ok) {
    boot = { ...boot, ...(await promptRes.json()) };
  }
  if (statsRes.ok) {
    stats = { ...stats, ...(await statsRes.json()) };
  }
  if (modelRes.ok) {
    modelStatus = { ...modelStatus, ...(await modelRes.json()) };
  }
} catch {
  // API offline — tools will fail gracefully
}

const thinkingLevel = (["off", "minimal", "low", "medium", "high", "xhigh"].includes(boot.thinking_level)
  ? boot.thinking_level
  : "off") as ThinkingLevel;

// -- Local Ollama model ─────────────────────────────────────────────────

const ollamaModel: Model<"openai-completions"> = {
  id: boot.model,
  name: `${boot.model} (Ollama)`,
  api: "openai-completions",
  provider: "ollama",
  baseUrl: `${window.location.origin}/v1`,
  reasoning: true,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: boot.context_window,
  maxTokens: boot.max_output_tokens,
  thinkingLevelMap: { off: "none", minimal: "low", xhigh: "high" },
  compat: {
    supportsDeveloperRole: false,
    supportsStore: false,
    supportsReasoningEffort: true,
    supportsUsageInStreaming: false,
    maxTokensField: "max_tokens",
    requiresToolResultName: false,
    requiresAssistantAfterToolResult: false,
    requiresThinkingAsText: false,
    thinkingFormat: "openai",
  },
};

// -- Storage (required by pi-web-ui internals) ───────────────────────────

const settings = new SettingsStore();
const providerKeys = new ProviderKeysStore();
const sessions = new SessionsStore();
const customProviders = new CustomProvidersStore();

const backend = new IndexedDBStorageBackend({
  dbName: "localrok-web",
  version: 2,
  stores: [
    settings.getConfig(),
    SessionsStore.getMetadataConfig(),
    providerKeys.getConfig(),
    customProviders.getConfig(),
    sessions.getConfig(),
  ],
});

settings.setBackend(backend);
providerKeys.setBackend(backend);
customProviders.setBackend(backend);
sessions.setBackend(backend);

const storage = new AppStorage(settings, providerKeys, sessions, customProviders, backend);
setAppStorage(storage);

// -- Agent ───────────────────────────────────────────────────────────────

const agent = new Agent({
  initialState: {
    systemPrompt: boot.text,
    model: ollamaModel,
    thinkingLevel,
    messages: [],
    tools: [],
  },
  convertToLlm: defaultConvertToLlm,
  getApiKey: async () => "ollama",
  onPayload: (payload) => ({
    ...(payload as Record<string, unknown>),
    max_tokens: boot.max_output_tokens,
  }),
});

// -- Mount header + ChatPanel ─────────────────────────────────────────────

const app = document.getElementById("app")!;
app.style.height = "100vh";
app.style.display = "flex";
app.style.flexDirection = "column";

const header = document.createElement("header");
header.className = "localrok-header";

const title = document.createElement("span");
title.className = "localrok-header-title";
title.textContent = "LocalRök";
header.appendChild(title);

const modelEl = document.createElement("span");
modelEl.className = "localrok-model";
if (modelStatus.status === "up") {
  modelEl.classList.add("localrok-model-up");
  modelEl.textContent = `Ollama · ${boot.model}`;
} else {
  modelEl.classList.add("localrok-model-down");
  modelEl.textContent = modelStatus.detail || "Ollama is not running.";
}
header.appendChild(modelEl);

const select = document.createElement("select");
select.className = "localrok-db-select";
const opt = document.createElement("option");
if (stats.documents === 0) {
  opt.textContent = "No index — run localrok ingest";
} else if (stats.docs.length === 1) {
  opt.textContent = `${stats.docs[0].filename} (${stats.sections} sections)`;
} else {
  opt.textContent = `${stats.documents} docs, ${stats.sections} sections`;
}
select.appendChild(opt);
header.appendChild(select);

app.appendChild(header);

const chatPanel = new ChatPanel();
await providerKeys.set("ollama", "ollama");

await chatPanel.setAgent(agent, {
  onApiKeyRequired: async (provider: string) => {
    await providerKeys.set(provider, "ollama");
    return true;
  },
  toolsFactory: () => getAllTools(),
});

const agentInterface = chatPanel.agentInterface!;
agentInterface.enableModelSelector = false;
agentInterface.showThemeToggle = true;

/** message-list ignores in-place pushes; re-render editor after idle so stop → send */
function syncChatUi(syncInput = false) {
  agent.state.messages = [...agent.state.messages];
  const list = agentInterface.querySelector("message-list") as { requestUpdate?: () => void } | null;
  list?.requestUpdate?.();
  if (syncInput) {
    agentInterface.requestUpdate();
  }
}

/** agent_end fires before finishRun(); sendMessage also bails if isStreaming is stuck */
const sendMessage = agentInterface.sendMessage.bind(agentInterface);
agentInterface.sendMessage = async (input, attachments) => {
  await agent.waitForIdle();
  (agent.state as { isStreaming: boolean }).isStreaming = false;
  agentInterface.requestUpdate();
  await sendMessage(input, attachments);
};

agent.subscribe((event) => {
  if (event.type === "message_end") {
    syncChatUi();
  } else if (event.type === "agent_end") {
    requestAnimationFrame(() => {
      (agent.state as { isStreaming: boolean }).isStreaming = false;
      syncChatUi(true);
    });
  }
});

chatPanel.style.flex = "1";
chatPanel.style.minHeight = "0";
app.appendChild(chatPanel);

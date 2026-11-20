import { createHash } from "node:crypto";
import { create } from "@bufbuild/protobuf";
import { contentToText } from "../../chat/inbound";
import { debugProviderDiagnostic } from "../../lib/debug";
import { didRunOptionalShutdownHooks, registerOptionalShutdownHook } from "../../lib/optional-shutdown-hooks";
import { isPinnedHttp1 } from "../../lib/upstream-http-version";
import type { TranslatorBudget } from "../../lib/translator-budget";
import type { OcxMessage, OcxProviderConfig, OcxToolResultMessage } from "../../types";
import { decodeCursorCallId } from "./call-id";
import { cursorCodexToWireModelId } from "./discovery";
import { resolveCursorNativeExecMode } from "./exec-policy";
import { encodeConnectFrame } from "./framing";
import {
  type AgentServerMessage,
  ExecServerMessageSchema,
  McpResultSchema,
  McpSuccessSchema,
  McpTextContentSchema,
  McpToolResultContentItemSchema,
} from "./gen/agent_pb";
import { execBytes } from "./native-exec-common";
import { cursorRequestUsesCodeMode } from "./tool-naming";
import { normalizeCursorToolResultText } from "./tool-result-normalize";
import type { CursorTransport } from "./transport";
import type { CursorRunRequest, CursorServerMessage } from "./types";

/**
 * Held Cursor runs (opt-in, OPENCODEX_CURSOR_HELD_STREAM=1).
 *
 * After a client tool call the adapter normally cancels the Cursor stream and, when the tool result
 * arrives, opens a new run whose active turn is a user-shaped "Continue:" message. Grok then opens
 * most later reasoning rounds with "The user wants me to continue". The Cursor Agent CLI instead keeps
 * the stream open and answers the pending exec with an `mcpResult`. This module owns the part of that
 * behaviour that is not transport code: which runs may be held, the registry that keeps them, and the
 * rules that decide whether a later request may take one over.
 *
 * First slice: exactly one bridged tool call pending, plain-text result, HTTP/2, Grok 4.6 and 4.7.
 * Anything else, and any doubt, falls back to the unchanged replay path.
 */

/** A held run waits this long after it was parked. Cursor itself kept an idle stream for 14+ minutes. */
export const HELD_RUN_LEASE_MS = 10 * 60_000;
export const HELD_RUN_MAX = 24;
/** Larger results take the replay path, which already truncates to the replay budget. */
export const HELD_RESULT_MAX_BYTES = 512 * 1024;

/**
 * Settings that shaped the parked transport: the wire version and whether native exec is allowed.
 * A resume on a route whose settings changed since the park must not reuse the old socket.
 */
export function heldTransportPolicy(provider: OcxProviderConfig): string {
  return `${isPinnedHttp1(provider.upstreamHttpVersion) ? "h1" : "h2"}:${resolveCursorNativeExecMode(provider)}`;
}

export function heldStreamEnabled(): boolean {
  return process.env.OPENCODEX_CURSOR_HELD_STREAM === "1";
}

/** Grok 4.6 and 4.7 wire ids, including effort and fast forms such as `grok-4.7-medium-fast`. */
export function isHeldStreamModel(modelId: string): boolean {
  const wire = cursorCodexToWireModelId(modelId).trim().toLowerCase().replace(/^cursor-/, "");
  // grok-4.7, grok-4.7-high, grok-4.7-high-fast and the catalog's grok-4.7--fast; nothing else after the version.
  return /^grok-4\.[67](?:-(?:none|minimal|low|medium|high|xhigh|max))?(?:--?fast)?$/.test(wire);
}

export function heldRunEligible(request: CursorRunRequest, http1: boolean): boolean {
  const scope = request._cursorIdentityScope?.trim();
  return heldStreamEnabled()
    && !http1
    && isHeldStreamModel(request.modelId)
    // "local" is the fallback scope for a request with no identifiable credential, and
    // "cursor-unowned:" is a fresh random scope minted per request, so a run parked under it could
    // never be claimed. Neither may meet another caller at a registry key.
    && !!scope && scope !== "local" && !scope.startsWith("cursor-unowned:")
    && request.rawMessages !== undefined
    && request.echoRetryContinuationText === undefined
    // Compaction and isolated helper turns store no checkpoint and belong to no thread.
    && request.contextUsageStoreCheckpoints !== false
    && request.contextUsageReset !== true;
}

export interface HeldResult {
  toolCallId: string;
  text: string;
  isError: boolean;
}

/** What the registry needs from a parked transport. The transport owns its own teardown. */
export interface HeldTransport extends CursorTransport {
  heldUsable(): boolean;
  /** Arms the next `run()` to answer the pending exec with `result` instead of opening a new run. */
  heldAdopt(budget: TranslatorBudget, result: HeldResult, dispatch: () => void): void;
  heldDispose(): void;
  heldResultWritten(): boolean;
}

/** The call exactly as it was handed to the client: tool name and the JSON argument text. */
export interface HeldCall {
  name: string;
  args: string;
}

/** Key order must not matter: a client may re-serialize the arguments it parsed. */
function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map(key => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(",")}}`;
  }
  return JSON.stringify(value) ?? "null";
}

function sameCall(announced: HeldCall, part: { name: string; namespace?: string; arguments?: Record<string, unknown> }): boolean {
  const qualified = part.namespace ? `${part.namespace}${part.name}` : part.name;
  if (announced.name !== part.name && announced.name !== qualified) return false;
  try {
    const sent = announced.args.trim() === "" ? {} : JSON.parse(announced.args);
    return canonicalJson(sent) === canonicalJson(part.arguments ?? {});
  } catch {
    return false;
  }
}

interface HeldEntry {
  key: string;
  scope: string;
  conversationId: string;
  modelId: string;
  policy: string;
  toolCallId: string;
  call: HeldCall;
  coveredCount: number;
  coverDigest: string;
  owner: object;
  handle: HeldTransport;
  timer: ReturnType<typeof setTimeout>;
}

const runs = new Map<string, HeldEntry>();
let leaseMs = HELD_RUN_LEASE_MS;
let shutdownArmed = false;

function frame(hash: ReturnType<typeof createHash>, value: string): void {
  const bytes = Buffer.from(value, "utf8");
  const length = Buffer.allocUnsafe(4);
  length.writeUInt32BE(bytes.byteLength);
  hash.update(length);
  hash.update(bytes);
}

function messageSignature(message: OcxMessage): string {
  if (message.role === "assistant") {
    const parts = message.content.map(part => {
      if (part.type === "text") return `t:${part.text}`;
      if (part.type === "toolCall") {
        return `c:${decodeCursorCallId(part.id)}:${part.namespace ?? ""}:${part.name}:${part.customWireName ?? ""}:${JSON.stringify(part.arguments ?? {})}`;
      }
      return `x:${part.type}`;
    });
    return `assistant\0${parts.join("\0")}`;
  }
  return `${message.role}\0${contentToText(message.content)}`;
}

/**
 * Digest of everything the held run's conversation was built from: model, instructions, tool
 * catalog and choice, and the first `coveredCount` messages. A held stream continues the history
 * Cursor already has, so a regenerated or edited branch must not match even with the same ids.
 */
export function heldCoverDigest(request: CursorRunRequest, coveredCount: number): string {
  const hash = createHash("sha256").update("ocx:cursor:held:");
  frame(hash, request.modelId);
  // Everything that changes what Cursor was asked for, so a changed effort, mode or output format
  // cannot adopt a stream that was started under the old ones.
  frame(hash, JSON.stringify([
    request.requestedModelParameters ?? null,
    request.routingLevel ?? null,
    request.maxMode ?? false,
    request.parallelToolCalls ?? null,
    request.textFormat ?? null,
    request.suppressDefaultCursorToolCatalog ?? false,
  ]));
  for (const line of request.system) frame(hash, line);
  frame(hash, JSON.stringify(request.tools ?? []));
  frame(hash, JSON.stringify(request.toolChoice ?? null));
  for (const message of (request.rawMessages ?? []).slice(0, coveredCount)) frame(hash, messageSignature(message));
  return hash.digest("hex");
}

/**
 * Keyed by scope and tool call id only. A client that sends no thread id (OpenCode over chat
 * completions) gets a new random conversation id on every request, so the conversation id cannot be
 * part of the key. The branch digest binds the history, instructions, tools and model instead.
 */
function heldKey(scope: string, toolCallId: string): string {
  return `${scope}\0${toolCallId}`;
}

function removeEntry(entry: HeldEntry): void {
  clearTimeout(entry.timer);
  if (runs.get(entry.key) === entry) runs.delete(entry.key);
}

function disposeEntry(entry: HeldEntry, reason: string): void {
  removeEntry(entry);
  debugProviderDiagnostic("cursor", "held-run-dispose", { reason, held: runs.size });
  try { entry.handle.heldDispose(); } catch { /* a failed close must not break the request path */ }
}

export function closeAllHeldRuns(): void {
  for (const entry of [...runs.values()]) disposeEntry(entry, "shutdown");
}

export function heldRunCount(): number {
  return runs.size;
}

/** Test seam: shorten the lease. */
export function setHeldRunLeaseMsForTests(ms: number | undefined): void {
  leaseMs = ms ?? HELD_RUN_LEASE_MS;
}

export function resetHeldRunsForTests(): void {
  for (const entry of [...runs.values()]) removeEntry(entry);
  leaseMs = HELD_RUN_LEASE_MS;
}

export function registerHeldRun(input: {
  request: CursorRunRequest;
  policy: string;
  toolCallId: string;
  call: HeldCall;
  owner: object;
  handle: HeldTransport;
}): string | undefined {
  const scope = input.request._cursorIdentityScope?.trim();
  const raw = input.request.rawMessages;
  if (!scope || !raw) return undefined;
  const key = heldKey(scope, input.toolCallId);
  const previous = runs.get(key);
  if (previous) disposeEntry(previous, "replaced");
  while (runs.size >= HELD_RUN_MAX) {
    const oldest = runs.values().next().value;
    if (!oldest) break;
    disposeEntry(oldest, "cap");
  }
  const entry: HeldEntry = {
    key,
    scope,
    conversationId: input.request.conversationId,
    modelId: input.request.modelId,
    policy: input.policy,
    toolCallId: input.toolCallId,
    call: input.call,
    coveredCount: raw.length,
    coverDigest: heldCoverDigest(input.request, raw.length),
    owner: input.owner,
    handle: input.handle,
    timer: setTimeout(() => disposeEntry(entry, "lease"), leaseMs),
  };
  entry.timer.unref?.();
  runs.set(key, entry);
  if (!shutdownArmed && !didRunOptionalShutdownHooks()) {
    shutdownArmed = true;
    registerOptionalShutdownHook("cursor-held-runs", closeAllHeldRuns);
  }
  debugProviderDiagnostic("cursor", "held-run-park", { held: runs.size });
  return key;
}

/** Called by a transport when it closed or died on its own. Only the registered owner may remove its entry. */
export function unregisterHeldRun(key: string | undefined, owner: object): void {
  if (!key) return;
  const entry = runs.get(key);
  if (entry && entry.owner === owner) removeEntry(entry);
}

/** The single result the request carries, or undefined when the request is not a clean continuation. */
function matchEntry(request: CursorRunRequest, entry: HeldEntry, policy: string): HeldResult | undefined {
  const raw = request.rawMessages;
  const last = raw?.at(-1);
  if (!raw || last?.role !== "toolResult" || request.echoRetryContinuationText !== undefined) return undefined;
  if (entry.modelId !== request.modelId || entry.policy !== policy) return undefined;
  const result: OcxToolResultMessage = last;
  const toolCallId = decodeCursorCallId(result.toolCallId);
  if (toolCallId !== entry.toolCallId) return undefined;
  const tail = raw.slice(entry.coveredCount);
  if (tail.length < 2 || tail.at(-1) !== last) return undefined;
  const calls = tail.slice(0, -1).flatMap(message =>
    message.role === "assistant" ? message.content.filter(part => part.type === "toolCall") : undefined);
  // Every message between the covered prefix and the result must be an assistant message that
  // carries exactly the one pending call. Extra user text, other results or other calls mean the
  // client's branch is not the one this stream is on.
  if (calls.some(call => call === undefined) || calls.length !== 1) return undefined;
  const sent = calls[0] as { id: string; name: string; namespace?: string; arguments?: Record<string, unknown> };
  if (decodeCursorCallId(sent.id) !== toolCallId) return undefined;
  // The result answers the call the model made, so the client's copy of that call must be the one
  // that was handed to it. An edited or regenerated call with the same id is a different branch.
  if (!sameCall(entry.call, sent)) return undefined;
  if (heldCoverDigest(request, entry.coveredCount) !== entry.coverDigest) return undefined;
  if (result.containsEncryptedContent) return undefined;
  if (Array.isArray(result.content) && result.content.some(part => part.type !== "text")) return undefined;
  const text = contentToText(result.content);
  if (Buffer.byteLength(text, "utf8") > HELD_RESULT_MAX_BYTES) return undefined;
  const normalized = normalizeCursorToolResultText(text, {
    toolName: result.toolName,
    toolNamespace: result.toolNamespace,
    isError: result.isError,
    codeMode: cursorRequestUsesCodeMode(request.tools, request.toolChoice),
  });
  return { toolCallId, text: normalized.text, isError: normalized.isError };
}

export interface HeldClaim {
  transport: HeldTransport;
  /** Hand the claimed transport the new request's budget and send accounting hook. */
  adopt(budget: TranslatorBudget, dispatch: () => void): void;
  dispose(): void;
}

/**
 * Take over the held run a request continues. A request in the same conversation that does not
 * continue it closes that run, because the stream is on a branch the client has left. The claim is
 * atomic: the entry leaves the registry before the caller does anything else with it.
 */
export function claimHeldRun(request: CursorRunRequest, policy: string): HeldClaim | undefined {
  if (runs.size === 0 || !heldStreamEnabled() || request.rawMessages?.at(-1)?.role !== "toolResult") return undefined;
  const scope = request._cursorIdentityScope?.trim();
  if (!scope || scope === "local" || scope.startsWith("cursor-unowned:")) return undefined;
  let claimed: { entry: HeldEntry; result: HeldResult } | undefined;
  for (const entry of [...runs.values()]) {
    if (entry.scope !== scope) continue;
    const result = claimed ? undefined : matchEntry(request, entry, policy);
    if (result) {
      claimed = { entry, result };
      continue;
    }
    // A request in the same conversation that does not continue this run has left its branch.
    // With a random per-request conversation id that case cannot be told apart, so the lease
    // closes the run instead.
    if (entry.conversationId === request.conversationId) disposeEntry(entry, "not-continued");
  }
  if (!claimed) return undefined;
  const { entry, result } = claimed;
  removeEntry(entry);
  if (!entry.handle.heldUsable()) {
    disposeEntry(entry, "unusable");
    return undefined;
  }
  debugProviderDiagnostic("cursor", "held-run-claim", { held: runs.size });
  return {
    transport: entry.handle,
    adopt: (budget, dispatch) => entry.handle.heldAdopt(budget, result, dispatch),
    dispose: () => disposeEntry(entry, "claim-aborted"),
  };
}

/** The Connect frame that answers one pending bridged exec with a plain-text `mcpResult`. */
export function heldResultFrame(exec: { id: number; execId: string }, text: string, isError: boolean): Uint8Array {
  const value = create(McpResultSchema, {
    result: {
      case: "success",
      value: create(McpSuccessSchema, {
        isError,
        content: [create(McpToolResultContentItemSchema, {
          content: { case: "text", value: create(McpTextContentSchema, { text }) },
        })],
      }),
    },
  });
  return encodeConnectFrame(execBytes(create(ExecServerMessageSchema, { id: exec.id, execId: exec.execId }), "mcpResult", value));
}

/**
 * Frames a parked run may receive. Measured on a live account: a conversation checkpoint within about
 * 0.3 s, then a heartbeat roughly every 10 s. Anything else (a sibling tool call, model output,
 * turnEnded, an interaction query) means the stream is no longer in the state the client was told
 * about, so the run is dropped and the next request replays history as it does today.
 */
export function heldFrameAllowed(message: AgentServerMessage): boolean {
  switch (message.message.case) {
    case "kvServerMessage":
    case "conversationCheckpointUpdate":
      return true;
    case "interactionUpdate": {
      const inner = message.message.value.message.case;
      // `undefined` is an update type this schema does not know. The mapper ignores those too, and a
      // live run saw one right after a tool call, so treating it as foreign would drop healthy runs.
      return inner === undefined || inner === "heartbeat" || inner === "tokenDelta";
    }
    default:
      return false;
  }
}

/** Replace every field of `target` with those of `fresh`, so closures holding `target` see the new turn. */
export function replaceStateInPlace(target: object, fresh: object): void {
  const record = target as Record<string, unknown>;
  for (const key of Object.keys(record)) delete record[key];
  Object.assign(record, fresh);
}

export interface HeldTurnSink {
  push(message: CursorServerMessage): void;
  fail(error: Error): void;
  finish(): void;
}

/** Per-transport bookkeeping. Plain data so live-transport.ts stays small. */
export interface HeldRunState {
  parked: boolean;
  dead: boolean;
  /** Bumped by each resume so an abort listener from an earlier turn can tell it is stale. */
  generation: number;
  key?: string;
  /** The request whose history the next adoption must continue. */
  request?: CursorRunRequest;
  /** The event-state object the stream closures hold; a resume replaces its contents in place. */
  state?: object;
  stateOptions?: Record<string, unknown>;
  /** Bridged execs seen this turn by raw tool call id, and the call ids actually announced to the client. */
  execs: Map<string, { id: number; execId: string }>;
  announced: Set<string>;
  /** Calls as emitted to the client, by raw call id, and the one currently being emitted. */
  calls: Map<string, HeldCall>;
  currentCall?: string;
  adoption?: { budget: TranslatorBudget; result: HeldResult; dispatch: () => void };
  sink?: HeldTurnSink;
  endTurn?: () => void;
  removeAbort?: () => void;
  failTurn?: (error: Error) => void;
  /** Set when a resume wrote the client's result; the request must not be replayed afterwards. */
  resultWritten: boolean;
}

export function newHeldRunState(): HeldRunState {
  return { parked: false, dead: false, generation: 0, execs: new Map(), announced: new Set(), calls: new Map(), resultWritten: false };
}

/** The one bridged call a turn may park on, or undefined when the turn does not qualify. */
export function heldParkTarget(
  held: HeldRunState,
  http1: boolean,
): { toolCallId: string; exec: { id: number; execId: string }; call: HeldCall } | undefined {
  const request = held.request;
  const skip = (reason: string): undefined => {
    if (heldStreamEnabled()) {
      debugProviderDiagnostic("cursor", "held-run-skip", {
        reason, execs: held.execs.size, announced: held.announced.size, http1,
      });
    }
    return undefined;
  };
  if (!request || held.dead) return skip("no-request");
  if (!heldRunEligible(request, http1)) return skip("not-eligible");
  if (held.execs.size !== 1 || held.announced.size !== 1) return skip("not-single-call");
  const [toolCallId, exec] = [...held.execs][0]!;
  const call = held.calls.get(toolCallId);
  return held.announced.has(toolCallId) && call ? { toolCallId, exec, call } : skip("call-not-announced");
}

import http2 from "node:http2";
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { create, fromBinary, toBinary } from "@bufbuild/protobuf";
import {
  AgentClientMessageSchema,
  AgentServerMessageSchema,
  ConversationStateStructureSchema,
  ExecServerMessageSchema,
  HeartbeatUpdateSchema,
  InteractionUpdateSchema,
  McpArgsSchema,
  McpToolCallSchema,
  TextDeltaUpdateSchema,
  ToolCallCompletedUpdateSchema,
  ToolCallSchema,
  ToolCallStartedUpdateSchema,
  TurnEndedUpdateSchema,
} from "../../../src/adapters/cursor/gen/agent_pb";
import { consumeConnectFrames, encodeConnectFrame } from "../../../src/adapters/cursor/framing";
import {
  claimHeldRun,
  heldCoverDigest,
  heldFrameAllowed,
  heldRunCount,
  heldRunEligible,
  heldTransportPolicy,
  isHeldStreamModel,
  registerHeldRun,
  resetHeldRunsForTests,
  setHeldRunLeaseMsForTests,
  type HeldTransport,
} from "../../../src/adapters/cursor/held-run";
import { createLiveCursorTransport } from "../../../src/adapters/cursor/live-transport";
import { cursorToolWireName } from "../../../src/adapters/cursor/tool-naming";
import { runCursorTurnWithRetry } from "../../../src/adapters/cursor/transport-retry";
import type { CursorRunRequest, CursorServerMessage } from "../../../src/adapters/cursor/types";
import type { OcxMessage } from "../../../src/types";
import { createTestTranslatorBudget } from "../../helpers/translator-budget";

/**
 * Held Cursor runs (src/adapters/cursor/held-run.ts, OPENCODEX_CURSOR_HELD_STREAM=1).
 *
 * The transport tests run against a real local HTTP/2 server so a park, a resume and a second park
 * exercise the same stream, which is where the prototype lost 35 of 73 runs to a stale abort listener.
 */

const PROVIDER = "opencodex-responses";
const SCOPE = "scope-a";
const MODEL = "grok-4.7-medium-fast";

function agent(message: Parameters<typeof create<typeof AgentServerMessageSchema>>[1]): Uint8Array {
  return encodeConnectFrame(toBinary(AgentServerMessageSchema, create(AgentServerMessageSchema, message)));
}

function toolCall(callId: string, name: string) {
  return create(ToolCallSchema, {
    tool: {
      case: "mcpToolCall",
      value: create(McpToolCallSchema, {
        args: create(McpArgsSchema, { name, toolName: name, toolCallId: callId, providerIdentifier: PROVIDER }),
      }),
    },
  });
}

function startedFrame(callId: string, name: string): Uint8Array {
  return agent({
    message: {
      case: "interactionUpdate",
      value: create(InteractionUpdateSchema, {
        message: { case: "toolCallStarted", value: create(ToolCallStartedUpdateSchema, { callId, modelCallId: callId, toolCall: toolCall(callId, name) }) },
      }),
    },
  });
}

function execFrame(id: number, callId: string, name: string): Uint8Array {
  return agent({
    message: {
      case: "execServerMessage",
      value: create(ExecServerMessageSchema, {
        id,
        execId: `exec-${callId}`,
        message: {
          case: "mcpArgs",
          value: create(McpArgsSchema, {
            name, toolName: name, toolCallId: callId, providerIdentifier: PROVIDER,
            args: { path: new TextEncoder().encode(JSON.stringify("d1.txt")) },
          }),
        },
      }),
    },
  });
}

function completedFrame(callId: string, name: string): Uint8Array {
  return agent({
    message: {
      case: "interactionUpdate",
      value: create(InteractionUpdateSchema, {
        message: { case: "toolCallCompleted", value: create(ToolCallCompletedUpdateSchema, { callId, modelCallId: callId, toolCall: toolCall(callId, name) }) },
      }),
    },
  });
}

function checkpointFrame(): Uint8Array {
  return agent({ message: { case: "conversationCheckpointUpdate", value: create(ConversationStateStructureSchema, { pendingToolCalls: ["held"] }) } });
}

function heartbeatFrame(): Uint8Array {
  return agent({ message: { case: "interactionUpdate", value: create(InteractionUpdateSchema, { message: { case: "heartbeat", value: create(HeartbeatUpdateSchema, {}) } }) } });
}

function textFrame(text: string): Uint8Array {
  return agent({ message: { case: "interactionUpdate", value: create(InteractionUpdateSchema, { message: { case: "textDelta", value: create(TextDeltaUpdateSchema, { text }) } }) } });
}

function turnEndedFrame(): Uint8Array {
  return agent({ message: { case: "interactionUpdate", value: create(InteractionUpdateSchema, { message: { case: "turnEnded", value: create(TurnEndedUpdateSchema, {}) } }) } });
}

const CATALOG = [{ name: "read_file", description: "read", parameters: { type: "object", properties: { path: { type: "string" } } } }];
const WIRE = cursorToolWireName(CATALOG[0]!, CATALOG);

const POLICY = heldTransportPolicy({ adapter: "cursor", baseUrl: "x", apiKey: "t" } as never);
const claim_ = (request: CursorRunRequest) => claimHeldRun(request, POLICY);

const userMessage = (text: string): OcxMessage => ({ role: "user", content: text, timestamp: 1 });
const assistantCall = (id: string, name = "read_file"): OcxMessage => ({
  role: "assistant", content: [{ type: "toolCall", id, name, arguments: { path: "d1.txt" } }], timestamp: 2,
});
const toolResult = (id: string, content: string, extra: Partial<Extract<OcxMessage, { role: "toolResult" }>> = {}): OcxMessage => ({
  role: "toolResult", toolCallId: id, toolName: "read_file", content, isError: false, timestamp: 3, ...extra,
});

function baseRequest(rawMessages: OcxMessage[], overrides: Partial<CursorRunRequest> = {}): CursorRunRequest {
  return {
    modelId: MODEL,
    conversationId: "cursor_held_test",
    _cursorIdentityScope: SCOPE,
    system: ["be brief"],
    messages: [{ role: "user", content: "read d1" }],
    rawMessages,
    tools: CATALOG,
    ...overrides,
  } as CursorRunRequest;
}

interface FakeServer {
  baseUrl: string;
  streams: http2.ServerHttp2Stream[];
  /** Raw client frames per stream (after the initial run request). */
  clientFrames: Array<ReturnType<typeof fromBinary<typeof AgentClientMessageSchema>>[]>;
  close(): Promise<void>;
}

async function startServer(onStream: (stream: http2.ServerHttp2Stream, index: number) => void): Promise<FakeServer> {
  const server = http2.createServer();
  const streams: http2.ServerHttp2Stream[] = [];
  const clientFrames: FakeServer["clientFrames"] = [];
  server.on("stream", stream => {
    const index = streams.length;
    streams.push(stream);
    clientFrames.push([]);
    stream.respond({ ":status": 200, "content-type": "application/connect+proto" });
    let pending = new Uint8Array();
    stream.on("data", (chunk: Buffer) => {
      const merged = new Uint8Array(pending.byteLength + chunk.byteLength);
      merged.set(pending); merged.set(chunk, pending.byteLength);
      const { frames, consumedBytes } = consumeConnectFrames(merged);
      pending = merged.slice(consumedBytes);
      for (const frame of frames) {
        try { clientFrames[index]!.push(fromBinary(AgentClientMessageSchema, frame.payload)); } catch { /* not a client message */ }
      }
    });
    stream.on("error", () => {});
    onStream(stream, index);
  });
  await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("no port");
  return {
    baseUrl: `http://127.0.0.1:${address.port}`,
    streams,
    clientFrames,
    close: () => new Promise<void>(resolve => { for (const s of streams) s.destroy(); server.close(() => resolve()); }),
  };
}

const mcpResults = (server: FakeServer, index = 0) =>
  server.clientFrames[index]!.flatMap(frame =>
    frame.message.case === "execClientMessage" && frame.message.value.message.case === "mcpResult" ? [frame.message.value] : []);

async function waitFor(check: () => boolean, label: string, timeoutMs = 4000): Promise<void> {
  const start = Date.now();
  while (!check()) {
    if (Date.now() - start > timeoutMs) throw new Error(`timed out waiting for ${label}`);
    await new Promise(resolve => setTimeout(resolve, 10));
  }
}

function makeFactory(baseUrl: string) {
  return (input: Parameters<typeof createLiveCursorTransport>[0]) =>
    createLiveCursorTransport({ ...input, provider: { adapter: "cursor", baseUrl, apiKey: "test-token" }, clientToolFinalizeGraceMs: 20 });
}

async function runTurn(factory: ReturnType<typeof makeFactory>, request: CursorRunRequest, signal?: AbortSignal) {
  const events: CursorServerMessage[] = [];
  const budget = createTestTranslatorBudget();
  await runCursorTurnWithRetry(factory, { provider: { adapter: "cursor", baseUrl: "x", apiKey: "t" }, translatorBudget: budget, headers: new Headers() }, request, signal, m => { events.push(m); });
  return events;
}

beforeEach(() => {
  process.env.OPENCODEX_CURSOR_HELD_STREAM = "1";
  resetHeldRunsForTests();
});
afterEach(() => {
  delete process.env.OPENCODEX_CURSOR_HELD_STREAM;
  resetHeldRunsForTests();
});

describe("held run gate", () => {
  test("flag is off unless the variable is exactly 1", () => {
    delete process.env.OPENCODEX_CURSOR_HELD_STREAM;
    expect(heldRunEligible(baseRequest([userMessage("hi")]), false)).toBe(false);
    process.env.OPENCODEX_CURSOR_HELD_STREAM = "true";
    expect(heldRunEligible(baseRequest([userMessage("hi")]), false)).toBe(false);
    process.env.OPENCODEX_CURSOR_HELD_STREAM = "1";
    expect(heldRunEligible(baseRequest([userMessage("hi")]), false)).toBe(true);
  });

  test("only Grok 4.6 and 4.7 wire ids qualify, including effort and fast forms", () => {
    for (const id of ["grok-4.6", "grok-4.7", "grok-4.7-medium-fast", "grok-4.6-xhigh", "cursor/grok-4.7--fast", "cursor-grok-4.7-high"]) {
      expect(isHeldStreamModel(id)).toBe(true);
    }
    for (const id of ["grok-4.5", "grok-4.70", "grok-4.7-experimental", "grok-4.7-high-extra", "grok-4.7-fast-high", "composer-2.5", "claude-sonnet-5-5", "gpt-5", "default", "grok-5"]) {
      expect(isHeldStreamModel(id)).toBe(false);
    }
  });

  test("a request without a real identity scope, with HTTP/1, or in an echo retry never qualifies", () => {
    expect(heldRunEligible(baseRequest([userMessage("x")], { _cursorIdentityScope: "local" }), false)).toBe(false);
    expect(heldRunEligible(baseRequest([userMessage("x")], { _cursorIdentityScope: undefined }), false)).toBe(false);
    expect(heldRunEligible(baseRequest([userMessage("x")], { _cursorIdentityScope: "cursor-unowned:abc" }), false)).toBe(false);
    expect(heldRunEligible(baseRequest([userMessage("x")]), true)).toBe(false);
    expect(heldRunEligible(baseRequest([userMessage("x")], { echoRetryContinuationText: "again" }), false)).toBe(false);
  });

  test("parked frames are limited to liveness and accounting", () => {
    const decode = (bytes: Uint8Array) => fromBinary(AgentServerMessageSchema, consumeConnectFrames(bytes).frames[0]!.payload);
    expect(heldFrameAllowed(decode(heartbeatFrame()))).toBe(true);
    expect(heldFrameAllowed(decode(checkpointFrame()))).toBe(true);
    expect(heldFrameAllowed(decode(startedFrame("c2", WIRE)))).toBe(false);
    expect(heldFrameAllowed(decode(execFrame(2, "c2", WIRE)))).toBe(false);
    expect(heldFrameAllowed(decode(textFrame("hello")))).toBe(false);
    expect(heldFrameAllowed(decode(turnEndedFrame()))).toBe(false);
  });
});

describe("held run registry", () => {
  const handle = (log: string[] = []): HeldTransport => ({
    run: async function* () {},
    writeClient() {},
    heldUsable: () => true,
    heldAdopt: () => { log.push("adopt"); },
    heldDispose: () => { log.push("dispose"); },
    heldResultWritten: () => false,
  });
  const parkedRaw = [userMessage("read d1")];
  const continuation = (callId: string, extra: OcxMessage[] = []) => [...parkedRaw, assistantCall(callId), toolResult(callId, "teal"), ...extra];
  const park = (log: string[], callId = "call-1", overrides: Partial<CursorRunRequest> = {}) =>
    registerHeldRun({ request: baseRequest(parkedRaw, overrides), policy: POLICY, toolCallId: callId, call: { name: "read_file", args: "{\"path\":\"d1.txt\"}" }, owner: log, handle: handle(log) });

  test("a continuation with the same history, model and tools claims the run exactly once", () => {
    const log: string[] = [];
    park(log);
    expect(heldRunCount()).toBe(1);
    const claim = claim_(baseRequest(continuation("call-1")));
    expect(claim).toBeDefined();
    expect(heldRunCount()).toBe(0);
    expect(claim_(baseRequest(continuation("call-1")))).toBeUndefined();
  });

  test("a different identity scope never claims the run and does not close it", () => {
    const log: string[] = [];
    park(log);
    expect(claim_(baseRequest(continuation("call-1"), { _cursorIdentityScope: "scope-b" }))).toBeUndefined();
    expect(heldRunCount()).toBe(1);
    expect(log).toEqual([]);
  });

  test("a client without a thread id gets a new conversation id per request and still claims its run", () => {
    park([]);
    // OpenCode over chat completions sends no thread id, so the id differs on every request.
    expect(claim_(baseRequest(continuation("call-1"), { conversationId: "cursor_other_random_id" }))).toBeDefined();
  });

  test("another conversation with a different history does not match and leaves the run alone", () => {
    const log: string[] = [];
    park(log);
    const other = baseRequest([userMessage("unrelated"), assistantCall("call-1"), toolResult("call-1", "x")], { conversationId: "elsewhere" });
    expect(claim_(other)).toBeUndefined();
    expect(heldRunCount()).toBe(1);
    expect(log).toEqual([]);
  });

  test("a forged or wrong tool call id misses and closes the run only through the branch rule", () => {
    const log: string[] = [];
    park(log);
    expect(claim_(baseRequest(continuation("call-zzz")))).toBeUndefined();
    // The request is in the same conversation but does not continue the held run, so the run is dropped.
    expect(log).toEqual(["dispose"]);
    expect(heldRunCount()).toBe(0);
  });

  test("a regenerated branch, changed instructions, changed tools or another model do not match", () => {
    for (const change of [
      { rawMessages: [userMessage("a different question"), assistantCall("call-1"), toolResult("call-1", "teal")] },
      { system: ["new instructions"] },
      { tools: [] },
      { modelId: "grok-4.6-medium" },
    ] satisfies Array<Partial<CursorRunRequest>>) {
      resetHeldRunsForTests();
      const log: string[] = [];
      park(log);
      const raw = (change as { rawMessages?: OcxMessage[] }).rawMessages ?? continuation("call-1");
      expect(claim_(baseRequest(raw, change))).toBeUndefined();
    }
  });

  test("extra user text, a second call or a second result between the prefix and the result does not match", () => {
    const log: string[] = [];
    park(log);
    expect(claim_(baseRequest([...parkedRaw, assistantCall("call-1"), userMessage("also this"), toolResult("call-1", "x")]))).toBeUndefined();
    resetHeldRunsForTests();
    park(log);
    expect(claim_(baseRequest([...parkedRaw, assistantCall("call-1"), toolResult("call-1", "x"), assistantCall("call-2"), toolResult("call-2", "y")]))).toBeUndefined();
  });

  test("image, encrypted and oversized results take the replay path", () => {
    const cases: OcxMessage[] = [
      toolResult("call-1", "", { content: [{ type: "image", data: "AAAA", mimeType: "image/png" }] as never }),
      toolResult("call-1", "x", { containsEncryptedContent: true }),
      toolResult("call-1", "x".repeat(600 * 1024)),
    ];
    for (const result of cases) {
      resetHeldRunsForTests();
      park([]);
      expect(claim_(baseRequest([...parkedRaw, assistantCall("call-1"), result]))).toBeUndefined();
    }
  });

  test("the lease closes a run that nothing continues", async () => {
    setHeldRunLeaseMsForTests(30);
    const log: string[] = [];
    park(log);
    await waitFor(() => heldRunCount() === 0, "lease expiry");
    expect(log).toEqual(["dispose"]);
  });

  test("the registry holds at most 24 runs and drops the oldest first", () => {
    const logs: string[][] = [];
    for (let i = 0; i < 25; i++) {
      const log: string[] = [];
      logs.push(log);
      park(log, `call-${i}`);
    }
    expect(heldRunCount()).toBe(24);
    expect(logs[0]).toEqual(["dispose"]);
    expect(logs[24]).toEqual([]);
  });

  test("editing the announced call's arguments, or changing effort or output settings, does not match", () => {
    const edited: OcxMessage = { role: "assistant", content: [{ type: "toolCall", id: "call-1", name: "read_file", arguments: { path: "other.txt" } }], timestamp: 2 };
    for (const [raw, overrides] of [
      [[...parkedRaw, edited, toolResult("call-1", "x")], {}],
      [continuation("call-1"), { requestedModelParameters: [{ id: "effort", value: "high" }] }],
      [continuation("call-1"), { parallelToolCalls: true }],
      [continuation("call-1"), { textFormat: { type: "json_object" } }],
      [continuation("call-1"), { maxMode: true }],
    ] as Array<[OcxMessage[], Partial<CursorRunRequest>]>) {
      resetHeldRunsForTests();
      park([]);
      expect(claim_(baseRequest(raw, overrides as Partial<CursorRunRequest>))).toBeUndefined();
    }
  });

  test("a changed transport policy (HTTP version or native exec mode) does not adopt the old socket", () => {
    park([]);
    const changed = heldTransportPolicy({ adapter: "cursor", baseUrl: "x", apiKey: "t", upstreamHttpVersion: "http1.1" } as never);
    expect(changed).not.toBe(POLICY);
    expect(claimHeldRun(baseRequest(continuation("call-1")), changed)).toBeUndefined();
  });

  test("compaction and isolated turns never qualify to be held", () => {
    expect(heldRunEligible(baseRequest([userMessage("x")], { contextUsageStoreCheckpoints: false }), false)).toBe(false);
    expect(heldRunEligible(baseRequest([userMessage("x")], { contextUsageReset: true }), false)).toBe(false);
  });

  test("a request that is not a tool-result continuation exits before scanning the registry", () => {
    const log: string[] = [];
    park(log);
    expect(claim_(baseRequest([userMessage("a new question")]))).toBeUndefined();
    // Not a continuation, so it is not evidence the client left the branch either.
    expect(heldRunCount()).toBe(1);
    expect(log).toEqual([]);
  });

  test("the digest covers history, instructions, tools and model", () => {
    const base = heldCoverDigest(baseRequest(parkedRaw), 1);
    expect(heldCoverDigest(baseRequest(parkedRaw), 1)).toBe(base);
    expect(heldCoverDigest(baseRequest(parkedRaw, { system: ["x"] }), 1)).not.toBe(base);
    expect(heldCoverDigest(baseRequest(parkedRaw, { tools: [] }), 1)).not.toBe(base);
    expect(heldCoverDigest(baseRequest([userMessage("other")]), 1)).not.toBe(base);
  });
});

describe("held run transport", () => {
  test("a Grok tool call parks, the next request answers it with an mcpResult on the same stream, and it parks again", async () => {
    const server = await startServer((stream, index) => {
      if (index !== 0) return;
      stream.write(startedFrame("call-1", WIRE));
      stream.write(execFrame(1, "call-1", WIRE));
      stream.write(completedFrame("call-1", WIRE));
      stream.write(checkpointFrame());
    });
    try {
      const factory = makeFactory(server.baseUrl);
      const first = await runTurn(factory, baseRequest([userMessage("read d1")]));
            expect(first.some(m => m.type === "tool_call_end")).toBe(true);
      await waitFor(() => heldRunCount() === 1, "the run to park");
      expect(server.streams).toHaveLength(1);

      // The client returns the result; the model asks for a second call on the same stream.
      const secondRequest = baseRequest([userMessage("read d1"), assistantCall("call-1"), toolResult("call-1", "File d1 teal")]);
      const secondServerSide = (async () => {
        await waitFor(() => mcpResults(server).length === 1, "the first mcpResult");
        // Cursor reports the call it was just answered for as completed. The client already has it.
        server.streams[0]!.write(completedFrame("call-1", WIRE));
        server.streams[0]!.write(startedFrame("call-2", WIRE));
        server.streams[0]!.write(execFrame(2, "call-2", WIRE));
        server.streams[0]!.write(completedFrame("call-2", WIRE));
      })();
      const second = await runTurn(factory, secondRequest);
      await secondServerSide;
      // Only the new call reaches the client; the answered call-1 is not announced a second time.
      expect(second.filter(m => m.type === "tool_call_end").map(m => (m as { id?: string }).id)).toEqual(["call-2"]);
      // No second connection: the result went onto the stream that was already open.
      expect(server.streams).toHaveLength(1);
      const results = mcpResults(server);
      expect(results).toHaveLength(1);
      expect(results[0]!.id).toBe(1);
      const content = (results[0]!.message.value as { result: { value: { content: Array<{ content: { value: { text: string } } }> } } }).result.value.content[0]!.content.value.text;
      expect(content).toBe("File d1 teal");
      await waitFor(() => heldRunCount() === 1, "the run to park again");

      // The run parked a second time, so a third request carrying the second result claims it again.
      const thirdRequest = baseRequest([
        userMessage("read d1"), assistantCall("call-1"), toolResult("call-1", "File d1 teal"),
        assistantCall("call-2"), toolResult("call-2", "File d2 otter"),
      ]);
      const claim = claim_(thirdRequest);
      expect(claim).toBeDefined();
      claim!.dispose();
      expect(heldRunCount()).toBe(0);
    } finally {
      await server.close();
    }
  });

  test("an abort signal from an earlier request does not close a run that has parked again", async () => {
    const server = await startServer((stream, index) => {
      if (index !== 0) return;
      stream.write(startedFrame("call-1", WIRE));
      stream.write(execFrame(1, "call-1", WIRE));
      stream.write(completedFrame("call-1", WIRE));
    });
    try {
      const factory = makeFactory(server.baseUrl);
      const firstSignal = new AbortController();
      await runTurn(factory, baseRequest([userMessage("read d1")]), firstSignal.signal);
      await waitFor(() => heldRunCount() === 1, "the run to park");
      // The server layer aborts a request's signal after it finishes. That must not touch a parked run.
      firstSignal.abort();
      await new Promise(resolve => setTimeout(resolve, 60));
      expect(heldRunCount()).toBe(1);
      expect(server.streams[0]!.destroyed || server.streams[0]!.closed).toBe(false);
    } finally {
      await server.close();
    }
  });

  test("a second tool call arriving while parked drops the run, so the client's result is replayed", async () => {
    const server = await startServer((stream, index) => {
      if (index !== 0) return;
      stream.write(startedFrame("call-1", WIRE));
      stream.write(execFrame(1, "call-1", WIRE));
      stream.write(completedFrame("call-1", WIRE));
      stream.write(checkpointFrame());
      // A sibling the client was never told about, after the turn was parked.
      setTimeout(() => {
        stream.write(startedFrame("call-2", WIRE));
        stream.write(execFrame(2, "call-2", WIRE));
      }, 150);
    });
    try {
      await runTurn(makeFactory(server.baseUrl), baseRequest([userMessage("read d1")]));
      await waitFor(() => heldRunCount() === 1, "the run to park");
      await waitFor(() => heldRunCount() === 0, "the run to be dropped", 3000);
      expect(heldRunCount()).toBe(0);
      expect(mcpResults(server)).toHaveLength(0);
    } finally {
      await server.close();
    }
  });

  test("frames queued behind the one that dropped a parked run are not processed", async () => {
    let wrote: Uint8Array | undefined;
    const server = await startServer((stream, index) => {
      if (index !== 0) return;
      stream.write(startedFrame("call-1", WIRE));
      stream.write(execFrame(1, "call-1", WIRE));
      stream.write(completedFrame("call-1", WIRE));
      stream.write(checkpointFrame());
      // One TCP chunk carrying a foreign frame followed by a native exec request.
      setTimeout(() => {
        const nativeExec = agent({
          message: { case: "execServerMessage", value: create(ExecServerMessageSchema, { id: 9, execId: "native-1", message: { case: "lsArgs", value: {} as never } }) },
        });
        const joined = new Uint8Array(textFrame("surprise").byteLength + nativeExec.byteLength);
        joined.set(textFrame("surprise")); joined.set(nativeExec, textFrame("surprise").byteLength);
        wrote = joined;
        stream.write(joined);
      }, 200);
    });
    try {
      await runTurn(makeFactory(server.baseUrl), baseRequest([userMessage("read d1")]));
      await waitFor(() => heldRunCount() === 1, "the run to park");
      await waitFor(() => wrote !== undefined && heldRunCount() === 0, "the run to be dropped");
      await new Promise(resolve => setTimeout(resolve, 150));
      // No reply to the native exec was written back: the run was dead before that frame was handled.
      const replies = server.clientFrames[0]!.filter(frame => frame.message.case === "execClientMessage" && frame.message.value.id === 9);
      expect(replies).toHaveLength(0);
    } finally {
      await server.close();
    }
  });

  test("once a resume wrote the result, a failure is not retried and the request is flagged replay-unsafe", async () => {
    const server = await startServer((stream, index) => {
      if (index !== 0) return;
      stream.write(startedFrame("call-1", WIRE));
      stream.write(execFrame(1, "call-1", WIRE));
      stream.write(completedFrame("call-1", WIRE));
      stream.write(checkpointFrame());
      // After the client's result arrives the server resets the stream.
      stream.on("data", () => { if (mcpResults(server).length >= 1) setTimeout(() => stream.close(http2.constants.NGHTTP2_INTERNAL_ERROR), 20); });
    });
    try {
      const factory = makeFactory(server.baseUrl);
      await runTurn(factory, baseRequest([userMessage("read d1")]));
      await waitFor(() => heldRunCount() === 1, "the run to park");
      const seen: CursorServerMessage[] = [];
      let thrown: unknown;
      try {
        const budget = createTestTranslatorBudget();
        await runCursorTurnWithRetry(factory, { provider: { adapter: "cursor", baseUrl: "x", apiKey: "t" }, translatorBudget: budget, headers: new Headers() },
          baseRequest([userMessage("read d1"), assistantCall("call-1"), toolResult("call-1", "File d1 teal")]), undefined, m => { seen.push(m); });
      } catch (error) { thrown = error; }
      expect(thrown).toBeDefined();
      // The result went out exactly once; no second connection was opened to replay it.
      expect(mcpResults(server)).toHaveLength(1);
      expect(server.streams).toHaveLength(1);
      expect(seen.some(m => m.type === "local_side_effect")).toBe(true);
    } finally {
      await server.close();
    }
  });

  test("turnEnded or text arriving while parked drops the run", async () => {
    for (const unexpected of [turnEndedFrame(), textFrame("surprise")]) {
      resetHeldRunsForTests();
      const server = await startServer((stream, index) => {
        if (index !== 0) return;
        stream.write(startedFrame("call-1", WIRE));
        stream.write(execFrame(1, "call-1", WIRE));
        stream.write(completedFrame("call-1", WIRE));
        // The checkpoint ends the transport's checkpoint grace, so the run parks promptly.
        stream.write(checkpointFrame());
        setTimeout(() => stream.write(unexpected), 250);
      });
      try {
        await runTurn(makeFactory(server.baseUrl), baseRequest([userMessage("read d1")]));
        await waitFor(() => heldRunCount() === 1, "the run to park");
        await waitFor(() => heldRunCount() === 0, "the run to be dropped", 3000);
        expect(heldRunCount()).toBe(0);
      } finally {
        await server.close();
      }
    }
  });

  test("heartbeats and checkpoints while parked keep the run and do not renew its lease", async () => {
    setHeldRunLeaseMsForTests(300);
    const server = await startServer((stream, index) => {
      if (index !== 0) return;
      stream.write(startedFrame("call-1", WIRE));
      stream.write(execFrame(1, "call-1", WIRE));
      stream.write(completedFrame("call-1", WIRE));
      const timer = setInterval(() => stream.write(heartbeatFrame()), 40);
      stream.on("close", () => clearInterval(timer));
      setTimeout(() => stream.write(checkpointFrame()), 60);
    });
    try {
      await runTurn(makeFactory(server.baseUrl), baseRequest([userMessage("read d1")]));
      await waitFor(() => heldRunCount() === 1, "the run to park");
      await new Promise(resolve => setTimeout(resolve, 150));
      expect(heldRunCount()).toBe(1);
      await waitFor(() => heldRunCount() === 0, "the lease to expire");
    } finally {
      await server.close();
    }
  });

  test("a model outside Grok 4.6 and 4.7 keeps the old behaviour and never parks", async () => {
    const server = await startServer((stream, index) => {
      if (index !== 0) return;
      stream.write(startedFrame("call-1", WIRE));
      stream.write(execFrame(1, "call-1", WIRE));
      stream.write(completedFrame("call-1", WIRE));
    });
    try {
      await runTurn(makeFactory(server.baseUrl), baseRequest([userMessage("read d1")], { modelId: "claude-sonnet-5-5" }));
      await new Promise(resolve => setTimeout(resolve, 150));
      expect(heldRunCount()).toBe(0);
    } finally {
      await server.close();
    }
  });

  test("with the flag off nothing parks", async () => {
    delete process.env.OPENCODEX_CURSOR_HELD_STREAM;
    const server = await startServer((stream, index) => {
      if (index !== 0) return;
      stream.write(startedFrame("call-1", WIRE));
      stream.write(execFrame(1, "call-1", WIRE));
      stream.write(completedFrame("call-1", WIRE));
    });
    try {
      await runTurn(makeFactory(server.baseUrl), baseRequest([userMessage("read d1")]));
      await new Promise(resolve => setTimeout(resolve, 150));
      expect(heldRunCount()).toBe(0);
    } finally {
      await server.close();
    }
  });
});

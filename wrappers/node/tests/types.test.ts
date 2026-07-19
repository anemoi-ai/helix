/**
 * Unit tests for @helix/node — no native addon required.
 * Tests type parsing, error mapping, and OpenAI-compat parameter validation.
 */

import { describe, it, expect } from "vitest";

// ── Error class hierarchy ─────────────────────────────────────────────────────

describe("HelixError hierarchy", () => {
  it("wraps native errors with the correct subclass", async () => {
    // Import the module but stub the native loader so we don't need the .node file.
    // We test only the pure-JS logic: error classes, type shapes, compat validation.
    const { HelixError, HelixValidationError, HelixModelNotFoundError } = await import("../src/error-types.js");

    const e = new HelixValidationError("bad param", -3, '{"error":{"param":"messages"}}');
    expect(e).toBeInstanceOf(HelixError);
    expect(e).toBeInstanceOf(HelixValidationError);
    expect(e.helixStatus).toBe(-3);
    expect(e.param).toBe("messages");
    expect(e.name).toBe("HelixValidationError");
  });

  it("HelixModelNotFoundError has correct name", async () => {
    const { HelixModelNotFoundError } = await import("../src/error-types.js");
    const e = new HelixModelNotFoundError("no such model", -4, "{}");
    expect(e.name).toBe("HelixModelNotFoundError");
    expect(e.helixStatus).toBe(-4);
  });
});

// ── OpenAI-compat parameter validation ────────────────────────────────────────

describe("OpenAI-compat parameter validation", () => {
  it("rejects unsupported parameters", async () => {
    const { buildRequest } = await import("../src/compat-util.js");
    expect(() => buildRequest({ messages: [], audio: { voice: "alloy" } })).toThrow();
  });

  it("passes supported parameters through", async () => {
    const { buildRequest } = await import("../src/compat-util.js");
    const req = buildRequest({
      messages: [{ role: "user", content: "hi" }],
      model: "qwen-test",
      temperature: 0.7,
      max_tokens: 100,
    });
    expect(req.temperature).toBe(0.7);
    expect(req.max_tokens).toBe(100);
  });

  it("maps max_completion_tokens to max_tokens", async () => {
    const { buildRequest } = await import("../src/compat-util.js");
    const req = buildRequest({ messages: [], max_completion_tokens: 50 });
    expect((req as any).max_tokens).toBe(50);
    expect((req as any).max_completion_tokens).toBeUndefined();
  });

  it("silently drops null/undefined parameters", async () => {
    const { buildRequest } = await import("../src/compat-util.js");
    const req = buildRequest({ messages: [], temperature: undefined, top_p: null });
    expect((req as any).temperature).toBeUndefined();
    expect((req as any).top_p).toBeUndefined();
  });
});

// ── Response type shapes ──────────────────────────────────────────────────────

describe("response type shapes", () => {
  it("ChatCompletion JSON parses correctly", () => {
    const json = JSON.parse(`{
      "id":"chatcmpl-1","object":"chat.completion","created":0,"model":"m",
      "choices":[{"index":0,"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}],
      "usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}
    }`);
    expect(json.choices[0].message.content).toBe("pong");
    expect(json.usage.total_tokens).toBe(7);
  });

  it("ChatCompletionChunk JSON parses correctly", () => {
    const json = JSON.parse(`{
      "id":"x","object":"chat.completion.chunk","created":0,"model":"m",
      "choices":[{"index":0,"delta":{"role":"assistant","content":"po"},"finish_reason":null}]
    }`);
    expect(json.choices[0].delta.content).toBe("po");
    expect(json.choices[0].delta.role).toBe("assistant");
    expect(json.choices[0].finish_reason).toBeNull();
  });

  it("tool_calls response parses correctly", () => {
    const json = JSON.parse(`{
      "id":"x","object":"chat.completion","created":0,"model":"m",
      "choices":[{"index":0,"message":{"role":"assistant","content":null,
        "tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{}"}}]},
        "finish_reason":"tool_calls"}],
      "usage":{"prompt_tokens":5,"completion_tokens":10,"total_tokens":15}
    }`);
    expect(json.choices[0].finish_reason).toBe("tool_calls");
    expect(json.choices[0].message.tool_calls[0].function.name).toBe("get_weather");
  });
});

package helix_test

import (
	"encoding/json"
	"errors"
	"testing"

	helix "github.com/anemoi-ai/helix-go"
)

// Unit tests — no live library required; tests JSON round-trip and error types.

func TestChatRequestMarshal(t *testing.T) {
	temp := 0.7
	maxTok := 100
	req := helix.ChatRequest{
		Model:       "qwen-test",
		Messages:    []helix.Message{{Role: "user", Content: "hi"}},
		Temperature: &temp,
		MaxTokens:   &maxTok,
	}
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]interface{}
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	if m["model"] != "qwen-test" {
		t.Errorf("model: got %v", m["model"])
	}
	if m["temperature"] != 0.7 {
		t.Errorf("temperature: got %v", m["temperature"])
	}
}

func TestChatRequestOmitsNilFields(t *testing.T) {
	req := helix.ChatRequest{
		Model:    "m",
		Messages: []helix.Message{},
	}
	data, _ := json.Marshal(req)
	var m map[string]interface{}
	json.Unmarshal(data, &m)
	if _, ok := m["temperature"]; ok {
		t.Error("temperature should be omitted when nil")
	}
	if _, ok := m["tools"]; ok {
		t.Error("tools should be omitted when nil")
	}
}

func TestChatCompletionUnmarshal(t *testing.T) {
	raw := `{
		"id":"chatcmpl-1","object":"chat.completion","created":0,"model":"m",
		"choices":[{"index":0,"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}
	}`
	var cc helix.ChatCompletion
	if err := json.Unmarshal([]byte(raw), &cc); err != nil {
		t.Fatal(err)
	}
	if cc.Choices[0].Message.Content != "pong" {
		t.Errorf("content: got %q", cc.Choices[0].Message.Content)
	}
	if cc.Usage.TotalTokens != 7 {
		t.Errorf("total_tokens: got %d", cc.Usage.TotalTokens)
	}
}

func TestChunkUnmarshal(t *testing.T) {
	raw := `{
		"id":"x","object":"chat.completion.chunk","created":0,"model":"m",
		"choices":[{"index":0,"delta":{"role":"assistant","content":"po"},"finish_reason":null}]
	}`
	var chunk helix.ChatCompletionChunk
	if err := json.Unmarshal([]byte(raw), &chunk); err != nil {
		t.Fatal(err)
	}
	if chunk.Choices[0].Delta.Content != "po" {
		t.Errorf("delta.content: got %q", chunk.Choices[0].Delta.Content)
	}
}

func TestErrorSentinels(t *testing.T) {
	// Verify the sentinel errors are distinct.
	sentinels := []error{
		helix.ErrValidation, helix.ErrModelNotFound, helix.ErrModelLoadFailed,
		helix.ErrOom, helix.ErrVramExhausted, helix.ErrContextFull,
		helix.ErrCancelled, helix.ErrBackend, helix.ErrUnsupportedFeature,
	}
	for i, a := range sentinels {
		for j, b := range sentinels {
			if i != j && errors.Is(a, b) {
				t.Errorf("sentinel %d should not match sentinel %d", i, j)
			}
		}
	}
}

func TestMessageMarshal(t *testing.T) {
	msg := helix.Message{Role: "user", Content: "hello"}
	data, _ := json.Marshal(msg)
	var m map[string]interface{}
	json.Unmarshal(data, &m)
	if m["role"] != "user" {
		t.Errorf("role: got %v", m["role"])
	}
	if m["content"] != "hello" {
		t.Errorf("content: got %v", m["content"])
	}
}

func TestToolCallUnmarshal(t *testing.T) {
	raw := `{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{}"}}`
	var tc helix.ToolCall
	if err := json.Unmarshal([]byte(raw), &tc); err != nil {
		t.Fatal(err)
	}
	if tc.Function.Name != "get_weather" {
		t.Errorf("tool name: got %q", tc.Function.Name)
	}
}

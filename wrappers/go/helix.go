/*
Package helix provides Go bindings for libhelix — local LLM inference.

Basic usage:

	h, err := helix.New(helix.Options{})
	model, err := h.LoadModel("qwen.gguf", helix.ModelOptions{})
	session, err := model.NewSession(helix.SessionOptions{})

	resp, err := session.ChatCompletions(ctx, helix.ChatRequest{
	    Model:    "qwen-2.5",
	    Messages: []helix.Message{{Role: "user", Content: "hi"}},
	})
	fmt.Println(resp.Choices[0].Message.Content)

Streaming:

	ch, errCh := session.StreamChatCompletions(ctx, req)
	for chunk := range ch {
	    fmt.Print(chunk.Choices[0].Delta.Content)
	}
	if err := <-errCh; err != nil { log.Fatal(err) }
*/
package helix

/*
#cgo LDFLAGS: -lhelix
#include <helix.h>
#include <stdlib.h>

// Go export has char* but helix expects const char*. Bridge via a static shim.
extern int helixStreamBridge(void* userData, char* chunkJson);

static int helixStreamTrampoline(void* userData, const char* chunkJson) {
    return helixStreamBridge(userData, (char*)chunkJson);
}
*/
import "C"

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"
)

// ── Error types ───────────────────────────────────────────────────────────────

// HelixError is returned for all libhelix errors.
type HelixError struct {
	Status  int
	Message string
	Param   string
	Code    string
	Type    string
}

func (e *HelixError) Error() string {
	if e.Param != "" {
		return fmt.Sprintf("helix(%d): %s (param: %s)", e.Status, e.Message, e.Param)
	}
	return fmt.Sprintf("helix(%d): %s", e.Status, e.Message)
}

// Sentinel error types for type-assertion in callers.
var (
	ErrValidation          = errors.New("validation error")
	ErrModelNotFound       = errors.New("model not found")
	ErrModelLoadFailed     = errors.New("model load failed")
	ErrOom                 = errors.New("out of memory")
	ErrVramExhausted       = errors.New("VRAM exhausted")
	ErrContextFull         = errors.New("context full")
	ErrCancelled           = errors.New("cancelled")
	ErrBackend             = errors.New("backend error")
	ErrUnsupportedFeature  = errors.New("unsupported feature")
)

type errorBody struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Param   string `json:"param"`
		Code    string `json:"code"`
	} `json:"error"`
}

func lastError() *errorBody {
	raw := C.GoString(C.helix_last_error_json())
	var body errorBody
	_ = json.Unmarshal([]byte(raw), &body)
	return &body
}

func checkStatus(rc C.helix_status_t) error {
	if rc == C.HELIX_OK {
		return nil
	}
	eb := lastError()
	he := &HelixError{
		Status:  int(rc),
		Message: eb.Error.Message,
		Param:   eb.Error.Param,
		Code:    eb.Error.Code,
		Type:    eb.Error.Type,
	}
	switch int(rc) {
	case -3:
		return fmt.Errorf("%w: %s", ErrValidation, he.Message)
	case -4:
		return fmt.Errorf("%w: %s", ErrModelNotFound, he.Message)
	case -5:
		return fmt.Errorf("%w: %s", ErrModelLoadFailed, he.Message)
	case -6:
		return fmt.Errorf("%w: %s", ErrOom, he.Message)
	case -7:
		return fmt.Errorf("%w: %s", ErrVramExhausted, he.Message)
	case -8:
		return fmt.Errorf("%w: %s", ErrContextFull, he.Message)
	case -9:
		return ErrCancelled
	case -10:
		return fmt.Errorf("%w: %s", ErrBackend, he.Message)
	case -11:
		return fmt.Errorf("%w: %s", ErrUnsupportedFeature, he.Message)
	default:
		return he
	}
}

// ── Request / response types ──────────────────────────────────────────────────

// Message is a chat message.
type Message struct {
	Role             string        `json:"role"`
	Content          string        `json:"content,omitempty"`
	ToolCalls        []ToolCall    `json:"tool_calls,omitempty"`
	ToolCallID       string        `json:"tool_call_id,omitempty"`
	ReasoningContent string        `json:"reasoning_content,omitempty"`
}

// ToolCall describes a function call made by the model.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction is the function name + arguments within a ToolCall.
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool defines a tool available to the model.
type Tool struct {
	Type     string          `json:"type"`
	Function ToolDefinition  `json:"function"`
}

// ToolDefinition is the schema for a tool's function.
type ToolDefinition struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Parameters  interface{} `json:"parameters"`
}

// ResponseFormat controls output format (text, json_object, json_schema).
type ResponseFormat struct {
	Type       string      `json:"type"`
	JSONSchema interface{} `json:"json_schema,omitempty"`
}

// StreamOptions controls streaming behaviour.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

// ChatRequest is the chat completions request payload.
type ChatRequest struct {
	Model          string            `json:"model"`
	Messages       []Message         `json:"messages"`
	Temperature    *float64          `json:"temperature,omitempty"`
	TopP           *float64          `json:"top_p,omitempty"`
	MaxTokens      *int              `json:"max_tokens,omitempty"`
	Stop           []string          `json:"stop,omitempty"`
	Seed           *int64            `json:"seed,omitempty"`
	N              *int              `json:"n,omitempty"`
	Tools          []Tool            `json:"tools,omitempty"`
	ToolChoice     interface{}       `json:"tool_choice,omitempty"`
	ResponseFormat *ResponseFormat   `json:"response_format,omitempty"`
	Logprobs       *bool             `json:"logprobs,omitempty"`
	TopLogprobs    *int              `json:"top_logprobs,omitempty"`
	LogitBias      map[string]float32 `json:"logit_bias,omitempty"`
	StreamOptions  *StreamOptions    `json:"stream_options,omitempty"`
}

// CompletionTokensDetails holds reasoning token details.
type CompletionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// Usage is the token usage for a completion.
type Usage struct {
	PromptTokens            int                      `json:"prompt_tokens"`
	CompletionTokens        int                      `json:"completion_tokens"`
	TotalTokens             int                      `json:"total_tokens"`
	CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

// ResponseMessage is the model's reply.
type ResponseMessage struct {
	Role             string     `json:"role"`
	Content          string     `json:"content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
}

// Choice is one completion option.
type Choice struct {
	Index        int             `json:"index"`
	Message      ResponseMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

// ChatCompletion is the complete non-streaming response.
type ChatCompletion struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// Delta is a streaming content delta.
type Delta struct {
	Role             string     `json:"role,omitempty"`
	Content          string     `json:"content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
}

// ChunkChoice is one choice in a streaming chunk.
type ChunkChoice struct {
	Index        int    `json:"index"`
	Delta        Delta  `json:"delta"`
	FinishReason string `json:"finish_reason,omitempty"`
}

// ChatCompletionChunk is a single streaming chunk.
type ChatCompletionChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []ChunkChoice `json:"choices"`
	Usage   *Usage        `json:"usage,omitempty"`
}

// ── Streaming callback registry ───────────────────────────────────────────────
// cgo doesn't allow closures as C callbacks. We use a global registry
// (callbackID → channel) and a pre-declared C trampoline.

var (
	cbMu      sync.Mutex
	cbRegistry = make(map[uint64]chan<- string) // "" = sentinel for done
	cbCounter  atomic.Uint64
)

func registerCallback(ch chan<- string) uint64 {
	id := cbCounter.Add(1)
	cbMu.Lock()
	cbRegistry[id] = ch
	cbMu.Unlock()
	return id
}

func unregisterCallback(id uint64) {
	cbMu.Lock()
	delete(cbRegistry, id)
	cbMu.Unlock()
}

//export helixStreamBridge
func helixStreamBridge(userData unsafe.Pointer, chunkJson *C.char) C.int {
	id := *(*uint64)(userData)
	cbMu.Lock()
	ch, ok := cbRegistry[id]
	cbMu.Unlock()
	if !ok {
		return 1 // cancel
	}
	if chunkJson == nil {
		ch <- "" // done sentinel
		return 0
	}
	ch <- C.GoString(chunkJson)
	return 0
}

// ── Session ───────────────────────────────────────────────────────────────────

// Session owns a KV cache and can run one request at a time.
type Session struct {
	ptr *C.helix_session_t
}

// ChatCompletions runs a synchronous chat completion.
func (s *Session) ChatCompletions(ctx context.Context, req ChatRequest) (*ChatCompletion, error) {
	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	cs := C.CString(string(reqJSON))
	defer C.free(unsafe.Pointer(cs))

	var out *C.char
	rc := C.helix_chat_completions(s.ptr, cs, &out)
	if err := checkStatus(rc); err != nil {
		return nil, err
	}
	defer C.helix_free(out)

	var result ChatCompletion
	if err := json.Unmarshal([]byte(C.GoString(out)), &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// StreamChatCompletions returns a channel of chunks and an error channel.
// Cancel via the context; drain errCh after the chunk channel closes.
func (s *Session) StreamChatCompletions(ctx context.Context, req ChatRequest) (<-chan ChatCompletionChunk, <-chan error) {
	chunkCh := make(chan ChatCompletionChunk, 32)
	errCh := make(chan error, 1)

	reqJSON, err := json.Marshal(req)
	if err != nil {
		close(chunkCh)
		errCh <- err
		close(errCh)
		return chunkCh, errCh
	}

	rawCh := make(chan string, 64)
	id := registerCallback(rawCh)

	go func() {
		defer close(chunkCh)
		defer func() { close(errCh) }()
		defer unregisterCallback(id)

		// Watch for context cancellation in a separate goroutine.
		cancelDone := make(chan struct{})
		go func() {
			defer close(cancelDone)
			select {
			case <-ctx.Done():
				C.helix_session_cancel(s.ptr)
			case <-cancelDone:
			}
		}()
		// cancelDone is closed when we return, which unblocks the goroutine above.
		defer func() { cancelDone <- struct{}{} }()

		cs := C.CString(string(reqJSON))
		defer C.free(unsafe.Pointer(cs))

		// helix_chat_completions_stream blocks until done; run in a goroutine
		// so we can concurrently read from rawCh.
		streamDone := make(chan C.helix_status_t, 1)
		idVal := id
		go func() {
			rc := C.helix_chat_completions_stream(
				s.ptr, cs,
				(C.helix_stream_cb)(C.helixStreamTrampoline),
				unsafe.Pointer(&idVal),
			)
			streamDone <- rc
		}()

		for {
			select {
			case raw, ok := <-rawCh:
				if !ok || raw == "" {
					// Wait for the stream goroutine to finish.
					rc := <-streamDone
					if rc != C.HELIX_OK && rc != C.HELIX_E_CANCELLED {
						errCh <- checkStatus(rc)
					}
					return
				}
				var chunk ChatCompletionChunk
				if err := json.Unmarshal([]byte(raw), &chunk); err == nil {
					chunkCh <- chunk
				}
			}
		}
	}()

	return chunkCh, errCh
}

// Cancel signals cooperative cancellation of any in-flight request.
func (s *Session) Cancel() {
	C.helix_session_cancel(s.ptr)
}

// Close destroys the session and frees its KV cache.
func (s *Session) Close() {
	if s.ptr != nil {
		C.helix_session_destroy(s.ptr)
		s.ptr = nil
	}
}

// ── Model ─────────────────────────────────────────────────────────────────────

// ModelOptions configure model loading.
type ModelOptions struct {
	Alias           string `json:"alias,omitempty"`
	NGpuLayers      int    `json:"n_gpu_layers,omitempty"`
	NCtx            int    `json:"n_ctx,omitempty"`
	MmprojectPath   string `json:"mmproj_path,omitempty"`
	ReasoningFormat string `json:"reasoning_format,omitempty"`
}

// Model is a loaded model, reference-counted across sessions.
type Model struct {
	ptr *C.helix_model_t
}

// SessionOptions configure session creation.
type SessionOptions struct{}

// NewSession creates a new inference session on this model.
func (m *Model) NewSession(options SessionOptions) (*Session, error) {
	cs := C.CString("{}")
	defer C.free(unsafe.Pointer(cs))

	var out *C.helix_session_t
	rc := C.helix_session_create(m.ptr, cs, &out)
	if err := checkStatus(rc); err != nil {
		return nil, err
	}
	return &Session{ptr: out}, nil
}

// Close releases the model reference.
func (m *Model) Close() {
	if m.ptr != nil {
		C.helix_model_release(m.ptr)
		m.ptr = nil
	}
}

// ── Helix (runtime) ───────────────────────────────────────────────────────────

// Options configure the Helix runtime.
type Options struct {
	LogLevel int `json:"log_level,omitempty"`
}

// Helix is the process-wide runtime. Create exactly one per process.
type Helix struct {
	ptr *C.helix_runtime_t
}

// New initialises the Helix runtime.
func New(options Options) (*Helix, error) {
	optsJSON, err := json.Marshal(options)
	if err != nil {
		return nil, err
	}
	cs := C.CString(string(optsJSON))
	defer C.free(unsafe.Pointer(cs))

	var out *C.helix_runtime_t
	rc := C.helix_runtime_create(cs, &out)
	if err := checkStatus(rc); err != nil {
		return nil, err
	}
	return &Helix{ptr: out}, nil
}

// LoadModel loads a model file. This call may take seconds to minutes.
func (h *Helix) LoadModel(modelPath string, options ModelOptions) (*Model, error) {
	type modelJSON struct {
		ModelPath string `json:"model_path"`
		ModelOptions
	}
	payload := modelJSON{ModelPath: modelPath, ModelOptions: options}
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	cs := C.CString(string(data))
	defer C.free(unsafe.Pointer(cs))

	var out *C.helix_model_t
	rc := C.helix_model_load(h.ptr, cs, &out)
	if err := checkStatus(rc); err != nil {
		return nil, err
	}
	return &Model{ptr: out}, nil
}

// AbiVersion returns the libhelix ABI version.
func (h *Helix) AbiVersion() uint32 { return uint32(C.helix_abi_version()) }

// VersionString returns the libhelix version string.
func (h *Helix) VersionString() string { return C.GoString(C.helix_version_string()) }

// Close destroys the runtime.
func (h *Helix) Close() {
	if h.ptr != nil {
		C.helix_runtime_destroy(h.ptr)
		h.ptr = nil
	}
}

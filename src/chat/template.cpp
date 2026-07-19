#include "template.hpp"
#include "../internal/error.hpp"
#include "../json/request.hpp"

#include "chat.h"
#include "common.h"

#include <string_view>
#include <variant>

namespace helix {

/* ------------------------------------------------------------------ */
/*  Conversion helpers: helix types → llama.cpp types                 */
/* ------------------------------------------------------------------ */

static common_chat_tool to_common_tool(const Tool& t) {
    common_chat_tool ct;
    ct.name        = t.function.name;
    ct.description = t.function.description;
    ct.parameters  = t.function.parameters.empty() ? "{}" : t.function.parameters;
    return ct;
}

static common_chat_tool_choice to_common_tool_choice(
    const std::optional<ToolChoice>& tool_choice,
    const std::vector<Tool>& /*tools*/) {

    if (!tool_choice) return COMMON_CHAT_TOOL_CHOICE_AUTO;

    return std::visit([](const auto& v) -> common_chat_tool_choice {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) {
            if (v == "none")     return COMMON_CHAT_TOOL_CHOICE_NONE;
            if (v == "required") return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
            return COMMON_CHAT_TOOL_CHOICE_AUTO; /* "auto" or unknown */
        } else {
            /* Specific function: map to REQUIRED; caller filters tools list. */
            return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        }
    }, *tool_choice);
}

/* When tool_choice is {type:"function", name:"X"}, restrict to that function only. */
static std::vector<Tool> filter_tools(const std::vector<Tool>& tools,
                                       const std::optional<ToolChoice>& tool_choice) {
    if (!tool_choice) return tools;
    if (const auto* tcf = std::get_if<ToolChoiceFunction>(&*tool_choice)) {
        for (const auto& t : tools) {
            if (t.function.name == tcf->name) return {t};
        }
        return tools; /* name not found: return all and let model decide */
    }
    return tools;
}

/* ------------------------------------------------------------------ */
/*  render_prompt                                                       */
/* ------------------------------------------------------------------ */

RenderResult render_prompt(const std::vector<Message>& messages,
                            const std::vector<std::string>& request_stop,
                            const common_chat_templates* tmpls,
                            const std::vector<Tool>& tools,
                            const std::optional<ToolChoice>& tool_choice,
                            bool parallel_tool_calls,
                            bool enable_thinking,
                            const std::string& reasoning_format) {
    const bool use_tools = !tools.empty() &&
        !(tool_choice && std::holds_alternative<std::string>(*tool_choice) &&
          std::get<std::string>(*tool_choice) == "none");

    /* Build the effective tools list (may be filtered for specific-function choice). */
    std::vector<Tool> active_tools = use_tools ? filter_tools(tools, tool_choice) : std::vector<Tool>{};

    /* Convert helix::Message → common_chat_msg */
    std::vector<common_chat_msg> chat_msgs;
    chat_msgs.reserve(messages.size());

    for (const auto& m : messages) {
        common_chat_msg cm;
        cm.role = (m.role == "developer") ? "system" : m.role;

        if (!m.content_parts.empty()) {
            for (const auto& part : m.content_parts) {
                common_chat_msg_content_part cp;
                if (part.type == "text") {
                    cp.type = "text";
                    cp.text = part.text;
                } else {
                    /* image_url and input_audio become media_marker placeholders;
                     * the actual bytes are passed separately to eval_media. */
                    cp.type = "media_marker";
                    /* mtmd_tokenize splits the prompt on this marker string. */
                    cp.text = "<__media__>";
                }
                cm.content_parts.push_back(std::move(cp));
            }
        } else {
            cm.content = m.content_str;
        }

        if (m.role == "tool") {
            cm.tool_call_id = m.tool_call_id;
        }

        if (m.role == "assistant" && !m.tool_calls.empty()) {
            for (const auto& tc : m.tool_calls) {
                common_chat_tool_call ctc;
                ctc.name      = tc.function_name;
                ctc.arguments = tc.function_arguments;
                ctc.id        = tc.id;
                cm.tool_calls.push_back(std::move(ctc));
            }
        }

        if (m.role == "assistant" && m.reasoning_content) {
            cm.reasoning_content = *m.reasoning_content;
        }

        chat_msgs.push_back(std::move(cm));
    }

    /* Map helix reasoning_format string to llama.cpp enum. */
    common_reasoning_format rf_enum = COMMON_REASONING_FORMAT_NONE;
    if (reasoning_format == "auto") {
        rf_enum = COMMON_REASONING_FORMAT_AUTO;
    } else if (reasoning_format == "deepseek-r1" || reasoning_format == "deepseek") {
        rf_enum = COMMON_REASONING_FORMAT_DEEPSEEK;
    } else if (reasoning_format == "qwq") {
        /* QwQ emits the same <think>...</think> convention as DeepSeek-R1, which
         * is exactly what our local extractor (src/chat/reasoning.cpp) keys on,
         * so DEEPSEEK is the correct mapping today regardless of which side does
         * the split. Coupling risk: if a future llama.cpp pin either changes the
         * DEEPSEEK tag convention or introduces a distinct COMMON_REASONING_FORMAT_QWQ,
         * this mapping AND reasoning.cpp's tags must be revisited together. The
         * llama.cpp pin is frozen per release, so this cannot drift silently
         * without a deliberate pin bump (see third_party/llama.cpp.commit). */
        rf_enum = COMMON_REASONING_FORMAT_DEEPSEEK;
    }
    /* "none" stays COMMON_REASONING_FORMAT_NONE */

    common_chat_templates_inputs inputs;
    inputs.messages              = chat_msgs;
    inputs.add_generation_prompt = true;
    inputs.use_jinja             = true;
    inputs.parallel_tool_calls   = parallel_tool_calls;
    inputs.enable_thinking       = enable_thinking;
    inputs.reasoning_format      = rf_enum;

    if (use_tools) {
        inputs.tool_choice = to_common_tool_choice(tool_choice, active_tools);
        for (const auto& t : active_tools) {
            inputs.tools.push_back(to_common_tool(t));
        }
    }

    common_chat_params params = common_chat_templates_apply(tmpls, inputs);

    RenderResult result;
    result.prompt            = params.prompt;
    result.supports_thinking = params.supports_thinking;
    result.thinking_start_tag = params.thinking_start_tag;
    result.thinking_end_tag  = params.thinking_end_tag;
    result.extract_reasoning = (rf_enum != COMMON_REASONING_FORMAT_NONE);

    /* Some templates (Qwen3 thinking mode) pre-open the think block in the
     * prompt, so the model's output starts mid-thought and contains only the
     * closing tag. Detect this by checking whether the rendered prompt ends
     * with the opening tag. Use the template-supplied start tag when there is
     * one — the budget sampler watches for that same string, and the two must
     * agree. Fall back to deriving it from the end tag (drop the '/':
     * "</think>" -> "<think>", "[/THINK]" -> "[THINK]"). */
    if (result.extract_reasoning && result.supports_thinking) {
        std::string open_tag = result.thinking_start_tag;
        if (open_tag.empty()) {
            open_tag = result.thinking_end_tag.empty()
                       ? std::string("<think>")
                       : result.thinking_end_tag;
            const auto slash = open_tag.find('/');
            if (slash != std::string::npos) {
                open_tag.erase(slash, 1);
            }
        }
        std::string_view tail(result.prompt);
        while (!tail.empty() &&
               (tail.back() == '\n' || tail.back() == '\r' ||
                tail.back() == ' '  || tail.back() == '\t')) {
            tail.remove_suffix(1);
        }
        if (tail.size() >= open_tag.size() &&
            tail.substr(tail.size() - open_tag.size()) == open_tag) {
            result.thinking_forced_open = true;
        }
    }

    /* Merge stop strings: template-suggested + request-supplied. */
    for (const auto& s : params.additional_stops) result.stop_strings.push_back(s);
    for (const auto& s : request_stop)            result.stop_strings.push_back(s);

    /* Copy grammar / parser fields when tools are active. */
    if (use_tools && !params.grammar.empty()) {
        result.grammar         = params.grammar;
        result.grammar_lazy    = params.grammar_lazy;
        result.parser_data     = params.parser;
        result.generation_prompt = params.generation_prompt;
        result.chat_format     = static_cast<int>(params.format);
        result.has_tools       = true;

        for (const auto& trig : params.grammar_triggers) {
            GrammarTrigger gt;
            gt.type  = static_cast<int>(trig.type);
            gt.value = trig.value;
            gt.token = static_cast<int32_t>(trig.token);
            result.grammar_triggers.push_back(gt);
        }
    } else if (use_tools && !params.parser.empty()) {
        /* Parser without explicit grammar (autoparser-only path). */
        result.parser_data       = params.parser;
        result.generation_prompt = params.generation_prompt;
        result.chat_format       = static_cast<int>(params.format);
        result.has_tools         = true;
    }

    return result;
}

} // namespace helix

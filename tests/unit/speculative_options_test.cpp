#include <gtest/gtest.h>
#include "src/session/session.hpp"
#include "src/internal/error.hpp"

using namespace helix;

/* parse_session_options is a free function declared in session.hpp and
 * defined in src/session/options.cpp (a minimal TU so it can be unit-tested
 * without dragging in the full Session class and its model/runtime deps). */

TEST(SpeculativeOptions, DefaultsToNone) {
    auto opts = parse_session_options(nullptr);
    EXPECT_EQ(opts.speculative.type, SpeculativeType::None);

    opts = parse_session_options("");
    EXPECT_EQ(opts.speculative.type, SpeculativeType::None);

    opts = parse_session_options("{}");
    EXPECT_EQ(opts.speculative.type, SpeculativeType::None);
    EXPECT_EQ(opts.speculative.n_max, 3);
    EXPECT_EQ(opts.speculative.n_min, 0);
    EXPECT_FLOAT_EQ(opts.speculative.p_min, 0.0f);
    EXPECT_EQ(opts.speculative.cache_type_k, "f16");
    EXPECT_EQ(opts.speculative.cache_type_v, "f16");
    EXPECT_TRUE(opts.speculative.backend_sampling);
    EXPECT_TRUE(opts.speculative.draft_model_path.empty());
}

TEST(SpeculativeOptions, ExplicitNoneIsNone) {
    auto opts = parse_session_options(R"({"speculative":{"type":"none"}})");
    EXPECT_EQ(opts.speculative.type, SpeculativeType::None);
}

TEST(SpeculativeOptions, NullTypeIsNone) {
    auto opts = parse_session_options(R"({"speculative":{"type":null}})");
    EXPECT_EQ(opts.speculative.type, SpeculativeType::None);
}

TEST(SpeculativeOptions, DraftMtpTypeParsed) {
    auto opts = parse_session_options(R"({"speculative":{"type":"draft-mtp"}})");
    EXPECT_EQ(opts.speculative.type, SpeculativeType::DraftMtp);
}

TEST(SpeculativeOptions, FullObjectParsed) {
    auto opts = parse_session_options(R"({
        "speculative": {
            "type": "draft-mtp",
            "model_path": "/models/draft.gguf",
            "n_max": 5,
            "n_min": 2,
            "p_min": 0.25,
            "backend_sampling": false,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0"
        }
    })");
    EXPECT_EQ(opts.speculative.type, SpeculativeType::DraftMtp);
    EXPECT_EQ(opts.speculative.draft_model_path, "/models/draft.gguf");
    EXPECT_EQ(opts.speculative.n_max, 5);
    EXPECT_EQ(opts.speculative.n_min, 2);
    EXPECT_FLOAT_EQ(opts.speculative.p_min, 0.25f);
    EXPECT_FALSE(opts.speculative.backend_sampling);
    EXPECT_EQ(opts.speculative.cache_type_k, "q8_0");
    EXPECT_EQ(opts.speculative.cache_type_v, "q8_0");
}

TEST(SpeculativeOptions, InvalidTypeThrows) {
    EXPECT_THROW(parse_session_options(R"({"speculative":{"type":"bogus"}})"),
                 helix::Error);
}

TEST(SpeculativeOptions, NonStringTypeThrows) {
    EXPECT_THROW(parse_session_options(R"({"speculative":{"type":123}})"),
                 helix::Error);
}

TEST(SpeculativeOptions, NonObjectSpeculativeThrows) {
    EXPECT_THROW(parse_session_options(R"({"speculative":"draft-mtp"})"),
                 helix::Error);
    EXPECT_THROW(parse_session_options(R"({"speculative":[1,2]})"),
                 helix::Error);
}

TEST(SpeculativeOptions, NegativeNMaxThrows) {
    EXPECT_THROW(parse_session_options(R"({"speculative":{"type":"draft-mtp","n_max":-1}})"),
                 helix::Error);
}

TEST(SpeculativeOptions, ZeroNMaxOk) {
    auto opts = parse_session_options(R"({"speculative":{"type":"draft-mtp","n_max":0}})");
    EXPECT_EQ(opts.speculative.n_max, 0);
}

TEST(SpeculativeOptions, NegativeNMinThrows) {
    EXPECT_THROW(parse_session_options(R"({"speculative":{"type":"draft-mtp","n_min":-1}})"),
                 helix::Error);
}

TEST(SpeculativeOptions, PMinOutOfRangeThrows) {
    EXPECT_THROW(parse_session_options(R"({"speculative":{"p_min":-0.1}})"),
                 helix::Error);
    EXPECT_THROW(parse_session_options(R"({"speculative":{"p_min":1.5}})"),
                 helix::Error);
}

TEST(SpeculativeOptions, PMinBoundaryOk) {
    auto opts = parse_session_options(R"({"speculative":{"p_min":0.0}})");
    EXPECT_FLOAT_EQ(opts.speculative.p_min, 0.0f);

    opts = parse_session_options(R"({"speculative":{"p_min":1.0}})");
    EXPECT_FLOAT_EQ(opts.speculative.p_min, 1.0f);
}

TEST(SpeculativeOptions, NonStringCacheTypeThrows) {
    EXPECT_THROW(parse_session_options(R"({"speculative":{"cache_type_k":123}})"),
                 helix::Error);
    EXPECT_THROW(parse_session_options(R"({"speculative":{"cache_type_v":false}})"),
                 helix::Error);
}

TEST(SpeculativeOptions, NonBooleanBackendSamplingThrows) {
    EXPECT_THROW(parse_session_options(R"({"speculative":{"backend_sampling":"yes"}})"),
                 helix::Error);
}

TEST(SpeculativeOptions, DoesNotAffectOtherSessionFields) {
    /* Speculative options coexist with the standard session fields. */
    auto opts = parse_session_options(R"({
        "n_ctx": 4096,
        "n_batch": 512,
        "seed": 42,
        "speculative": {"type": "draft-mtp", "n_max": 4}
    })");
    EXPECT_EQ(opts.n_ctx, 4096);
    EXPECT_EQ(opts.n_batch, 512);
    EXPECT_EQ(opts.seed, 42u);
    EXPECT_EQ(opts.speculative.type, SpeculativeType::DraftMtp);
    EXPECT_EQ(opts.speculative.n_max, 4);
}

TEST(SwaFullOption, DefaultsToTrue) {
    auto opts = parse_session_options("{}");
    EXPECT_TRUE(opts.swa_full);
}

TEST(SwaFullOption, FalseParsed) {
    auto opts = parse_session_options(R"({"swa_full": false})");
    EXPECT_FALSE(opts.swa_full);
}

TEST(SwaFullOption, TrueParsed) {
    auto opts = parse_session_options(R"({"swa_full": true})");
    EXPECT_TRUE(opts.swa_full);
}

TEST(SwaFullOption, NonBooleanThrows) {
    /* Present-but-mistyped options are errors, matching the request parser. */
    EXPECT_THROW(parse_session_options(R"({"swa_full": "yes"})"), helix::Error);
}

/* ---- strict typing for the simple session options ---- */

TEST(SessionOptionsParse, StringNCtxThrows) {
    EXPECT_THROW(parse_session_options(R"({"n_ctx": "4096"})"), helix::Error);
}

TEST(SessionOptionsParse, StringSeedThrows) {
    EXPECT_THROW(parse_session_options(R"({"seed": "42"})"), helix::Error);
}

TEST(SessionOptionsParse, NonBooleanPrefixCacheThrows) {
    EXPECT_THROW(parse_session_options(R"({"prefix_cache": 1})"), helix::Error);
}

TEST(SessionOptionsParse, NullFieldTreatedAsAbsent) {
    auto opts = parse_session_options(R"({"n_ctx": null, "prefix_cache": null})");
    EXPECT_EQ(opts.n_ctx, 0);
    EXPECT_TRUE(opts.prefix_cache);
}

/* ---- main-context KV cache types (1.4) ---- */

TEST(CacheTypeOptions, DefaultsToF16) {
    auto opts = parse_session_options("{}");
    EXPECT_EQ(opts.cache_type_k, "f16");
    EXPECT_EQ(opts.cache_type_v, "f16");
}

TEST(CacheTypeOptions, TopLevelParsed) {
    auto opts = parse_session_options(
        R"({"cache_type_k": "q8_0", "cache_type_v": "q8_0"})");
    EXPECT_EQ(opts.cache_type_k, "q8_0");
    EXPECT_EQ(opts.cache_type_v, "q8_0");
}

TEST(CacheTypeOptions, IndependentOfSpeculativePair) {
    /* Top-level types configure the main context; speculative{} keeps its
     * own pair for the MTP draft context. */
    auto opts = parse_session_options(R"({
        "cache_type_k": "q8_0",
        "speculative": {"cache_type_k": "q4_0"}
    })");
    EXPECT_EQ(opts.cache_type_k, "q8_0");
    EXPECT_EQ(opts.cache_type_v, "f16");
    EXPECT_EQ(opts.speculative.cache_type_k, "q4_0");
    EXPECT_EQ(opts.speculative.cache_type_v, "f16");
}

TEST(CacheTypeOptions, NonStringThrows) {
    EXPECT_THROW(parse_session_options(R"({"cache_type_k": 8})"), helix::Error);
    EXPECT_THROW(parse_session_options(R"({"cache_type_v": true})"), helix::Error);
}

TEST(CacheTypeOptions, NullTreatedAsAbsent) {
    auto opts = parse_session_options(
        R"({"cache_type_k": null, "cache_type_v": null})");
    EXPECT_EQ(opts.cache_type_k, "f16");
    EXPECT_EQ(opts.cache_type_v, "f16");
}

/* NOTE: value-set validation ("q8_0" ok, "q9_9" rejected) happens at session
 * creation (parse_cache_type in session.cpp), which needs a live model — it
 * is covered by the CacheType integration tests. */

/* ---- context_shift: documented since 1.1, implemented in 1.6 ---- */

TEST(SessionOptionsParse, ContextShiftDefaultsToFalse) {
    auto opts = parse_session_options("{}");
    EXPECT_FALSE(opts.context_shift);
}

TEST(SessionOptionsParse, ContextShiftTrueParsed) {
    /* Rejected as unimplemented in 1.3–1.5; a working option since 1.6.
     * Capability checks (embedding/MTP exclusion, llama_memory_can_shift)
     * live at session creation and are covered by integration tests. */
    auto opts = parse_session_options(R"({"context_shift": true})");
    EXPECT_TRUE(opts.context_shift);
}

TEST(SessionOptionsParse, ContextShiftFalseParsed) {
    auto opts = parse_session_options(R"({"context_shift": false})");
    EXPECT_FALSE(opts.context_shift);
}

TEST(SessionOptionsParse, ContextShiftNonBooleanThrows) {
    EXPECT_THROW(parse_session_options(R"({"context_shift": "on"})"), helix::Error);
}

/* ---- lora: session-level adapter activation (1.6, F6) ---- */

TEST(LoraSessionOptions, AbsentByDefault) {
    auto opts = parse_session_options("{}");
    EXPECT_FALSE(opts.lora.has_value());
}

TEST(LoraSessionOptions, NullTreatedAsAbsent) {
    auto opts = parse_session_options(R"({"lora": null})");
    EXPECT_FALSE(opts.lora.has_value());
}

TEST(LoraSessionOptions, EmptyArrayIsExplicitNone) {
    /* [] must stay distinguishable from an absent option: it deactivates
     * every adapter, while absent activates all of them. */
    auto opts = parse_session_options(R"({"lora": []})");
    ASSERT_TRUE(opts.lora.has_value());
    EXPECT_TRUE(opts.lora->empty());
}

TEST(LoraSessionOptions, EntriesParsed) {
    auto opts = parse_session_options(
        R"({"lora": [{"name":"support","scale":0.8}, {"name":"tone"}]})");
    ASSERT_TRUE(opts.lora.has_value());
    ASSERT_EQ(opts.lora->size(), 2u);
    EXPECT_EQ((*opts.lora)[0].name, "support");
    ASSERT_TRUE((*opts.lora)[0].scale.has_value());
    EXPECT_FLOAT_EQ(*(*opts.lora)[0].scale, 0.8f);
    EXPECT_EQ((*opts.lora)[1].name, "tone");
    EXPECT_FALSE((*opts.lora)[1].scale.has_value());
}

TEST(LoraSessionOptions, NotArrayThrows) {
    EXPECT_THROW(parse_session_options(R"({"lora": "support"})"), helix::Error);
    EXPECT_THROW(parse_session_options(R"({"lora": {"name":"support"}})"),
                 helix::Error);
}

TEST(LoraSessionOptions, EntryNotObjectThrows) {
    EXPECT_THROW(parse_session_options(R"({"lora": ["support"]})"), helix::Error);
}

TEST(LoraSessionOptions, MissingOrEmptyNameThrows) {
    EXPECT_THROW(parse_session_options(R"({"lora": [{"scale":1.0}]})"),
                 helix::Error);
    EXPECT_THROW(parse_session_options(R"({"lora": [{"name":""}]})"),
                 helix::Error);
    EXPECT_THROW(parse_session_options(R"({"lora": [{"name":42}]})"),
                 helix::Error);
}

TEST(LoraSessionOptions, NonNumberScaleThrows) {
    EXPECT_THROW(
        parse_session_options(R"({"lora": [{"name":"a","scale":"1.0"}]})"),
        helix::Error);
}

TEST(LoraSessionOptions, DuplicateNameThrows) {
    EXPECT_THROW(parse_session_options(
                     R"({"lora": [{"name":"a"}, {"name":"a","scale":0.5}]})"),
                 helix::Error);
}

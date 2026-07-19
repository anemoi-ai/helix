#include <gtest/gtest.h>
#include "src/json/request.hpp"
#include "src/internal/error.hpp"

using namespace helix;

namespace {

RerankRequest parse(const std::string& body) {
    RerankRequest r = RerankRequest::from_json(body);
    r.validate();
    return r;
}

} // namespace

TEST(RerankRequest, MinimalParsed) {
    auto r = parse(R"({"model":"m","query":"q","documents":["a","b"]})");
    EXPECT_EQ(r.model, "m");
    EXPECT_EQ(r.query, "q");
    ASSERT_EQ(r.documents.size(), 2u);
    EXPECT_EQ(r.documents[0], "a");
    EXPECT_EQ(r.documents[1], "b");
    EXPECT_EQ(r.top_n, 0) << "unset top_n means return all";
}

TEST(RerankRequest, TopNParsed) {
    auto r = parse(R"({"model":"m","query":"q","documents":["a"],"top_n":5})");
    EXPECT_EQ(r.top_n, 5);
}

TEST(RerankRequest, NullTopNTreatedAsAbsent) {
    auto r = parse(R"({"model":"m","query":"q","documents":["a"],"top_n":null})");
    EXPECT_EQ(r.top_n, 0);
}

TEST(RerankRequest, UnknownFieldsIgnored) {
    EXPECT_NO_THROW(parse(
        R"({"model":"m","query":"q","documents":["a"],"return_text":true})"));
}

TEST(RerankRequest, MissingModelThrows) {
    EXPECT_THROW(parse(R"({"query":"q","documents":["a"]})"), helix::Error);
}

TEST(RerankRequest, MissingQueryThrows) {
    EXPECT_THROW(parse(R"({"model":"m","documents":["a"]})"), helix::Error);
}

TEST(RerankRequest, MissingDocumentsThrows) {
    EXPECT_THROW(parse(R"({"model":"m","query":"q"})"), helix::Error);
}

TEST(RerankRequest, EmptyDocumentsThrows) {
    EXPECT_THROW(parse(R"({"model":"m","query":"q","documents":[]})"),
                 helix::Error);
}

TEST(RerankRequest, NonArrayDocumentsThrows) {
    EXPECT_THROW(parse(R"({"model":"m","query":"q","documents":"a"})"),
                 helix::Error);
}

TEST(RerankRequest, NonStringDocumentElementThrows) {
    EXPECT_THROW(parse(R"({"model":"m","query":"q","documents":["a",1]})"),
                 helix::Error);
}

TEST(RerankRequest, EmptyStringDocumentThrows) {
    EXPECT_THROW(parse(R"({"model":"m","query":"q","documents":["a",""]})"),
                 helix::Error);
}

TEST(RerankRequest, EmptyQueryThrows) {
    EXPECT_THROW(parse(R"({"model":"m","query":"","documents":["a"]})"),
                 helix::Error);
}

TEST(RerankRequest, TopNZeroThrows) {
    EXPECT_THROW(parse(R"({"model":"m","query":"q","documents":["a"],"top_n":0})"),
                 helix::Error);
    EXPECT_THROW(parse(R"({"model":"m","query":"q","documents":["a"],"top_n":-1})"),
                 helix::Error);
}

TEST(RerankRequest, NonIntegerTopNThrows) {
    EXPECT_THROW(parse(R"({"model":"m","query":"q","documents":["a"],"top_n":"5"})"),
                 helix::Error);
}

TEST(RerankRequest, TooManyDocumentsThrows) {
    std::string body = R"({"model":"m","query":"q","documents":[)";
    for (int i = 0; i < 1025; ++i) {
        if (i) body += ',';
        body += "\"d\"";
    }
    body += "]}";
    EXPECT_THROW(parse(body), helix::Error);
}

TEST(RerankRequest, MaxDocumentsAccepted) {
    std::string body = R"({"model":"m","query":"q","documents":[)";
    for (int i = 0; i < 1024; ++i) {
        if (i) body += ',';
        body += "\"d\"";
    }
    body += "]}";
    auto r = parse(body);
    EXPECT_EQ(r.documents.size(), 1024u);
}

TEST(RerankRequest, MalformedJsonThrows) {
    EXPECT_THROW(RerankRequest::from_json("{nope"), helix::Error);
    EXPECT_THROW(RerankRequest::from_json("[]"), helix::Error);
}

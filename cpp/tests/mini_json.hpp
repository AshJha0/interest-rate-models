#ifndef IRM_TESTS_MINI_JSON_HPP
#define IRM_TESTS_MINI_JSON_HPP

// Minimal self-contained JSON reader for the flat golden-value schema:
//   { "cases": [ {"name": "...", "inputs": {...}, "expect": {...}, "tol": x} ] }
// Supports objects, arrays, strings (no escapes beyond \" \\ \/ \n \t \r),
// numbers, true/false/null. Test-only code — not part of the library API.

#include <cctype>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_json {

class Value;
using ValuePtr = std::shared_ptr<Value>;

class Value {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<ValuePtr> arr;
    std::map<std::string, ValuePtr> obj;

    double as_number() const {
        if (type != Type::Number) throw std::runtime_error("mini_json: not a number");
        return num;
    }
    const std::string& as_string() const {
        if (type != Type::String) throw std::runtime_error("mini_json: not a string");
        return str;
    }
    const std::vector<ValuePtr>& as_array() const {
        if (type != Type::Array) throw std::runtime_error("mini_json: not an array");
        return arr;
    }
    const Value& at(const std::string& key) const {
        if (type != Type::Object) throw std::runtime_error("mini_json: not an object");
        auto it = obj.find(key);
        if (it == obj.end()) throw std::runtime_error("mini_json: missing key '" + key + "'");
        return *it->second;
    }
    bool has(const std::string& key) const {
        return type == Type::Object && obj.count(key) > 0;
    }
};

class Parser {
public:
    explicit Parser(const std::string& text) : s_(text) {}

    ValuePtr parse() {
        ValuePtr v = parse_value();
        skip_ws();
        if (pos_ != s_.size()) throw std::runtime_error("mini_json: trailing characters");
        return v;
    }

private:
    const std::string& s_;
    std::size_t pos_ = 0;

    void skip_ws() {
        while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) ++pos_;
    }
    char peek() {
        skip_ws();
        if (pos_ >= s_.size()) throw std::runtime_error("mini_json: unexpected end of input");
        return s_[pos_];
    }
    void expect(char c) {
        if (peek() != c) {
            throw std::runtime_error(std::string("mini_json: expected '") + c + "'");
        }
        ++pos_;
    }

    ValuePtr parse_value() {
        const char c = peek();
        switch (c) {
            case '{': return parse_object();
            case '[': return parse_array();
            case '"': return parse_string();
            case 't':
            case 'f': return parse_bool();
            case 'n': return parse_null();
            default: return parse_number();
        }
    }

    ValuePtr parse_object() {
        expect('{');
        auto v = std::make_shared<Value>();
        v->type = Value::Type::Object;
        if (peek() == '}') {
            ++pos_;
            return v;
        }
        while (true) {
            ValuePtr key = parse_string();
            expect(':');
            v->obj[key->str] = parse_value();
            const char c = peek();
            if (c == ',') {
                ++pos_;
                continue;
            }
            expect('}');
            return v;
        }
    }

    ValuePtr parse_array() {
        expect('[');
        auto v = std::make_shared<Value>();
        v->type = Value::Type::Array;
        if (peek() == ']') {
            ++pos_;
            return v;
        }
        while (true) {
            v->arr.push_back(parse_value());
            const char c = peek();
            if (c == ',') {
                ++pos_;
                continue;
            }
            expect(']');
            return v;
        }
    }

    ValuePtr parse_string() {
        expect('"');
        auto v = std::make_shared<Value>();
        v->type = Value::Type::String;
        while (pos_ < s_.size() && s_[pos_] != '"') {
            char c = s_[pos_++];
            if (c == '\\') {
                if (pos_ >= s_.size()) throw std::runtime_error("mini_json: bad escape");
                const char e = s_[pos_++];
                switch (e) {
                    case '"': c = '"'; break;
                    case '\\': c = '\\'; break;
                    case '/': c = '/'; break;
                    case 'n': c = '\n'; break;
                    case 't': c = '\t'; break;
                    case 'r': c = '\r'; break;
                    default: throw std::runtime_error("mini_json: unsupported escape");
                }
            }
            v->str.push_back(c);
        }
        if (pos_ >= s_.size()) throw std::runtime_error("mini_json: unterminated string");
        ++pos_;  // closing quote
        return v;
    }

    ValuePtr parse_bool() {
        auto v = std::make_shared<Value>();
        v->type = Value::Type::Bool;
        if (s_.compare(pos_, 4, "true") == 0) {
            v->b = true;
            pos_ += 4;
        } else if (s_.compare(pos_, 5, "false") == 0) {
            v->b = false;
            pos_ += 5;
        } else {
            throw std::runtime_error("mini_json: bad literal");
        }
        return v;
    }

    ValuePtr parse_null() {
        if (s_.compare(pos_, 4, "null") != 0) throw std::runtime_error("mini_json: bad literal");
        pos_ += 4;
        return std::make_shared<Value>();
    }

    ValuePtr parse_number() {
        skip_ws();
        const std::size_t start = pos_;
        while (pos_ < s_.size() &&
               (std::isdigit(static_cast<unsigned char>(s_[pos_])) || s_[pos_] == '-' ||
                s_[pos_] == '+' || s_[pos_] == '.' || s_[pos_] == 'e' || s_[pos_] == 'E')) {
            ++pos_;
        }
        if (pos_ == start) throw std::runtime_error("mini_json: expected a number");
        auto v = std::make_shared<Value>();
        v->type = Value::Type::Number;
        v->num = std::stod(s_.substr(start, pos_ - start));
        return v;
    }
};

inline ValuePtr parse(const std::string& text) { return Parser(text).parse(); }

}  // namespace mini_json

#endif  // IRM_TESTS_MINI_JSON_HPP

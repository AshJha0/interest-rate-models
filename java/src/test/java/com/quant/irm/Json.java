package com.quant.irm;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal recursive-descent JSON parser for the golden-case file — just
 * enough for the flat schema {@code {"cases":[{"name","inputs","expect","tol"}]}}.
 * Numbers become {@link Double}, objects {@code Map<String,Object>}, arrays
 * {@code List<Object>}. Package-private, test tree only.
 */
final class Json {

    private final String src;
    private int pos;

    private Json(String src) {
        this.src = src;
    }

    /** Parse a complete JSON document; trailing garbage is an error. */
    static Object parse(String text) {
        Json p = new Json(text);
        Object v = p.value();
        p.skipWs();
        if (p.pos != text.length()) {
            throw new IllegalArgumentException("trailing content at offset " + p.pos);
        }
        return v;
    }

    private void skipWs() {
        while (pos < src.length() && Character.isWhitespace(src.charAt(pos))) {
            pos++;
        }
    }

    private char peek() {
        if (pos >= src.length()) {
            throw new IllegalArgumentException("unexpected end of JSON");
        }
        return src.charAt(pos);
    }

    private void expect(char c) {
        if (peek() != c) {
            throw new IllegalArgumentException(
                    "expected '" + c + "' at offset " + pos + ", got '" + peek() + "'");
        }
        pos++;
    }

    private Object value() {
        skipWs();
        char c = peek();
        switch (c) {
            case '{':
                return object();
            case '[':
                return array();
            case '"':
                return string();
            case 't':
                literal("true");
                return Boolean.TRUE;
            case 'f':
                literal("false");
                return Boolean.FALSE;
            case 'n':
                literal("null");
                return null;
            default:
                return number();
        }
    }

    private void literal(String lit) {
        if (!src.startsWith(lit, pos)) {
            throw new IllegalArgumentException("bad literal at offset " + pos);
        }
        pos += lit.length();
    }

    private Map<String, Object> object() {
        Map<String, Object> out = new LinkedHashMap<>();
        expect('{');
        skipWs();
        if (peek() == '}') {
            pos++;
            return out;
        }
        while (true) {
            skipWs();
            String key = string();
            skipWs();
            expect(':');
            out.put(key, value());
            skipWs();
            if (peek() == ',') {
                pos++;
            } else {
                expect('}');
                return out;
            }
        }
    }

    private List<Object> array() {
        List<Object> out = new ArrayList<>();
        expect('[');
        skipWs();
        if (peek() == ']') {
            pos++;
            return out;
        }
        while (true) {
            out.add(value());
            skipWs();
            if (peek() == ',') {
                pos++;
            } else {
                expect(']');
                return out;
            }
        }
    }

    private String string() {
        expect('"');
        StringBuilder sb = new StringBuilder();
        while (true) {
            char c = src.charAt(pos++);
            if (c == '"') {
                return sb.toString();
            }
            if (c == '\\') {
                char e = src.charAt(pos++);
                switch (e) {
                    case '"': sb.append('"'); break;
                    case '\\': sb.append('\\'); break;
                    case '/': sb.append('/'); break;
                    case 'b': sb.append('\b'); break;
                    case 'f': sb.append('\f'); break;
                    case 'n': sb.append('\n'); break;
                    case 'r': sb.append('\r'); break;
                    case 't': sb.append('\t'); break;
                    case 'u':
                        sb.append((char) Integer.parseInt(src.substring(pos, pos + 4), 16));
                        pos += 4;
                        break;
                    default:
                        throw new IllegalArgumentException("bad escape \\" + e);
                }
            } else {
                sb.append(c);
            }
        }
    }

    private Double number() {
        int start = pos;
        while (pos < src.length()
                && "+-.eE0123456789".indexOf(src.charAt(pos)) >= 0) {
            pos++;
        }
        if (pos == start) {
            throw new IllegalArgumentException("bad token at offset " + pos);
        }
        return Double.valueOf(src.substring(start, pos));
    }
}

//! JSON text the way Go's `encoding/json` writes it, so rows this server
//! stores read the same as rows RAGFlow's Go server stores: object keys
//! sorted, `<`, `>`, `&`, U+2028 and U+2029 escaped, floats in Go's shortest
//! form (an integral float prints without a fraction).

use serde_json::Value;

/// `json.Marshal` of a decoded JSON value.
pub fn marshal(value: &Value) -> String {
    let mut out = String::new();
    write(value, &mut out);
    out
}

fn write(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&number(n)),
        Value::String(s) => string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            // Go sorts keys by their UTF-8 bytes; so does `str`'s ordering.
            keys.sort();
            out.push('{');
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                string(key, out);
                out.push(':');
                write(&map[key], out);
            }
            out.push('}');
        }
    }
}

fn number(n: &serde_json::Number) -> String {
    if let Some(i) = n.as_i64() {
        return i.to_string();
    }
    if let Some(u) = n.as_u64() {
        return u.to_string();
    }
    let f = n.as_f64().unwrap_or(0.0);
    let abs = f.abs();
    if abs != 0.0 && !(1e-6..1e21).contains(&abs) {
        // 'e' form: Go writes e+21 and e-07 as e+21 and e-7.
        let formatted = format!("{f:e}");
        let (mantissa, exp) = formatted.split_once('e').unwrap_or((&formatted, "0"));
        let exp: i32 = exp.parse().unwrap_or(0);
        return if exp < 0 {
            format!("{mantissa}e-{}", -exp)
        } else {
            format!("{mantissa}e+{exp:02}")
        };
    }
    format!("{f}")
}

fn string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '<' | '>' | '&' | '\u{2028}' | '\u{2029}' => {
                out.push_str(&format!("\\u{:04x}", c as u32))
            }
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn keys_sorted_and_html_escaped() {
        let v = json!({"b": "<a&b>", "a": [1, 2.5, true, null], "é": "\u{2028}"});
        // `§` stands for a backslash, so the expectation reads as the text.
        let want =
            r#"{"a":[1,2.5,true,null],"b":"§u003ca§u0026b§u003e","é":"§u2028"}"#.replace('§', "\\");
        assert_eq!(marshal(&v), want);
    }

    #[test]
    fn control_characters() {
        assert_eq!(marshal(&json!("a\u{1}\n\"\\")), r#""a\u0001\n\"\\""#);
    }

    #[test]
    fn floats_follow_go() {
        let parse = |s: &str| serde_json::from_str::<Value>(s).unwrap();
        assert_eq!(marshal(&parse("5.0")), "5");
        assert_eq!(marshal(&parse("0.25")), "0.25");
        assert_eq!(marshal(&parse("1e21")), "1e+21");
        assert_eq!(marshal(&parse("1.5e-7")), "1.5e-7");
        assert_eq!(marshal(&parse("0.000001")), "0.000001");
        assert_eq!(marshal(&parse("123456789012")), "123456789012");
    }
}

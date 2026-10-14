//! Request tokens, byte for byte as upstream's Go server reads them
//! (`internal/common/token.go`, `internal/service/user.go`).
//!
//! The UI's session token is not a JWT: it is Python's itsdangerous
//! `URLSafeTimedSerializer(secret_key).dumps(access_token)`. Verification is
//! HMAC-SHA1 over `<payload>.<timestamp>` with the key
//! `SHA1("itsdangerous" + "signer" + secret_key)`, a 24 hour maximum age, and
//! a payload that is a JSON string (the user's `access_token`), zlib
//! compressed when its encoding starts with a dot.

use std::io::Read;

use base64::alphabet::URL_SAFE;
use base64::engine::general_purpose::{GeneralPurpose, GeneralPurposeConfig};
use base64::engine::DecodePaddingMode;
use base64::Engine;
use hmac::{Hmac, Mac};
use sha1::{Digest, Sha1};

/// Upstream's `accessTokenExpireSeconds`.
pub const MAX_AGE_SECS: i64 = 86_400;

/// The URL-safe alphabet without padding, as Go's `base64.RawURLEncoding`
/// decodes it: trailing bits are not checked.
const RAW_URL: GeneralPurpose = GeneralPurpose::new(
    &URL_SAFE,
    GeneralPurposeConfig::new()
        .with_encode_padding(false)
        .with_decode_padding_mode(DecodePaddingMode::RequireNone)
        .with_decode_allow_trailing_bits(true),
);

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TokenError {
    #[error("empty authorization")]
    Empty,
    #[error("signature missing")]
    SignatureMissing,
    #[error("signature does not match")]
    SignatureMismatch,
    #[error("timestamp missing")]
    TimestampMissing,
    #[error("malformed timestamp")]
    MalformedTimestamp,
    #[error("signature age {0} > {MAX_AGE_SECS} seconds")]
    Expired(i64),
    #[error("signature age {0} < 0 seconds")]
    FromTheFuture(i64),
    #[error("failed to decode payload")]
    Payload,
    #[error("failed to parse payload")]
    NotAString,
}

fn signature(value: &str, secret: &str) -> String {
    let mut key = Sha1::new();
    key.update(b"itsdangerous");
    key.update(b"signer");
    key.update(secret.as_bytes());
    let key = key.finalize();
    let mut mac = <Hmac<Sha1> as Mac>::new_from_slice(&key).expect("HMAC takes any key length");
    mac.update(value.as_bytes());
    RAW_URL.encode(mac.finalize().into_bytes())
}

fn constant_time_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.bytes().zip(b.bytes()).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

/// Big-endian integer of the last eight bytes (`bytesToInt`).
fn bytes_to_int(data: &[u8]) -> i64 {
    let data = if data.len() > 8 {
        &data[data.len() - 8..]
    } else {
        data
    };
    let mut buf = [0u8; 8];
    buf[8 - data.len()..].copy_from_slice(data);
    u64::from_be_bytes(buf) as i64
}

/// Verifies the signature and the age; returns the still-encoded payload.
fn unsign(token: &str, secret: &str, now: i64) -> Result<String, TokenError> {
    let token = token.trim();
    let Some(li) = token.rfind('.') else {
        return Err(TokenError::SignatureMissing);
    };
    let (value, sig) = (&token[..li], &token[li + 1..]);
    if !constant_time_eq(sig, &signature(value, secret)) {
        return Err(TokenError::SignatureMismatch);
    }
    let Some(li) = value.rfind('.') else {
        return Err(TokenError::TimestampMissing);
    };
    let (payload, ts) = (&value[..li], &value[li + 1..]);
    let ts = RAW_URL
        .decode(ts)
        .map_err(|_| TokenError::MalformedTimestamp)?;
    let age = now - bytes_to_int(&ts);
    if age > MAX_AGE_SECS {
        return Err(TokenError::Expired(age));
    }
    if age < 0 {
        return Err(TokenError::FromTheFuture(age));
    }
    Ok(payload.to_string())
}

fn decode_payload(payload: &str) -> Result<Vec<u8>, TokenError> {
    let (payload, compressed) = match payload.strip_prefix('.') {
        Some(rest) => (rest, true),
        None => (payload, false),
    };
    let raw = RAW_URL.decode(payload).map_err(|_| TokenError::Payload)?;
    if !compressed {
        return Ok(raw);
    }
    let mut out = Vec::new();
    flate2::read::ZlibDecoder::new(raw.as_slice())
        .read_to_end(&mut out)
        .map_err(|_| TokenError::Payload)?;
    Ok(out)
}

/// `common.ExtractAccessToken`: the access token a UI session token carries.
/// `now` is unix seconds.
pub fn extract_access_token(
    authorization: &str,
    secret: &str,
    now: i64,
) -> Result<String, TokenError> {
    if authorization.is_empty() {
        return Err(TokenError::Empty);
    }
    let token = authorization
        .strip_prefix("Bearer ")
        .unwrap_or(authorization);
    let payload = unsign(token, secret, now)?;
    let json = decode_payload(&payload)?;
    serde_json::from_slice::<String>(&json).map_err(|_| TokenError::NotAString)
}

/// Mints a session token the way `common.DumpAccessToken` does. Only tests
/// and tooling need it; the server never issues tokens.
pub fn dump_access_token(access_token: &str, secret: &str, now: i64) -> String {
    let json = serde_json::to_vec(access_token).expect("a string serializes");
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    std::io::Write::write_all(&mut z, &json).expect("write to a Vec");
    let compressed = z.finish().expect("finish a Vec encoder");
    let payload = if compressed.len() + 1 < json.len() {
        format!(".{}", RAW_URL.encode(&compressed))
    } else {
        RAW_URL.encode(&json)
    };
    let ts = (now as u64).to_be_bytes();
    let first = ts.iter().position(|b| *b != 0).unwrap_or(ts.len());
    let value = format!("{payload}.{}", RAW_URL.encode(&ts[first..]));
    let sig = signature(&value, secret);
    format!("{value}.{sig}")
}

/// `GetUserByAPIToken`'s header split: on single spaces, two parts take the
/// second, one part is the token itself, anything else is rejected. The
/// scheme word is not checked.
pub fn api_token_from_header(authorization: &str) -> Option<&str> {
    if authorization.is_empty() {
        return None;
    }
    let parts: Vec<&str> = authorization.split(' ').collect();
    match parts.as_slice() {
        [one] => Some(one),
        [_, token] => Some(token),
        _ => None,
    }
}

/// `GetUserByBetaAPIToken`'s header split: whitespace-separated fields; a
/// lone field is the token unless it is `Bearer` (any case).
pub fn beta_token_from_header(authorization: &str) -> Option<&str> {
    let parts: Vec<&str> = authorization.split_whitespace().collect();
    let token = match parts.as_slice() {
        [one] if one.eq_ignore_ascii_case("bearer") => return None,
        [one] => *one,
        [_, token] => *token,
        _ => return None,
    };
    (!token.is_empty()).then_some(token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let secret = "s3cret";
        let at = "0123456789abcdef0123456789abcdef";
        let t = dump_access_token(at, secret, 1_800_000_000);
        assert_eq!(extract_access_token(&t, secret, 1_800_000_005).unwrap(), at);
        assert_eq!(
            extract_access_token(&format!("Bearer {t}"), secret, 1_800_000_005).unwrap(),
            at
        );
        assert_eq!(
            extract_access_token(&format!("  {t} "), secret, 1_800_000_005).unwrap(),
            at,
            "unsign trims whitespace around the token"
        );
    }

    #[test]
    fn age_bounds() {
        let t = dump_access_token("a".repeat(32).as_str(), "k", 1_000_000);
        assert!(extract_access_token(&t, "k", 1_000_000 + MAX_AGE_SECS).is_ok());
        assert_eq!(
            extract_access_token(&t, "k", 1_000_000 + MAX_AGE_SECS + 1),
            Err(TokenError::Expired(MAX_AGE_SECS + 1))
        );
        assert_eq!(
            extract_access_token(&t, "k", 999_999),
            Err(TokenError::FromTheFuture(-1))
        );
    }

    #[test]
    fn wrong_secret_and_garbage() {
        let t = dump_access_token("a".repeat(32).as_str(), "k", 10);
        assert_eq!(
            extract_access_token(&t, "other", 10),
            Err(TokenError::SignatureMismatch)
        );
        assert_eq!(extract_access_token("", "k", 10), Err(TokenError::Empty));
        assert_eq!(
            extract_access_token("nodots", "k", 10),
            Err(TokenError::SignatureMissing)
        );
        assert_eq!(
            extract_access_token("ragflow-abc", "k", 10),
            Err(TokenError::SignatureMissing)
        );
    }

    #[test]
    fn long_tokens_are_compressed() {
        let at = "ab".repeat(400);
        let t = dump_access_token(&at, "k", 77);
        assert!(t.starts_with('.'), "payload compressed: {t}");
        assert_eq!(extract_access_token(&t, "k", 77).unwrap(), at);
    }

    #[test]
    fn api_token_split() {
        assert_eq!(api_token_from_header("ragflow-x"), Some("ragflow-x"));
        assert_eq!(api_token_from_header("Bearer ragflow-x"), Some("ragflow-x"));
        assert_eq!(api_token_from_header("Foo ragflow-x"), Some("ragflow-x"));
        assert_eq!(api_token_from_header("Bearer  ragflow-x"), None);
        assert_eq!(api_token_from_header("a b c"), None);
        assert_eq!(api_token_from_header(""), None);
    }

    #[test]
    fn beta_token_split() {
        assert_eq!(beta_token_from_header("abc"), Some("abc"));
        assert_eq!(beta_token_from_header("  Bearer   abc "), Some("abc"));
        assert_eq!(beta_token_from_header("bearer"), None);
        assert_eq!(beta_token_from_header("BEARER"), None);
        assert_eq!(beta_token_from_header("a b c"), None);
        assert_eq!(beta_token_from_header("   "), None);
    }

    /// Tokens minted by upstream's own `common.DumpAccessToken`
    /// (`fixtures/gen`), with their secrets: every one must verify, and the
    /// ones whose payload is neither compressed nor HTML-escaped must be
    /// reproduced byte for byte.
    #[test]
    fn go_minted_tokens() {
        let text = include_str!("../fixtures/go_tokens.json");
        let doc: serde_json::Value = serde_json::from_str(text).unwrap();
        let cases = doc["cases"].as_array().unwrap();
        assert!(cases.len() >= 12);
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let secret = case["secret"].as_str().unwrap();
            let access = case["access_token"].as_str().unwrap();
            let token = case["token"].as_str().unwrap();
            // The timestamp is the second-to-last dot segment.
            let mut parts = token.rsplit('.');
            let _sig = parts.next().unwrap();
            let ts = parts.next().unwrap();
            let ts = bytes_to_int(&RAW_URL.decode(ts).unwrap());
            assert_eq!(
                extract_access_token(token, secret, ts + 1).unwrap(),
                access,
                "{name}"
            );
            assert_eq!(
                extract_access_token(&format!("Bearer {token}"), secret, ts).unwrap(),
                access,
                "{name} with prefix"
            );
            assert_eq!(
                extract_access_token(token, secret, ts + MAX_AGE_SECS + 1),
                Err(TokenError::Expired(MAX_AGE_SECS + 1)),
                "{name} expired"
            );
            assert!(
                extract_access_token(token, "not-the-secret", ts).is_err(),
                "{name} wrong secret"
            );
            // The signature over Go's own `<payload>.<timestamp>` is byte-exact.
            let (value, sig) = token.rsplit_once('.').unwrap();
            assert_eq!(signature(value, secret), sig, "{name}: signature");
            // Minting agrees wherever zlib would not matter: Go and flate2
            // compress short inputs differently, so only an uncompressed
            // payload without HTML escapes is compared whole.
            if !token.starts_with('.') && !access.contains(['<', '>', '&']) {
                let mine = dump_access_token(access, secret, ts);
                if !mine.starts_with('.') {
                    assert_eq!(mine, token, "{name}: byte-exact");
                }
            }
        }
    }
}

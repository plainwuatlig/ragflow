//! `gojianfan.T2S`, the Traditional → Simplified table the Go query builder
//! applies (module `github.com/siongui/gojianfan`, pinned in upstream go.mod).
//!
//! The map is built the way the Go package's `init` builds it: walk `ChT` by
//! byte offset, read the rune at the same byte offset of `ChS`, later entries
//! overwrite earlier ones.

use std::collections::HashMap;
use std::sync::OnceLock;

#[path = "jianfan_table.rs"]
mod table;

fn t2s_map() -> &'static HashMap<char, char> {
    static MAP: OnceLock<HashMap<char, char>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut m = HashMap::new();
        assert_eq!(table::CH_T.len(), table::CH_S.len());
        for (idx, t) in table::CH_T.char_indices() {
            let s = table::CH_S[idx..].chars().next().expect("aligned tables");
            m.insert(t, s);
        }
        m
    })
}

/// `gojianfan.T2S`.
pub fn t2s(s: &str) -> String {
    let m = t2s_map();
    s.chars().map(|c| *m.get(&c).unwrap_or(&c)).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn converts_common_traditional() {
        assert_eq!(super::t2s("營業資訊 abc"), "营业资讯 abc");
    }
}

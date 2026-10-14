//! WordNet synsets as `internal/service/nlp/wordnet.go` @ v1.0.0-rc1 reads
//! them: `index.*` for lemma → offsets, `*.exc` for exceptions, `data.*` read
//! at the offset for the synset's first lemma.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const POSES: [&str; 4] = ["n", "v", "a", "r"];

fn file_suffix(pos: &str) -> Option<&'static str> {
    match pos {
        "n" => Some("noun"),
        "v" => Some("verb"),
        "a" | "s" => Some("adj"),
        "r" => Some("adv"),
        _ => None,
    }
}

fn substitutions(pos: &str) -> &'static [(&'static str, &'static str)] {
    match pos {
        "n" => &[
            ("s", ""),
            ("ses", "s"),
            ("ves", "f"),
            ("xes", "x"),
            ("zes", "z"),
            ("ches", "ch"),
            ("shes", "sh"),
            ("men", "man"),
            ("ies", "y"),
        ],
        "v" => &[
            ("s", ""),
            ("ies", "y"),
            ("es", "e"),
            ("es", ""),
            ("ed", "e"),
            ("ed", ""),
            ("ing", "e"),
            ("ing", ""),
        ],
        "a" => &[("er", ""), ("est", ""), ("er", "e"), ("est", "e")],
        _ => &[],
    }
}

pub struct WordNet {
    dir: PathBuf,
    lemma_pos_offsets: HashMap<String, HashMap<String, Vec<i64>>>,
    exceptions: HashMap<String, HashMap<String, Vec<String>>>,
    data_files: Mutex<HashMap<String, File>>,
}

#[derive(Debug, Clone)]
pub struct Synset {
    pub name: String,
    pub lemmas: Vec<String>,
}

impl WordNet {
    /// `NewWordNet(dir)`; an error means "no WordNet" to the caller.
    pub fn new(dir: &Path) -> Result<Self, String> {
        let mut exceptions: HashMap<String, HashMap<String, Vec<String>>> = HashMap::new();
        for pos in POSES {
            exceptions.insert(pos.to_string(), HashMap::new());
        }
        for pos in POSES {
            let suffix = file_suffix(pos).unwrap();
            let Ok(f) = File::open(dir.join(format!("{suffix}.exc"))) else {
                continue;
            };
            for line in BufReader::new(f).split(b'\n') {
                let line = line.map_err(|e| e.to_string())?;
                let line = String::from_utf8_lossy(&line);
                let fs: Vec<&str> = line.split_whitespace().collect();
                if fs.len() >= 2 {
                    exceptions.get_mut(pos).unwrap().insert(
                        fs[0].to_string(),
                        fs[1..].iter().map(|s| s.to_string()).collect(),
                    );
                }
            }
        }
        let mut lemma_pos_offsets: HashMap<String, HashMap<String, Vec<i64>>> = HashMap::new();
        for pos in POSES {
            let suffix = file_suffix(pos).unwrap();
            let path = dir.join(format!("index.{suffix}"));
            let f =
                File::open(&path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
            for line in BufReader::new(f).split(b'\n') {
                let line = line.map_err(|e| e.to_string())?;
                let line = String::from_utf8_lossy(&line);
                if line.is_empty() || line.as_bytes()[0] == b' ' {
                    continue;
                }
                let fs: Vec<&str> = line.split_whitespace().collect();
                if fs.len() < 6 {
                    continue;
                }
                let lemma = crate::gocompat::to_lower(fs[0]);
                let file_pos = fs[1].to_string();
                let Ok(n_synsets) = fs[2].parse::<i64>() else {
                    continue;
                };
                let Ok(n_pointers) = fs[3].parse::<i64>() else {
                    continue;
                };
                let mut idx = 4usize;
                let mut i = 0;
                while i < n_pointers && idx < fs.len() {
                    idx += 1;
                    i += 1;
                }
                if idx >= fs.len() || fs[idx].parse::<i64>().is_err() {
                    continue;
                }
                idx += 1;
                if idx >= fs.len() || fs[idx].parse::<i64>().is_err() {
                    continue;
                }
                idx += 1;
                let mut offsets = Vec::new();
                let mut i = 0;
                while i < n_synsets && idx < fs.len() {
                    if let Ok(o) = fs[idx].parse::<i64>() {
                        offsets.push(o);
                    }
                    idx += 1;
                    i += 1;
                }
                lemma_pos_offsets
                    .entry(lemma)
                    .or_default()
                    .insert(file_pos, offsets);
            }
        }
        Ok(WordNet {
            dir: dir.to_path_buf(),
            lemma_pos_offsets,
            exceptions,
            data_files: Mutex::new(HashMap::new()),
        })
    }

    fn morphy(&self, form: &str, pos: &str) -> Vec<String> {
        let form = crate::gocompat::to_lower(form);
        let mut forms: Vec<String> = self
            .exceptions
            .get(pos)
            .and_then(|m| m.get(&form))
            .cloned()
            .unwrap_or_default();
        if forms.is_empty() {
            for (old, new) in substitutions(pos) {
                if form.ends_with(old) {
                    forms.push(format!("{}{}", &form[..form.len() - old.len()], new));
                }
            }
        }
        let mut all = vec![form];
        all.extend(forms);
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for f in all {
            if self
                .lemma_pos_offsets
                .get(&f)
                .is_some_and(|m| m.contains_key(pos))
                && seen.insert(f.clone())
            {
                out.push(f);
            }
        }
        out
    }

    fn read_line_at(&self, pos: &str, offset: i64) -> Result<String, String> {
        let pos = if pos == "s" { "a" } else { pos };
        let mut files = self.data_files.lock().unwrap_or_else(|e| e.into_inner());
        if !files.contains_key(pos) {
            let suffix = file_suffix(pos).ok_or_else(|| format!("unknown POS: {pos}"))?;
            let f =
                File::open(self.dir.join(format!("data.{suffix}"))).map_err(|e| e.to_string())?;
            files.insert(pos.to_string(), f);
        }
        let f = files.get_mut(pos).unwrap();
        f.seek(SeekFrom::Start(offset as u64))
            .map_err(|e| e.to_string())?;
        // bufio.Reader.ReadString('\n'): read until newline; EOF without one is an error.
        let mut buf = Vec::new();
        let mut byte = [0u8; 4096];
        loop {
            let n = f.read(&mut byte).map_err(|e| e.to_string())?;
            if n == 0 {
                return Err("EOF".into());
            }
            if let Some(p) = byte[..n].iter().position(|b| *b == b'\n') {
                buf.extend_from_slice(&byte[..=p]);
                break;
            }
            buf.extend_from_slice(&byte[..n]);
        }
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    fn synset_from_pos_and_offset(&self, pos: &str, offset: i64) -> Result<Synset, String> {
        let line = self.read_line_at(pos, offset)?;
        if line.len() < 8 {
            return Err("short line".into());
        }
        let line_offset = line[..8].trim();
        if line_offset != format!("{offset:08}") {
            return Err("offset mismatch".into());
        }
        let (data, _gloss) = line.split_once('|').ok_or("invalid line format")?;
        let fs: Vec<&str> = data.split_whitespace().collect();
        if fs.len() < 4 {
            return Err("too few fields".into());
        }
        fs[0].parse::<i64>().map_err(|e| e.to_string())?;
        let ss_type = fs[2];
        let w_cnt = i64::from_str_radix(fs[3], 16).map_err(|e| e.to_string())?;
        let mut lemmas = Vec::new();
        let mut idx = 4usize;
        let mut i = 0;
        while i < w_cnt && idx + 1 < fs.len() {
            let mut lemma = fs[idx];
            if let Some(p) = lemma.find('(') {
                lemma = &lemma[..p];
            }
            lemmas.push(lemma.to_string());
            idx += 2;
            i += 1;
        }
        if lemmas.is_empty() {
            return Err("no lemmas found".into());
        }
        let sense = self.find_sense_number(&lemmas[0], pos, offset);
        let name = if sense > 0 {
            format!("{}.{}.{:02}", lemmas[0], ss_type, sense)
        } else {
            format!("{}.{}.01", crate::gocompat::to_lower(&lemmas[0]), ss_type)
        };
        Ok(Synset { name, lemmas })
    }

    fn find_sense_number(&self, lemma: &str, pos: &str, offset: i64) -> i64 {
        let lemma = crate::gocompat::to_lower(lemma);
        if let Some(offsets) = self.lemma_pos_offsets.get(&lemma).and_then(|m| m.get(pos)) {
            for (i, off) in offsets.iter().enumerate() {
                if *off == offset {
                    return i as i64 + 1;
                }
            }
        }
        1
    }

    /// `Synsets(lemma, "")`: all parts of speech.
    pub fn synsets(&self, lemma: &str) -> Vec<Synset> {
        let lemma = crate::gocompat::to_lower(lemma);
        let mut results = Vec::new();
        let mut seen = HashSet::new();
        for p in POSES {
            for form in self.morphy(&lemma, p) {
                let Some(offsets) = self.lemma_pos_offsets.get(&form).and_then(|m| m.get(p)) else {
                    continue;
                };
                for &offset in offsets {
                    if seen.insert(format!("{p}-{offset}")) {
                        if let Ok(s) = self.synset_from_pos_and_offset(p, offset) {
                            results.push(s);
                        }
                    }
                }
            }
        }
        results
    }
}

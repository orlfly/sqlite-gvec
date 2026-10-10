//! Full-text search via SQLite's built-in FTS5 module.
//!
//! gvec creates FTS5 virtual tables as shadow tables alongside its node
//! tables. Sync triggers keep the index up-to-date on writes. Search
//! returns BM25-ranked `(rowid, score)` pairs.

use crate::error::Result;
use crate::sql::{SqlExecutor, SqlValue};
use serde_json::Value;

/// Supported FTS5 tokenizers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tokenizer {
    Unicode61,
    Porter,
    Trigram,
}

impl Tokenizer {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "unicode61" | "" => Some(Tokenizer::Unicode61),
            "porter" => Some(Tokenizer::Porter),
            "trigram" => Some(Tokenizer::Trigram),
            _ => None,
        }
    }

    pub fn to_fts5_arg(&self) -> &'static str {
        match self {
            Tokenizer::Unicode61 => "unicode61",
            Tokenizer::Porter => "porter",
            Tokenizer::Trigram => "trigram",
        }
    }
}

/// Build the FTS5 shadow-table name for a given label + property.
pub fn fts_table_name(prefix: &str, label: &str, property: &str) -> String {
    format!("{}_fts_{}_{}", prefix, label, property)
}

/// Build the trigger names for sync.
fn trigger_names(prefix: &str, label: &str, property: &str) -> (String, String, String) {
    let base = format!("{}_fts_trig_{}_{}", prefix, label, property);
    (
        format!("{}_ins", base),
        format!("{}_upd", base),
        format!("{}_del", base),
    )
}

/// Validate that `s` is a safe SQL identifier: ASCII letter / digit / underscore,
/// must not be empty, must not start with a digit.
fn validate_identifier(kind: &str, s: &str) -> Result<()> {
    if s.is_empty() {
        return Err(crate::error::Error::Parse {
            message: format!("{kind} must not be empty"),
            position: None,
        });
    }
    let mut chars = s.chars();
    let first = chars.next().unwrap();
    if !(first.is_ascii_alphabetic() || first == '_') {
        return Err(crate::error::Error::Parse {
            message: format!("{kind} '{s}' must start with a letter or underscore"),
            position: None,
        });
    }
    for c in chars {
        if !(c.is_ascii_alphanumeric() || c == '_') {
            return Err(crate::error::Error::Parse {
                message: format!("{kind} '{s}' contains invalid character '{c}'"),
                position: None,
            });
        }
    }
    Ok(())
}

/// Wrap an identifier in double quotes, escaping embedded `"` as `""`.
fn quote_ident(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

/// Escape a single-quoted string for SQL: `'` becomes `''`.
fn quote_string(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

/// Create an FTS5 virtual table for a node property, plus sync triggers.
pub fn create_text_index(
    exec: &dyn SqlExecutor,
    prefix: &str,
    label: &str,
    property: &str,
    tokenizer: &Tokenizer,
) -> Result<()> {
    validate_identifier("prefix", prefix)?;
    validate_identifier("label", label)?;
    validate_identifier("property", property)?;
    let fts_tbl = fts_table_name(prefix, label, property);
    let nodes_tbl = format!("{}_nodes", prefix);
    let tok = tokenizer.to_fts5_arg();
    let lbl_pat_lit = quote_string(&format!("%\"{}\"%", label));
    let prop_json = format!("$.{}", property);

    // 1. Create FTS5 virtual table (standalone content — stores its own copy).
    //    `IF NOT EXISTS` makes this idempotent against repeated calls with
    //    the same (prefix, label, property) tuple.
    exec.execute(
        &format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS {} USING fts5(content, tokenize={})",
            fts_tbl,
            quote_string(tok)
        ),
        &[],
    )?;

    // 2. Backfill existing matching nodes. We use `INSERT OR REPLACE` so the
    //    second call of `create_text_index` for an already-indexed
    //    (label, property) does not fail with a rowid UNIQUE conflict: each
    //    existing FTS5 row is replaced with the current property value, and
    //    new labels/properties get a fresh row. This makes both
    //    `create_text_index` and `CREATE TEXT INDEX` idempotent at the
    //    user-visible level.
    exec.execute(
        &format!(
            "INSERT OR REPLACE INTO {}(rowid, content) SELECT id, json_extract(properties, '{}') FROM {} WHERE labels LIKE ?1 AND json_extract(properties, '{}') IS NOT NULL",
            fts_tbl, prop_json, nodes_tbl, prop_json
        ),
        &[SqlValue::Text(format!("%\"{}\"%", label))],
    )?;

    // 3. Sync triggers using WHEN clauses to filter by label.
    // NOTE: FTS5 'delete' uses the special {tbl} placeholder for the table name;
    // we emit literal braces by escaping them in the format string as `{{...}}`.
    let (trig_ins, trig_upd, trig_del) = trigger_names(prefix, label, property);
    let q_ins = quote_string(&lbl_pat_lit); // single-quoted SQL literal

    exec.execute(
        &format!(
            "CREATE TRIGGER IF NOT EXISTS {ti} AFTER INSERT ON {nt} WHEN new.labels LIKE {lp} BEGIN
                INSERT INTO {ft}(rowid, content) VALUES (new.id, json_extract(new.properties, '{pj}'));
            END",
            ti = trig_ins, nt = nodes_tbl, lp = q_ins, ft = fts_tbl, pj = prop_json,
        ),
        &[],
    )?;

    exec.execute(
        &format!(
            "CREATE TRIGGER IF NOT EXISTS {td} AFTER DELETE ON {nt} WHEN old.labels LIKE {lp} BEGIN
                INSERT INTO {ft}({ft}, rowid, content) VALUES('delete', old.id, json_extract(old.properties, '{pj}'));
            END",
            td = trig_del, nt = nodes_tbl, lp = q_ins, ft = fts_tbl, pj = prop_json,
        ),
        &[],
    )?;

    exec.execute(
        &format!(
            "CREATE TRIGGER IF NOT EXISTS {tu} AFTER UPDATE ON {nt} WHEN new.labels LIKE {lp} BEGIN
                INSERT INTO {ft}({ft}, rowid, content) VALUES('delete', old.id, json_extract(old.properties, '{pj}'));
                INSERT INTO {ft}(rowid, content) VALUES (new.id, json_extract(new.properties, '{pj}'));
            END",
            tu = trig_upd, nt = nodes_tbl, lp = q_ins, ft = fts_tbl, pj = prop_json,
        ),
        &[],
    )?;

    Ok(())
}

/// Drop an FTS5 index and its triggers.
pub fn drop_text_index(
    exec: &dyn SqlExecutor,
    prefix: &str,
    label: &str,
    property: &str,
) -> Result<()> {
    validate_identifier("prefix", prefix)?;
    validate_identifier("label", label)?;
    validate_identifier("property", property)?;
    let fts_tbl = fts_table_name(prefix, label, property);
    let (trig_ins, trig_upd, trig_del) = trigger_names(prefix, label, property);
    for t in [&trig_ins, &trig_upd, &trig_del] {
        let _ = exec.execute(&format!("DROP TRIGGER IF EXISTS {}", quote_ident(t)), &[]);
    }
    exec.execute(
        &format!("DROP TABLE IF EXISTS {}", quote_ident(&fts_tbl)),
        &[],
    )?;
    Ok(())
}

/// Search the FTS5 index for `query`, returning up to `k` `(rowid, bm25_score)` pairs.
/// BM25 scores from FTS5 are negative (lower = more relevant); we negate so
/// higher score = more relevant.
pub fn text_search(
    exec: &dyn SqlExecutor,
    prefix: &str,
    label: &str,
    property: &str,
    query: &str,
    k: usize,
) -> Result<Vec<(i64, f32)>> {
    validate_identifier("prefix", prefix)?;
    validate_identifier("label", label)?;
    validate_identifier("property", property)?;
    let fts_tbl = fts_table_name(prefix, label, property);
    let q_tbl = quote_ident(&fts_tbl);
    let sql = format!(
        "SELECT rowid, -bm25({}) AS score FROM {} WHERE {} MATCH ?1 ORDER BY score DESC LIMIT ?2",
        q_tbl, q_tbl, q_tbl
    );
    let rows = exec.query_all_json(
        &sql,
        &[
            SqlValue::Text(query.to_string()),
            SqlValue::Integer(k as i64),
        ],
    )?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        if row.len() < 2 {
            continue;
        }
        let rowid = match &row[0] {
            Value::Number(n) => n.as_i64().unwrap_or(0),
            Value::String(s) => s.parse().unwrap_or(0),
            _ => 0,
        };
        let score = match &row[1] {
            Value::Number(n) => n.as_f64().unwrap_or(0.0) as f32,
            _ => 0.0,
        };
        out.push((rowid, score));
    }
    Ok(out)
}

/// Check whether FTS5 is available in the current SQLite build.
pub fn fts5_available(exec: &dyn SqlExecutor) -> bool {
    match exec.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS __gvec_fts5_probe USING fts5(x)",
        &[],
    ) {
        Ok(_) => {
            let _ = exec.execute("DROP TABLE IF EXISTS __gvec_fts5_probe", &[]);
            true
        }
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_parse() {
        assert_eq!(Tokenizer::parse("unicode61"), Some(Tokenizer::Unicode61));
        assert_eq!(Tokenizer::parse("porter"), Some(Tokenizer::Porter));
        assert_eq!(Tokenizer::parse("trigram"), Some(Tokenizer::Trigram));
        assert_eq!(Tokenizer::parse("unknown"), None);
    }

    #[test]
    fn fts_table_name_format() {
        assert_eq!(fts_table_name("g", "Doc", "body"), "g_fts_Doc_body");
    }
}

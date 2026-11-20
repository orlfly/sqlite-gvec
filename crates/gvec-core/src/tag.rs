//! Exact tag indexing over JSON array or string node properties.
//!
//! A tag index stores one row per `(index_id, node_id, tag)` in a shared
//! shadow table. Values are extracted with SQLite's JSON1 `json_each`, so a
//! property may be either a string or an array of strings/numbers. Matching is
//! exact and case-sensitive.

use crate::error::Result;
use crate::sql::{SqlExecutor, SqlValue};
use serde_json::Value;

/// Metadata for one tag index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagIndexMeta {
    pub id: i64,
    pub label: String,
    pub property: String,
}

fn tag_indexes_table(prefix: &str) -> String {
    format!("{prefix}_tag_indexes")
}

fn tags_table(prefix: &str) -> String {
    format!("{prefix}_tags")
}

fn trigger_names(prefix: &str, label: &str, property: &str) -> (String, String, String) {
    // Length-prefix label and property so `("Doc_x", "y")` and
    // `("Doc", "x_y")` cannot produce the same trigger name.
    let base = format!(
        "{prefix}_tag_trig_{}_{}_{}_{}",
        label.len(),
        label,
        property.len(),
        property
    );
    (
        format!("{base}_ins"),
        format!("{base}_upd"),
        format!("{base}_del"),
    )
}

fn quote_ident(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

fn quote_string(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

/// Create the tag index shadow tables. Idempotent.
pub fn ensure_schema(exec: &dyn SqlExecutor, prefix: &str) -> Result<()> {
    crate::storage::validate_ident("prefix", prefix)?;
    exec.execute(
        &format!(
            "CREATE TABLE IF NOT EXISTS {} (
                id       INTEGER PRIMARY KEY,
                label    TEXT NOT NULL,
                property TEXT NOT NULL,
                UNIQUE(label, property)
            )",
            quote_ident(&tag_indexes_table(prefix))
        ),
        &[],
    )?;
    exec.execute(
        &format!(
            "CREATE TABLE IF NOT EXISTS {} (
                index_id INTEGER NOT NULL,
                node_id  INTEGER NOT NULL,
                tag      TEXT NOT NULL,
                PRIMARY KEY (index_id, node_id, tag)
            )",
            quote_ident(&tags_table(prefix))
        ),
        &[],
    )?;
    exec.execute(
        &format!(
            "CREATE INDEX IF NOT EXISTS {} ON {}(index_id, tag, node_id)",
            quote_ident(&format!("{}_tag_lookup", prefix)),
            quote_ident(&tags_table(prefix))
        ),
        &[],
    )?;
    Ok(())
}

/// List every tag index for a graph.
pub fn list_tag_indexes(exec: &dyn SqlExecutor, prefix: &str) -> Result<Vec<TagIndexMeta>> {
    crate::storage::validate_ident("prefix", prefix)?;
    let rows = exec.query_all_json(
        &format!(
            "SELECT id, label, property FROM {} ORDER BY id",
            quote_ident(&tag_indexes_table(prefix))
        ),
        &[],
    )?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        if row.len() < 3 {
            continue;
        }
        out.push(TagIndexMeta {
            id: row[0].as_i64().unwrap_or(0),
            label: row[1].as_str().unwrap_or_default().to_string(),
            property: row[2].as_str().unwrap_or_default().to_string(),
        });
    }
    Ok(out)
}

/// Find one tag index by label and property.
pub fn find_tag_index(
    exec: &dyn SqlExecutor,
    prefix: &str,
    label: &str,
    property: &str,
) -> Result<Option<TagIndexMeta>> {
    crate::storage::validate_ident("prefix", prefix)?;
    crate::storage::validate_ident("label", label)?;
    crate::storage::validate_ident("property", property)?;
    let row = exec.query_one_json(
        &format!(
            "SELECT id, label, property FROM {} WHERE label = ?1 AND property = ?2",
            quote_ident(&tag_indexes_table(prefix))
        ),
        &[
            SqlValue::Text(label.to_string()),
            SqlValue::Text(property.to_string()),
        ],
    )?;
    Ok(row.and_then(|r| {
        if r.len() < 3 {
            return None;
        }
        Some(TagIndexMeta {
            id: r[0].as_i64().unwrap_or(0),
            label: r[1].as_str().unwrap_or_default().to_string(),
            property: r[2].as_str().unwrap_or_default().to_string(),
        })
    }))
}

/// Create or rebuild a tag index and its sync triggers.
pub fn create_tag_index(
    exec: &dyn SqlExecutor,
    prefix: &str,
    label: &str,
    property: &str,
) -> Result<TagIndexMeta> {
    crate::storage::validate_ident("label", label)?;
    crate::storage::validate_ident("property", property)?;
    ensure_schema(exec, prefix)?;

    let meta_table = quote_ident(&tag_indexes_table(prefix));
    exec.execute(
        &format!("INSERT OR IGNORE INTO {meta_table} (label, property) VALUES (?1, ?2)"),
        &[
            SqlValue::Text(label.to_string()),
            SqlValue::Text(property.to_string()),
        ],
    )?;
    let meta = find_tag_index(exec, prefix, label, property)?
        .ok_or_else(|| crate::error::Error::Storage("tag index disappeared after create".into()))?;

    let nodes_tbl = quote_ident(&format!("{prefix}_nodes"));
    let tags_tbl = quote_ident(&tags_table(prefix));
    let prop_json = format!("$.{property}");
    let label_lit = quote_string(&format!("%\"{label}\"%"));
    let (trig_ins, trig_upd, trig_del) = trigger_names(prefix, label, property);
    let values_of = |column: &str| {
        format!(
            "json_each(CASE json_type({column}, '{prop_json}')
                WHEN 'array' THEN json_extract({column}, '{prop_json}')
                WHEN 'text' THEN json_quote(json_extract({column}, '{prop_json}'))
                WHEN 'integer' THEN json_extract({column}, '{prop_json}')
                WHEN 'real' THEN json_extract({column}, '{prop_json}')
             END)"
        )
    };
    let backfill_values = values_of("n.properties");
    let insert_values = values_of("new.properties");
    // JSON numeric equality treats 42 and 42.0 as the same value, while
    // `CAST(42.0 AS TEXT)` is "42.0". Store integral reals in integer form so a
    // tag-index anchor for integer 42 also finds a property written as 42.0.
    let tag_value = "CASE WHEN j.type = 'real' AND CAST(j.value AS INTEGER) = j.value \
                     THEN CAST(CAST(j.value AS INTEGER) AS TEXT) \
                     ELSE CAST(j.value AS TEXT) END";

    // Rebuild in place: drop old triggers and rows, then backfill from the
    // current node state. A repeated CREATE TAG INDEX is idempotent.
    for t in [&trig_ins, &trig_upd, &trig_del] {
        exec.execute(&format!("DROP TRIGGER IF EXISTS {}", quote_ident(t)), &[])?;
    }
    exec.execute(
        &format!("DELETE FROM {tags_tbl} WHERE index_id = ?1"),
        &[SqlValue::from(meta.id)],
    )?;

    exec.execute(
        &format!(
            "INSERT OR IGNORE INTO {tags_tbl}(index_id, node_id, tag)
             SELECT ?1, n.id, {tag_value}
             FROM {nodes_tbl} AS n, {backfill_values} AS j
             WHERE n.labels LIKE ?2
               AND json_type(n.properties, '{prop_json}') IN ('array', 'text', 'integer', 'real')
               AND j.type IN ('text', 'integer', 'real')
               AND j.value IS NOT NULL"
        ),
        &[
            SqlValue::from(meta.id),
            SqlValue::Text(format!("%\"{label}\"%")),
        ],
    )?;

    let idx = meta.id;
    exec.execute(
        &format!(
            "CREATE TRIGGER {ti} AFTER INSERT ON {nodes_tbl} BEGIN
                INSERT OR IGNORE INTO {tags_tbl}(index_id, node_id, tag)
                SELECT {idx}, new.id, {tag_value}
                FROM {insert_values} AS j
                WHERE new.labels LIKE {label_lit}
                  AND json_type(new.properties, '{prop_json}') IN ('array', 'text', 'integer', 'real')
                  AND j.type IN ('text', 'integer', 'real')
                  AND j.value IS NOT NULL;
            END",
            ti = quote_ident(&trig_ins),
        ),
        &[],
    )?;
    exec.execute(
        &format!(
            "CREATE TRIGGER {tu} AFTER UPDATE ON {nodes_tbl} BEGIN
                DELETE FROM {tags_tbl} WHERE index_id = {idx} AND node_id = old.id;
                INSERT OR IGNORE INTO {tags_tbl}(index_id, node_id, tag)
                SELECT {idx}, new.id, {tag_value}
                FROM {insert_values} AS j
                WHERE new.labels LIKE {label_lit}
                  AND json_type(new.properties, '{prop_json}') IN ('array', 'text', 'integer', 'real')
                  AND j.type IN ('text', 'integer', 'real')
                  AND j.value IS NOT NULL;
            END",
            tu = quote_ident(&trig_upd),
        ),
        &[],
    )?;
    exec.execute(
        &format!(
            "CREATE TRIGGER {td} AFTER DELETE ON {nodes_tbl} BEGIN
                DELETE FROM {tags_tbl} WHERE index_id = {idx} AND node_id = old.id;
            END",
            td = quote_ident(&trig_del),
        ),
        &[],
    )?;

    Ok(meta)
}

/// Drop a tag index, its tags, and its triggers. Missing indexes are a no-op.
pub fn drop_tag_index(
    exec: &dyn SqlExecutor,
    prefix: &str,
    label: &str,
    property: &str,
) -> Result<()> {
    crate::storage::validate_ident("prefix", prefix)?;
    crate::storage::validate_ident("label", label)?;
    crate::storage::validate_ident("property", property)?;
    let (trig_ins, trig_upd, trig_del) = trigger_names(prefix, label, property);
    for t in [&trig_ins, &trig_upd, &trig_del] {
        exec.execute(&format!("DROP TRIGGER IF EXISTS {}", quote_ident(t)), &[])?;
    }
    if let Some(meta) = find_tag_index(exec, prefix, label, property)? {
        exec.execute(
            &format!(
                "DELETE FROM {} WHERE index_id = ?1",
                quote_ident(&tags_table(prefix))
            ),
            &[SqlValue::from(meta.id)],
        )?;
        exec.execute(
            &format!(
                "DELETE FROM {} WHERE id = ?1",
                quote_ident(&tag_indexes_table(prefix))
            ),
            &[SqlValue::from(meta.id)],
        )?;
    }
    Ok(())
}

/// Number of distinct nodes carrying at least one tag in the index.
pub fn tag_count(exec: &dyn SqlExecutor, prefix: &str, index_id: i64) -> Result<usize> {
    let row = exec.query_one_json(
        &format!(
            "SELECT COUNT(DISTINCT node_id) FROM {} WHERE index_id = ?1",
            quote_ident(&tags_table(prefix))
        ),
        &[SqlValue::from(index_id)],
    )?;
    Ok(row
        .and_then(|r| r.first().and_then(|v| v.as_i64()))
        .unwrap_or(0) as usize)
}

/// Fetch node ids for an exact tag. `limit` is only used by the public API;
/// query integration asks for every match.
pub fn tag_search_ids(
    exec: &dyn SqlExecutor,
    prefix: &str,
    index_id: i64,
    tag: &str,
    limit: Option<usize>,
) -> Result<Vec<i64>> {
    if limit == Some(0) {
        return Ok(Vec::new());
    }
    let table = quote_ident(&tags_table(prefix));
    let (sql, params) = match limit {
        Some(k) => (
            format!(
                "SELECT node_id FROM {table} WHERE index_id = ?1 AND tag = ?2 ORDER BY node_id LIMIT ?3"
            ),
            vec![
                SqlValue::from(index_id),
                SqlValue::Text(tag.to_string()),
                SqlValue::Integer(k as i64),
            ],
        ),
        None => (
            format!("SELECT node_id FROM {table} WHERE index_id = ?1 AND tag = ?2 ORDER BY node_id"),
            vec![SqlValue::from(index_id), SqlValue::Text(tag.to_string())],
        ),
    };
    let rows = exec.query_all_json(&sql, &params)?;
    Ok(rows
        .into_iter()
        .filter_map(|row| row.first().and_then(|v| v.as_i64()))
        .collect())
}

/// Validate a tag property value against a tag without touching SQLite.
pub fn value_has_tag(value: &Value, tag: &str) -> bool {
    match value {
        Value::String(s) => s == tag,
        Value::Number(n) => number_tag(n).is_some_and(|t| t == tag),
        Value::Array(items) => items.iter().any(|item| match item {
            Value::String(s) => s == tag,
            Value::Number(n) => number_tag(n).is_some_and(|t| t == tag),
            _ => false,
        }),
        _ => false,
    }
}

/// Canonical tag text for a JSON number. Integral reals use integer form so
/// `42` and `42.0` compare equal, matching JSON numeric equality.
pub fn number_tag(n: &serde_json::Number) -> Option<String> {
    if let Some(i) = n.as_i64() {
        return Some(i.to_string());
    }
    if let Some(u) = n.as_u64() {
        return Some(u.to_string());
    }
    let f = n.as_f64()?;
    if f.fract() == 0.0 && f.abs() <= 9_007_199_254_740_992.0 {
        Some((f as i64).to_string())
    } else {
        Some(f.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn string_and_array_tags_match_exactly() {
        assert!(value_has_tag(&json!("database"), "database"));
        assert!(value_has_tag(&json!(["database", "rust"]), "rust"));
        assert!(value_has_tag(&json!([1, 2, 3]), "2"));
        assert!(value_has_tag(&json!(42), "42"));
        assert!(value_has_tag(&json!(42.0), "42"));
        assert!(!value_has_tag(&json!(["databases"]), "database"));
        assert!(!value_has_tag(&json!(["Database"]), "database"));
        assert!(!value_has_tag(&json!({"tag": "database"}), "database"));
    }
}

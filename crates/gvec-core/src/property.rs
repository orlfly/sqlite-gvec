use serde_json::Value;

/// Resolve a dotted path like `n.address.city` against a `serde_json::Value`.
///
/// `None` for missing intermediate or leaf nodes; `Some(None)` if the
/// intermediate is an object but the leaf is missing; the JSON value otherwise.
pub fn resolve<'a>(root: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = root;
    for part in path.split('.') {
        match current {
            Value::Object(map) => {
                current = map.get(part)?;
            }
            _ => return None,
        }
    }
    Some(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn resolves_top_level() {
        let v = json!({"name": "Alice"});
        assert_eq!(resolve(&v, "name"), Some(&json!("Alice")));
    }

    #[test]
    fn resolves_nested() {
        let v = json!({"address": {"city": "NYC"}});
        assert_eq!(resolve(&v, "address.city"), Some(&json!("NYC")));
    }

    #[test]
    fn missing_returns_none() {
        let v = json!({"name": "Alice"});
        assert_eq!(resolve(&v, "address.city"), None);
    }

    #[test]
    fn wrong_type_returns_none() {
        let v = json!({"name": "Alice"});
        assert_eq!(resolve(&v, "name.length"), None);
    }
}

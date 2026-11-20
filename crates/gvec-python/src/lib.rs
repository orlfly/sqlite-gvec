//! Python bindings for gvec, implemented in Rust with PyO3.
//!
//! This module replaces the previous ctypes/`sqlite3` Python wrapper. The
//! entire engine, including graph queries, FTS, and HNSW/IVF vector indexes,
//! runs in-process through the `gvec` Rust binding; Python only receives
//! converted result values.
//!
//! Build with `maturin develop` (see `pyproject.toml`) or manually:
//!
//! ```text
//! cargo build --manifest-path crates/gvec-python/Cargo.toml
//! cp target/debug/lib_gvec.so \
//!    bindings/python/gvec/_gvec.$(python3-config --extension-suffix)
//! ```
//!
//! Import from Python as `from gvec import Database, Vector, Distance`.

use gvec::{
    Database as RustDatabase, Distance as CoreDistance, Graph as RustGraph, VectorIndexMeta,
    VectorValue,
};
use pyo3::conversion::IntoPyObjectExt;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use serde_json::Value as Json;

/// Map an engine error to a Python exception.
fn to_pyerr(error: gvec::Error) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn json_to_py(py: Python<'_>, value: &Json) -> PyResult<Py<PyAny>> {
    match value {
        Json::Null => Ok(py.None()),
        Json::Bool(b) => (*b).into_py_any(py),
        Json::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_py_any(py)
            } else if let Some(u) = n.as_u64() {
                u.into_py_any(py)
            } else {
                n.as_f64().unwrap_or(0.0).into_py_any(py)
            }
        }
        Json::String(s) => s.as_str().into_py_any(py),
        Json::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            list.into_py_any(py)
        }
        Json::Object(map) => {
            let dict = PyDict::new(py);
            for (key, value) in map {
                dict.set_item(key, json_to_py(py, value)?)?;
            }
            dict.into_py_any(py)
        }
    }
}

/// Convert Python values accepted in index options to JSON. Supports the
/// value types used by gvec options: bool, int, float, str, list/tuple, dict.
fn py_to_json(value: &Bound<'_, PyAny>) -> PyResult<Json> {
    if value.is_none() {
        return Ok(Json::Null);
    }
    if let Ok(b) = value.extract::<bool>() {
        return Ok(Json::Bool(b));
    }
    if let Ok(i) = value.extract::<i64>() {
        return Ok(Json::Number(i.into()));
    }
    if let Ok(f) = value.extract::<f64>() {
        return Ok(Json::Number(
            serde_json::Number::from_f64(f).unwrap_or_else(|| 0.into()),
        ));
    }
    if let Ok(s) = value.extract::<String>() {
        return Ok(Json::String(s));
    }
    if let Ok(list) = value.cast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            out.push(py_to_json(&item)?);
        }
        return Ok(Json::Array(out));
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        let mut out = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            out.push(py_to_json(&item)?);
        }
        return Ok(Json::Array(out));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        let mut out = serde_json::Map::new();
        for (key, item) in dict.iter() {
            let key = key
                .extract::<String>()
                .map_err(|_| PyValueError::new_err("option keys must be strings"))?;
            out.insert(key, py_to_json(&item)?);
        }
        return Ok(Json::Object(out));
    }
    Err(PyValueError::new_err(format!(
        "unsupported option value: {}",
        value.repr()?
    )))
}

/// A float32 query vector.
#[pyclass(module = "gvec._gvec")]
#[derive(Clone)]
pub struct Vector {
    #[pyo3(get, set)]
    values: Vec<f32>,
}

#[pymethods]
impl Vector {
    #[new]
    fn new(values: Vec<f32>) -> Self {
        Self { values }
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.values).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!("Vector({:?})", self.values)
    }

    fn __len__(&self) -> usize {
        self.values.len()
    }
}

/// Distance metric constants, preserving the previous Python API:
/// `Distance.L2`, `Distance.COSINE`, `Distance.INNER_PRODUCT`.
#[pyclass(module = "gvec._gvec")]
pub struct Distance;

#[pymethods]
impl Distance {
    #[classattr]
    const L2: &'static str = "l2";

    #[classattr]
    const COSINE: &'static str = "cosine";

    #[classattr]
    const INNER_PRODUCT: &'static str = "ip";
}

fn metric_from_str(metric: &str) -> PyResult<CoreDistance> {
    CoreDistance::parse(metric)
        .ok_or_else(|| PyValueError::new_err(format!("unknown distance metric: {metric}")))
}

/// An in-process database. Unlike the old ctypes binding there is no SQLite
/// extension to load; the Rust core embeds SQLite directly.
#[pyclass(module = "gvec._gvec")]
pub struct Database {
    inner: Option<RustDatabase>,
    path: String,
}

#[pymethods]
impl Database {
    #[new]
    #[pyo3(signature = (path = ":memory:"))]
    fn new(path: &str) -> PyResult<Self> {
        let inner = if path == ":memory:" {
            RustDatabase::open_in_memory()
        } else {
            RustDatabase::open(path)
        }
        .map_err(to_pyerr)?;
        Ok(Self {
            inner: Some(inner),
            path: path.to_string(),
        })
    }

    fn graph(&self, name: &str) -> PyResult<Graph> {
        let database = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;
        let graph = database.graph(name).map_err(to_pyerr)?;
        Ok(Graph { inner_graph: graph })
    }

    fn close(&mut self) {
        // Graph handles own clones of the shared connection, so closing the
        // Database releases only this handle. This mirrors sqlite3 semantics.
        self.inner = None;
    }

    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    #[getter]
    fn version(&self) -> &'static str {
        gvec::VERSION
    }
}

/// A named graph backed by the in-process engine.
#[pyclass(unsendable, module = "gvec._gvec")]
pub struct Graph {
    inner_graph: RustGraph,
}

#[pymethods]
impl Graph {
    #[getter]
    fn name(&self) -> &str {
        &self.inner_graph.name
    }

    /// Run a Cypher-like query and return rows as lists of Python values.
    fn run(&self, py: Python<'_>, cypher: &str) -> PyResult<Vec<Vec<Py<PyAny>>>> {
        let result = self.inner_graph.run(cypher).map_err(to_pyerr)?;
        let mut rows = Vec::with_capacity(result.rows.len());
        for row in result.rows {
            let mut converted = Vec::with_capacity(row.len());
            for value in &row {
                converted.push(json_to_py(py, value)?);
            }
            rows.push(converted);
        }
        Ok(rows)
    }

    /// Brute-force KNN over caller-supplied vectors. Kept for API parity
    /// with the previous Python wrapper; for persisted data use
    /// `vector_search`.
    #[pyo3(signature = (query, stored, k = 10, metric = "l2"))]
    fn knn(
        &self,
        py: Python<'_>,
        query: PyRef<'_, Vector>,
        stored: Vec<(i64, Py<Vector>)>,
        k: usize,
        metric: &str,
    ) -> PyResult<Vec<(i64, f32)>> {
        let distance = metric_from_str(metric)?;
        let query_vector = VectorValue::F32(query.values.clone());
        let mut stored_vectors = Vec::with_capacity(stored.len());
        for (id, vector) in stored {
            stored_vectors.push((id, VectorValue::F32(vector.borrow(py).values.clone())));
        }
        Ok(self
            .inner_graph
            .knn(&query_vector, &stored_vectors, k, distance))
    }

    /// Create or reconfigure an ANN vector index over `label.property`.
    #[pyo3(signature = (label, property, algorithm = "hnsw", metric = None, options = None))]
    fn create_vector_index(
        &self,
        py: Python<'_>,
        label: &str,
        property: &str,
        algorithm: &str,
        metric: Option<&str>,
        options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let options = match options {
            Some(value) => Some(py_to_json(value)?),
            None => None,
        };
        let index = self
            .inner_graph
            .create_vector_index(label, property, algorithm, metric, options)
            .map_err(to_pyerr)?;
        index_meta_to_py(py, index)
    }

    fn drop_vector_index(&self, label: &str, property: &str) -> PyResult<()> {
        self.inner_graph
            .drop_vector_index(label, property)
            .map_err(to_pyerr)
    }

    fn list_vector_indexes(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let indexes = self.inner_graph.list_vector_indexes().map_err(to_pyerr)?;
        indexes
            .into_iter()
            .map(|index| index_meta_to_py(py, index))
            .collect()
    }

    fn vector_count(&self, label: &str, property: &str) -> PyResult<usize> {
        self.inner_graph
            .vector_count(label, property)
            .map_err(to_pyerr)
    }

    /// ANN search using the configured index, falling back to exact L2 scan.
    #[pyo3(signature = (label, property, query, k = 10))]
    fn vector_search(
        &self,
        label: &str,
        property: &str,
        query: PyRef<'_, Vector>,
        k: usize,
    ) -> PyResult<Vec<(i64, f32)>> {
        let query_vector = VectorValue::F32(query.values.clone());
        self.inner_graph
            .vector_search(label, property, &query_vector, k)
            .map_err(to_pyerr)
    }

    /// Exact scan with an explicit metric.
    #[pyo3(signature = (label, property, query, k = 10, metric = "l2"))]
    fn exact_vector_search(
        &self,
        label: &str,
        property: &str,
        query: PyRef<'_, Vector>,
        k: usize,
        metric: &str,
    ) -> PyResult<Vec<(i64, f32)>> {
        let distance = metric_from_str(metric)?;
        let query_vector = VectorValue::F32(query.values.clone());
        self.inner_graph
            .exact_vector_search(label, property, &query_vector, k, distance)
            .map_err(to_pyerr)
    }

    #[pyo3(signature = (label, property, tokenizer = None))]
    fn create_text_index(
        &self,
        label: &str,
        property: &str,
        tokenizer: Option<&str>,
    ) -> PyResult<()> {
        self.inner_graph
            .create_text_index(label, property, tokenizer)
            .map_err(to_pyerr)
    }

    fn drop_text_index(&self, label: &str, property: &str) -> PyResult<()> {
        self.inner_graph
            .drop_text_index(label, property)
            .map_err(to_pyerr)
    }

    #[pyo3(signature = (label, property, query, k = 10))]
    fn text_search(
        &self,
        label: &str,
        property: &str,
        query: &str,
        k: usize,
    ) -> PyResult<Vec<(i64, f32)>> {
        self.inner_graph
            .text_search(label, property, query, k)
            .map_err(to_pyerr)
    }
}

fn index_meta_to_py(py: Python<'_>, index: VectorIndexMeta) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("label", index.label)?;
    dict.set_item("property", index.property)?;
    dict.set_item("algorithm", index.algorithm)?;
    dict.set_item("metric", index.metric)?;
    dict.set_item("params", json_to_py(py, &index.params)?)?;
    dict.set_item("dim", index.dim)?;
    dict.set_item("element_type", index.element_type)?;
    dict.into_py_any(py)
}

/// Python module implemented in Rust.
#[pymodule]
fn _gvec(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Database>()?;
    module.add_class::<Graph>()?;
    module.add_class::<Vector>()?;
    module.add_class::<Distance>()?;
    module.add("VERSION", gvec::VERSION)?;
    Ok(())
}

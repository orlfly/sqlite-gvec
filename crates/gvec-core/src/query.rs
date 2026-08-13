//! Cypher-like query pipeline: lexer → parser → planner → executor.
//!
//! Scope matches `specs/graph-query-language/spec.md`.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;

use crate::error::{Error, Result};
use crate::ids::{EdgeId, NodeId};
use crate::node::Node;
use crate::property::resolve as resolve_path;
use crate::storage::Storage;

// ===========================================================================
// AST
// ===========================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Statement {
    Create(Create),
    Match(Match),
    Set(Set),
    Delete(Delete),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Query {
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Create {
    pub patterns: Vec<Pattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Match {
    pub optional: bool,
    pub patterns: Vec<Pattern>,
    pub where_: Option<Expr>,
    pub return_: Option<Return>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Set {
    pub assigns: Vec<Assign>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Assign {
    pub variable: String,
    pub property: Option<String>,
    pub value: Expr,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Delete {
    pub variables: Vec<String>,
    pub detach: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Pattern {
    pub segments: Vec<Segment>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Segment {
    pub node: NodePattern,
    pub edge: Option<EdgePattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NodePattern {
    pub variable: String,
    pub labels: Vec<String>,
    pub properties: Vec<(String, Expr)>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EdgePattern {
    pub variable: Option<String>,
    pub types: Vec<String>,
    pub direction: Direction,
    pub variable_length: Option<(usize, Option<usize>)>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Direction {
    Out,
    In,
    Either,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Return {
    pub items: Vec<ReturnItem>,
    pub order_by: Vec<OrderItem>,
    pub skip: Option<usize>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReturnItem {
    pub expr: Expr,
    pub alias: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrderItem {
    pub expr: Expr,
    pub desc: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Expr {
    Variable(String, Option<String>),
    VectorNn { variable: String, property: Option<String>, k: i64, query: Box<Expr> },
    Literal(Value),
    Cmp(Box<Expr>, CmpOp, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CmpOp { Eq, Ne, Lt, Le, Gt, Ge }

// ===========================================================================
// Lexer
// ===========================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: Tok,
    pub pos: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Kw(String),
    Ident(String),
    Colon, LParen, RParen, LBracket, RBracket,
    Arrow, Dash,
    Dot, Comma, Star, DotDot,
    Pipe,
    Eq, Ne, Lt, Le, Gt, Ge,
    Plus, Minus,
    LBrace, RBrace,
    StrLit(String),
    NumLit(f64),
    IntLit(i64),
    BoolLit(bool),
    NullLit,
    Eof,
}

pub struct Lexer<'a> { src: &'a str, pos: usize }

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Self { Self { src, pos: 0 } }
    fn peek(&self) -> Option<char> { self.src[self.pos..].chars().next() }
    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }
    fn at(&self, s: &str) -> bool { self.src[self.pos..].starts_with(s) }
    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() { self.pos += c.len_utf8(); }
            else if self.at("//") {
                while let Some(c) = self.peek() {
                    self.pos += c.len_utf8();
                    if c == '\n' { break; }
                }
            } else { break; }
        }
    }
    fn ident_or_kw(&mut self) -> Tok {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' { self.pos += c.len_utf8(); } else { break; }
        }
        let s = &self.src[start..self.pos];
        match s.to_ascii_uppercase().as_str() {
            "TRUE" => Tok::BoolLit(true),
            "FALSE" => Tok::BoolLit(false),
            "NULL" => Tok::NullLit,
            "AND" | "OR" | "NOT" | "IN" | "CREATE" | "MATCH" | "OPTIONAL" | "RETURN" | "SET" | "DELETE" | "DETACH" | "WHERE" | "ORDER" | "BY" | "SKIP" | "LIMIT" | "ASC" | "DESC" | "AS" | "VECTOR_NN" => Tok::Kw(s.to_ascii_uppercase()),
            _ => Tok::Ident(s.into()),
        }
    }
    fn num(&mut self) -> Result<Tok> {
        let start = self.pos;
        let mut is_float = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() { self.pos += c.len_utf8(); }
            // Treat '.' as a decimal point only if it's followed by a digit AND
            // not by '..' (the range operator inside `[*m..n]`).
            else if c == '.' && !is_float && !self.at("..") && {
                let next = self.src[self.pos + 1..].chars().next();
                matches!(next, Some(d) if d.is_ascii_digit())
            } {
                is_float = true;
                self.pos += 1;
            }
            else { break; }
        }
        let s = &self.src[start..self.pos];
        if is_float {
            Ok(Tok::NumLit(s.parse::<f64>().map_err(|e| Error::Parse { message: e.to_string(), position: None })?))
        } else {
            Ok(Tok::IntLit(s.parse::<i64>().map_err(|e| Error::Parse { message: e.to_string(), position: None })?))
        }
    }
    fn string(&mut self) -> Result<Tok> {
        self.bump();
        let mut out = String::new();
        while let Some(c) = self.peek() {
            if c == '"' { self.bump(); return Ok(Tok::StrLit(out)); }
            if c == '\\' {
                self.bump();
                match self.peek() {
                    Some('n') => { out.push('\n'); self.bump(); }
                    Some('t') => { out.push('\t'); self.bump(); }
                    Some('"') => { out.push('"'); self.bump(); }
                    Some('\\') => { out.push('\\'); self.bump(); }
                    _ => return Err(Error::Parse { message: "bad escape".into(), position: None }),
                }
            } else {
                out.push(c);
                self.pos += c.len_utf8();
            }
        }
        Err(Error::Parse { message: "unterminated string".into(), position: None })
    }
    pub fn next(&mut self) -> Result<Token> {
        self.skip_ws();
        let pos = self.pos;
        let Some(c) = self.peek() else { return Ok(Token { kind: Tok::Eof, pos }); };
        let kind = match c {
            '(' => { self.bump(); Tok::LParen }
            ')' => { self.bump(); Tok::RParen }
            '[' => { self.bump(); Tok::LBracket }
            ']' => { self.bump(); Tok::RBracket }
            '{' => { self.bump(); Tok::LBrace }
            '}' => { self.bump(); Tok::RBrace }
            ',' => { self.bump(); Tok::Comma }
            ':' => { self.bump(); Tok::Colon }
            '.' => {
                if self.at("..") { self.pos += 2; Tok::DotDot }
                else { self.bump(); Tok::Dot }
            }
            '*' => { self.bump(); Tok::Star }
            '+' => { self.bump(); Tok::Plus }
            '-' => {
                if self.at("->") { self.pos += 2; Tok::Arrow }
                else { self.bump(); Tok::Dash }
            }
            '=' => { self.bump(); Tok::Eq }
            '<' => {
                if self.at("<-") { self.pos += 2; Tok::Arrow }
                else if self.at("<=") { self.pos += 2; Tok::Le }
                else { self.bump(); Tok::Lt }
            }
            '>' => {
                if self.at(">=") { self.pos += 2; Tok::Ge }
                else { self.bump(); Tok::Gt }
            }
            '!' => {
                if self.at("!=") { self.pos += 2; Tok::Ne }
                else { return Err(Error::Parse { message: "expected !=".into(), position: Some((pos, pos+1)) }); }
            }
            '"' => return self.string().map(|k| Token { kind: k, pos }),
            '0'..='9' => return self.num().map(|k| Token { kind: k, pos }),
            c if c.is_alphabetic() || c == '_' => return Ok(Token { kind: self.ident_or_kw(), pos }),
            other => return Err(Error::Parse { message: format!("unexpected character {:?}", other), position: Some((pos, pos+other.len_utf8())) }),
        };
        Ok(Token { kind, pos })
    }
}

// ===========================================================================
// Parser
// ===========================================================================

pub struct Parser { tokens: Vec<Token>, pos: usize, src: String }

impl Parser {
    pub fn new(src: &str) -> Result<Self> {
        let mut l = Lexer::new(src);
        let mut toks = Vec::new();
        loop {
            let t = l.next()?;
            let done = t.kind == Tok::Eof;
            toks.push(t);
            if done { break; }
        }
        Ok(Self { tokens: toks, pos: 0, src: src.to_string() })
    }

    fn tokens_src(&self) -> &str { &self.src }

    fn peek(&self) -> &Tok { &self.tokens[self.pos].kind }
    fn bump(&mut self) -> Token {
        let t = self.tokens[self.pos].clone();
        self.pos += 1;
        t
    }
    fn eat_kw(&mut self, kw: &str) -> bool {
        if let Tok::Kw(k) = self.peek() {
            if k == kw { self.pos += 1; return true; }
        }
        false
    }
    fn expect_kw(&mut self, kw: &str) -> Result<()> {
        if !self.eat_kw(kw) {
            return Err(Error::Parse {
                message: format!("expected keyword {}", kw),
                position: Some((self.tokens[self.pos].pos, self.tokens[self.pos].pos + 1)),
            });
        }
        Ok(())
    }
    fn expect_tok(&mut self, t: Tok, label: &str) -> Result<()> {
        if std::mem::discriminant(self.peek()) != std::mem::discriminant(&t) {
            return Err(Error::Parse {
                message: format!("expected {}", label),
                position: Some((self.tokens[self.pos].pos, self.tokens[self.pos].pos + 1)),
            });
        }
        self.bump();
        Ok(())
    }

    pub fn parse_query(&mut self) -> Result<Query> {
        let mut stmts = Vec::new();
        loop {
            match self.peek() {
                Tok::Eof => break,
                _ => stmts.push(self.parse_statement()?),
            }
        }
        if stmts.is_empty() { return Err(Error::Parse { message: "empty query".into(), position: None }); }
        Ok(Query { statements: stmts })
    }

    fn parse_statement(&mut self) -> Result<Statement> {
        match self.peek() {
            Tok::Kw(k) if k == "CREATE" => {
                self.bump();
                Ok(Statement::Create(Create { patterns: self.parse_patterns()? }))
            }
            Tok::Kw(k) if k == "MATCH" || k == "OPTIONAL" => Ok(Statement::Match(self.parse_match()?)),
            Tok::Kw(k) if k == "SET" => {
                self.bump();
                let assigns = self.parse_assign_list()?;
                Ok(Statement::Set(Set { assigns }))
            }
            Tok::Kw(k) if k == "DELETE" => {
                self.bump();
                let mut detach = false;
                if self.eat_kw("DETACH") { detach = true; }
                let mut variables = Vec::new();
                loop {
                    if let Tok::Ident(v) = self.peek() { variables.push(v.clone()); self.bump(); }
                    else { return Err(Error::Parse { message: "expected variable".into(), position: None }); }
                    if !matches!(self.peek(), Tok::Comma) { break; }
                    self.bump();
                }
                Ok(Statement::Delete(Delete { variables, detach }))
            }
            _ => Err(Error::Parse { message: format!("expected statement, got {:?}", self.peek()), position: None }),
        }
    }

    fn parse_match(&mut self) -> Result<Match> {
        let mut optional = false;
        if self.eat_kw("OPTIONAL") { optional = true; }
        self.expect_kw("MATCH")?;
        let patterns = self.parse_patterns()?;
        let mut where_ = None;
        if self.eat_kw("WHERE") { where_ = Some(self.parse_expr()?); }
        let mut return_ = None;
        if self.eat_kw("RETURN") { return_ = Some(self.parse_return_clause()?); }
        Ok(Match { optional, patterns, where_, return_ })
    }

    fn parse_patterns(&mut self) -> Result<Vec<Pattern>> {
        let mut out = vec![self.parse_pattern()?];
        while matches!(self.peek(), Tok::Comma) { self.bump(); out.push(self.parse_pattern()?); }
        Ok(out)
    }

    fn parse_pattern(&mut self) -> Result<Pattern> {
        let node = self.parse_node_pattern()?;
        let mut segments = vec![Segment { node, edge: None }];
        loop {
            // The edge prefix can be: `-`, `->`, `<-`. The lexer produces:
            //   `-`  → Tok::Dash
            //   `->` → Tok::Arrow
            //   `<-` → Tok::Arrow  (the lexer maps both `->` and `<-` to Arrow)
            // We need to distinguish `<-` from `->` to set direction. We do
            // this by inspecting the raw source at the current token position.
            if !matches!(self.peek(), Tok::Dash | Tok::Arrow) { break; }
            let tok_pos = self.tokens[self.pos].pos;
            let leading_src = &self.tokens_src()[tok_pos..];
            let leading_incoming = leading_src.starts_with("<-");
            self.bump(); // consume - or -> or <-
            // For bare `->` or `<-` without `[`, we'd need a different path.
            // We require the bracket form: -[...]-> or <-[...]- or -[...]->
            if !matches!(self.peek(), Tok::LBracket) {
                return Err(Error::Parse { message: "expected '[' after dash/arrow in relationship pattern".into(), position: None });
            }
            self.expect_tok(Tok::LBracket, "[")?;
            let variable = if let Tok::Ident(s) = self.peek() { let x = s.clone(); self.bump(); Some(x) } else { None };
            let mut types = Vec::new();
            if matches!(self.peek(), Tok::Colon) {
                self.bump();
                loop {
                    if let Tok::Ident(s) = self.peek() { types.push(s.clone()); self.bump(); } else { return Err(Error::Parse { message: "expected type".into(), position: None }); }
                    if !matches!(self.peek(), Tok::Pipe) { break; }
                    self.bump();
                }
            }
            let mut vlen = None;
            if matches!(self.peek(), Tok::Star) {
                self.bump();
                let lo = self.parse_int_or_wildcard()?;
                let mut hi = None;
                if matches!(self.peek(), Tok::DotDot) { self.bump(); hi = Some(self.parse_int_or_wildcard()?); }
                vlen = Some((lo, hi));
            }
            self.expect_tok(Tok::RBracket, "]")?;
            // Trailing direction: `->` = out, `-` = either/undirected.
            let trailing_out = matches!(self.peek(), Tok::Arrow);
            if trailing_out { self.bump(); }
            else if matches!(self.peek(), Tok::Dash) { self.bump(); }
            // Combine: `<-[...]-` = In,  `<-[...]->` = Either (both),  `-[...]->` = Out,  `-[...]-` = Either
            let direction = match (leading_incoming, trailing_out) {
                (true, false) => Direction::In,
                (false, true) => Direction::Out,
                _ => Direction::Either,
            };
            let next = self.parse_node_pattern()?;
            segments.push(Segment { node: next, edge: Some(EdgePattern { variable, types, direction, variable_length: vlen }) });
        }
        Ok(Pattern { segments })
    }

    fn parse_int_or_wildcard(&mut self) -> Result<usize> {
        match self.peek().clone() {
            Tok::IntLit(n) => { self.bump(); Ok(n as usize) }
            Tok::Star => { self.bump(); Ok(0) }
            _ => Err(Error::Parse { message: "expected integer or *".into(), position: None }),
        }
    }

    fn parse_node_pattern(&mut self) -> Result<NodePattern> {
        self.expect_tok(Tok::LParen, "(")?;
        let variable = if let Tok::Ident(s) = self.peek() { let x = s.clone(); self.bump(); x } else {
            return Err(Error::Parse { message: "expected variable in node pattern".into(), position: None });
        };
        let mut labels = Vec::new();
        loop {
            if !matches!(self.peek(), Tok::Colon) { break; }
            self.bump();
            if let Tok::Ident(s) = self.peek() { labels.push(s.clone()); self.bump(); } else { return Err(Error::Parse { message: "expected label".into(), position: None }); }
        }
        let mut properties = Vec::new();
        if matches!(self.peek(), Tok::LBrace) {
            self.bump();
            if !matches!(self.peek(), Tok::RBrace) {
                loop {
                    let key = if let Tok::Ident(s) = self.peek() { let x = s.clone(); self.bump(); x } else { return Err(Error::Parse { message: "expected property key".into(), position: None }); };
                    self.expect_tok(Tok::Colon, ":")?;
                    let value = self.parse_expr()?;
                    properties.push((key, value));
                    if !matches!(self.peek(), Tok::Comma) { break; }
                    self.bump();
                }
            }
            self.expect_tok(Tok::RBrace, "}")?;
        }
        self.expect_tok(Tok::RParen, ")")?;
        Ok(NodePattern { variable, labels, properties })
    }

    fn parse_assign_list(&mut self) -> Result<Vec<Assign>> {
        let mut out = vec![self.parse_assign()?];
        while matches!(self.peek(), Tok::Comma) { self.bump(); out.push(self.parse_assign()?); }
        Ok(out)
    }

    fn parse_assign(&mut self) -> Result<Assign> {
        let variable = if let Tok::Ident(s) = self.peek() { let x = s.clone(); self.bump(); x } else {
            return Err(Error::Parse { message: "expected variable".into(), position: None });
        };
        let mut property = None;
        if matches!(self.peek(), Tok::Dot) {
            self.bump();
            if let Tok::Ident(s) = self.peek() { property = Some(s.clone()); self.bump(); } else { return Err(Error::Parse { message: "expected property name".into(), position: None }); }
        }
        self.expect_tok(Tok::Eq, "=")?;
        let value = self.parse_expr()?;
        Ok(Assign { variable, property, value })
    }

    fn parse_return_clause(&mut self) -> Result<Return> {
        let mut items = Vec::new();
        loop {
            let expr = self.parse_expr()?;
            let mut alias = None;
            if self.eat_kw("AS") { if let Tok::Ident(s) = self.peek() { alias = Some(s.clone()); self.bump(); } }
            items.push(ReturnItem { expr, alias });
            if !matches!(self.peek(), Tok::Comma) { break; }
            self.bump();
        }
        let mut order_by = Vec::new();
        if self.eat_kw("ORDER") {
            self.expect_kw("BY")?;
            loop {
                let expr = self.parse_expr()?;
                let desc = if self.eat_kw("DESC") { true } else { self.eat_kw("ASC"); false };
                order_by.push(OrderItem { expr, desc });
                if !matches!(self.peek(), Tok::Comma) { break; }
                self.bump();
            }
        }
        let mut skip = None;
        if self.eat_kw("SKIP") { if let Tok::IntLit(n) = self.peek().clone() { skip = Some(n as usize); self.bump(); } }
        let mut limit = None;
        if self.eat_kw("LIMIT") { if let Tok::IntLit(n) = self.peek().clone() { limit = Some(n as usize); self.bump(); } }
        Ok(Return { items, order_by, skip, limit })
    }

    pub fn parse_expr(&mut self) -> Result<Expr> { self.parse_or() }

    fn parse_or(&mut self) -> Result<Expr> {
        let mut left = self.parse_and()?;
        while self.eat_kw("OR") { let right = self.parse_and()?; left = Expr::Or(Box::new(left), Box::new(right)); }
        Ok(left)
    }
    fn parse_and(&mut self) -> Result<Expr> {
        let mut left = self.parse_not()?;
        while self.eat_kw("AND") { let right = self.parse_not()?; left = Expr::And(Box::new(left), Box::new(right)); }
        Ok(left)
    }
    fn parse_not(&mut self) -> Result<Expr> {
        if self.eat_kw("NOT") { Ok(Expr::Not(Box::new(self.parse_not()?))) } else { self.parse_cmp() }
    }
    fn parse_cmp(&mut self) -> Result<Expr> {
        let left = self.parse_add()?;
        let op = match self.peek() {
            Tok::Eq => Some(CmpOp::Eq),
            Tok::Ne => Some(CmpOp::Ne),
            Tok::Lt => Some(CmpOp::Lt),
            Tok::Le => Some(CmpOp::Le),
            Tok::Gt => Some(CmpOp::Gt),
            Tok::Ge => Some(CmpOp::Ge),
            _ => None,
        };
        if let Some(op) = op { self.bump(); let right = self.parse_add()?; return Ok(Expr::Cmp(Box::new(left), op, Box::new(right))); }
        Ok(left)
    }
    fn parse_add(&mut self) -> Result<Expr> { self.parse_primary() }

    fn parse_primary(&mut self) -> Result<Expr> {
        match self.peek().clone() {
            Tok::LParen => { self.bump(); let e = self.parse_expr()?; self.expect_tok(Tok::RParen, ")")?; Ok(e) }
            Tok::StrLit(s) => { self.bump(); Ok(Expr::Literal(Value::String(s))) }
            Tok::IntLit(n) => { self.bump(); Ok(Expr::Literal(Value::from(n))) }
            Tok::NumLit(f) => { self.bump(); Ok(Expr::Literal(Value::from(f))) }
            Tok::BoolLit(b) => { self.bump(); Ok(Expr::Literal(Value::Bool(b))) }
            Tok::NullLit => { self.bump(); Ok(Expr::Literal(Value::Null)) }
            Tok::Ident(s) => {
                self.bump();
                if s.eq_ignore_ascii_case("VECTOR_NN") && matches!(self.peek(), Tok::LParen) {
                    self.bump();
                    let (var, prop) = self.parse_var_with_prop()?;
                    self.expect_tok(Tok::Comma, ",")?;
                    let qexpr = self.parse_expr()?;
                    self.expect_tok(Tok::Comma, ",")?;
                    let k = match self.peek().clone() {
                        Tok::IntLit(n) => { self.bump(); n }
                        _ => return Err(Error::Parse { message: "expected integer k".into(), position: None }),
                    };
                    self.expect_tok(Tok::RParen, ")")?;
                    return Ok(Expr::VectorNn { variable: var, property: prop, k, query: Box::new(qexpr) });
                }
                let mut property = None;
                if matches!(self.peek(), Tok::Dot) {
                    self.bump();
                    if let Tok::Ident(p) = self.peek() { property = Some(p.clone()); self.bump(); }
                }
                Ok(Expr::Variable(s, property))
            }
            _ => Err(Error::Parse { message: format!("unexpected token {:?}", self.peek()), position: None }),
        }
    }

    fn parse_var_with_prop(&mut self) -> Result<(String, Option<String>)> {
        if let Tok::Ident(s) = self.peek() { let x = s.clone(); self.bump();
            let mut prop = None;
            if matches!(self.peek(), Tok::Dot) {
                self.bump();
                if let Tok::Ident(p) = self.peek() { prop = Some(p.clone()); self.bump(); }
            }
            Ok((x, prop))
        } else { Err(Error::Parse { message: "expected variable".into(), position: None }) }
    }
}

// ===========================================================================
// Executor
// ===========================================================================

pub type Row = BTreeMap<String, Value>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultSet {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
}

impl ResultSet {
    pub fn to_json(&self) -> Value {
        Value::Array(self.rows.iter().map(|r| Value::Array(r.clone())).collect())
    }
}

pub fn execute(storage: &Storage, query: &Query) -> Result<ResultSet> {
    let mut state = ExecState::default();
    for stmt in &query.statements {
        state = exec_statement(storage, stmt, state)?;
    }
    if let Some(Statement::Match(m)) = query.statements.last() {
        if let Some(ret) = &m.return_ {
            return materialize(&state.rows, ret);
        }
    }
    let mut cols: Vec<String> = Vec::new();
    for r in &state.rows {
        for k in r.keys() { if !cols.contains(k) { cols.push(k.clone()); } }
    }
    let rows: Vec<Vec<Value>> = state.rows.iter()
        .map(|r| cols.iter().map(|k| r.get(k).cloned().unwrap_or(Value::Null)).collect())
        .collect();
    Ok(ResultSet { columns: cols, rows })
}

#[derive(Default, Clone)]
struct ExecState { rows: Vec<Row> }

fn exec_statement(storage: &Storage, stmt: &Statement, mut state: ExecState) -> Result<ExecState> {
    match stmt {
        Statement::Create(c) => {
            for pattern in &c.patterns {
                state.rows = execute_create(storage, pattern, state.rows)?;
            }
            Ok(state)
        }
        Statement::Match(m) => {
            if state.rows.is_empty() { state.rows = vec![Row::new()]; }
            let mut current = state.rows.clone();
            for pattern in &m.patterns {
                current = execute_match(storage, pattern, &current, m.optional)?;
            }
            if let Some(w) = &m.where_ {
                current = current.into_iter().filter(|r| eval(w, r).as_bool().unwrap_or(false)).collect();
            }
            state.rows = current;
            Ok(state)
        }
        Statement::Set(s) => {
            for r in &mut state.rows {
                for a in &s.assigns {
                    let val = eval(&a.value, r);
                    apply_assign(storage, r, a, val)?;
                }
            }
            Ok(state)
        }
        Statement::Delete(d) => {
            let mut nodes = std::collections::BTreeSet::<i64>::new();
            let mut edges = std::collections::BTreeSet::<i64>::new();
            for r in &state.rows {
                for v in &d.variables {
                    if let Some(val) = r.get(v) {
                        if let Some(Value::Object(map)) = val.get("node") {
                            if let Some(id) = map.get("id").and_then(|x| x.as_i64()) { nodes.insert(id); }
                        }
                        if val.get("source").is_some() && val.get("target").is_some() {
                            if let Some(id) = val.get("id").and_then(|x| x.as_i64()) { edges.insert(id); }
                        }
                    }
                }
            }
            if d.detach {
                for &n in &nodes {
                    let incident = storage.list_edges(Some(NodeId(n)), None)?;
                    for e in incident { edges.insert(e.id.0); }
                    let incident2 = storage.list_edges(None, Some(NodeId(n)))?;
                    for e in incident2 { edges.insert(e.id.0); }
                }
            }
            for &id in &edges { storage.delete_edge(EdgeId(id))?; }
            for &id in &nodes { storage.delete_node(NodeId(id))?; }
            Ok(state)
        }
    }
}

fn execute_create(storage: &Storage, p: &Pattern, rows: Vec<Row>) -> Result<Vec<Row>> {
    let mut out = rows;
    let first = &p.segments[0].node;
    let first_labels: Vec<String> = first.labels.clone();
    let first_props = json_from_properties(&first.properties, &BTreeMap::new())?;
    let first_id = storage.create_node(&first_labels, &first_props)?;
    for r in &mut out {
        r.insert(first.variable.clone(), node_to_value(storage.get_node(first_id)?));
    }
    // The parser attaches the incoming edge to the *next* segment, not the previous one.
    let mut prev_id = first_id;
    for seg in &p.segments[1..] {
        let edge_pat = seg.edge.as_ref().expect("non-first segment must have incoming edge");
        let next = &seg.node;
        let next_labels: Vec<String> = next.labels.clone();
        let next_props = json_from_properties(&next.properties, &BTreeMap::new())?;
        let next_id = storage.create_node(&next_labels, &next_props)?;
        let edge_props = serde_json::json!({});
        let edge_id = storage.create_edge(
            prev_id,
            next_id,
            edge_pat.types.first().map(|s| s.as_str()),
            &edge_props,
        )?;
        for r in &mut out {
            if let Some(var) = &edge_pat.variable {
                if let Ok(edges) = storage.list_edges(Some(prev_id), Some(next_id)) {
                    if let Some(e) = edges.into_iter().find(|e| e.id == edge_id) {
                        r.insert(var.clone(), edge_to_value(e));
                    }
                }
            }
            r.insert(next.variable.clone(), node_to_value(storage.get_node(next_id)?));
        }
        prev_id = next_id;
    }
    Ok(out)
}

fn execute_match(storage: &Storage, p: &Pattern, base_rows: &[Row], optional: bool) -> Result<Vec<Row>> {
    let first_node = &p.segments[0].node;
    let mut nodes: Vec<Node> = if first_node.labels.is_empty() {
        storage.list_nodes_by_label("")?
    } else {
        let mut all: Vec<Node> = Vec::new();
        for lbl in &first_node.labels {
            for n in storage.list_nodes_by_label(lbl)? { all.push(n); }
        }
        all.sort_by_key(|n| n.id.0);
        all.dedup_by_key(|n| n.id.0);
        all
    };
    // Apply inline property filters from the node pattern, e.g.
    // `MATCH (n:Person {name: "Alice"})`. Properties are matched against the
    // node's stored property map.
    if !first_node.properties.is_empty() {
        nodes.retain(|n| properties_match(&n.properties, &first_node.properties));
    }
    let mut out = Vec::new();
    let bases: Vec<Row> = if base_rows.is_empty() { vec![Row::new()] } else { base_rows.to_vec() };
    for n in nodes {
        for base in &bases {
            let mut row = base.clone();
            row.insert(first_node.variable.clone(), node_to_value(n.clone()));
            match extend_pattern(storage, &p.segments[1..], n.id, &row) {
                Some(rows) => for r in rows { out.push(r); },
                None => {
                    if optional {
                        out.push(row);
                    }
                }
            }
        }
    }
    if optional && out.is_empty() && !base_rows.is_empty() {
        for b in base_rows { out.push(b.clone()); }
    }
    Ok(out)
}

/// Return true if every `(key, expr)` in `filters` matches the corresponding
/// property in `props`. `expr` must be a literal for this v0.1 path.
fn properties_match(props: &Value, filters: &[(String, Expr)]) -> bool {
    filters.iter().all(|(k, e)| {
        let want = match e { Expr::Literal(v) => v, _ => return false };
        match resolve_path(props, k) {
            Some(got) => json_eq(got, want),
            None => false,
        }
    })
}

fn json_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        _ => a == b,
    }
}

fn extend_pattern(storage: &Storage, segs: &[Segment], anchor: NodeId, row: &Row) -> Option<Vec<Row>> {
    if segs.is_empty() { return Some(vec![row.clone()]); }
    let edge_pat = segs[0].edge.as_ref().expect("non-first segment must have edge");
    let next_node = &segs[0].node;
    let candidates = match edge_pat.direction {
        Direction::Out => storage.list_edges(Some(anchor), None).unwrap_or_default(),
        Direction::In => storage.list_edges(None, Some(anchor)).unwrap_or_default(),
        Direction::Either => {
            let mut v = storage.list_edges(Some(anchor), None).unwrap_or_default();
            v.extend(storage.list_edges(None, Some(anchor)).unwrap_or_default());
            v
        }
    };
    let mut out = Vec::new();
    for edge in candidates {
        if !edge_pat.types.is_empty() {
            let t = edge.edge_type.clone().unwrap_or_default();
            if !edge_pat.types.iter().any(|x| x == &t) { continue; }
        }
        let other = if edge.source == anchor { edge.target } else { edge.source };
        if !next_node.labels.is_empty() {
            let node = storage.get_node(other).ok()?;
            if !next_node.labels.iter().any(|l| node.labels.iter().any(|x| x == l)) { continue; }
        }
        let mut new_row = row.clone();
        if let Some(var) = &edge_pat.variable {
            new_row.insert(var.clone(), edge_to_value(edge.clone()));
        }
        new_row.insert(next_node.variable.clone(), node_to_value(storage.get_node(other).ok()?));
        if let Some((lo, hi)) = edge_pat.variable_length {
            let mut frontier: Vec<(Row, NodeId, usize)> = vec![(new_row.clone(), other, 1)];
            let mut accepted: Vec<Row> = Vec::new();
            if lo == 0 { accepted.push(new_row.clone()); }
            while let Some((cur_row, cur_anchor, depth)) = frontier.pop() {
                if depth >= lo {
                    // Bind the destination variable to the current node at this depth
                    let mut final_row = cur_row.clone();
                    final_row.insert(next_node.variable.clone(), node_to_value(storage.get_node(cur_anchor).ok()?));
                    accepted.push(final_row);
                }
                if depth >= hi.unwrap_or(usize::MAX) { continue; }
                let next_edges = match edge_pat.direction {
                    Direction::Out => storage.list_edges(Some(cur_anchor), None).unwrap_or_default(),
                    Direction::In => storage.list_edges(None, Some(cur_anchor)).unwrap_or_default(),
                    Direction::Either => {
                        let mut v = storage.list_edges(Some(cur_anchor), None).unwrap_or_default();
                        v.extend(storage.list_edges(None, Some(cur_anchor)).unwrap_or_default());
                        v
                    }
                };
                for e in next_edges {
                    // Filter by edge type on EVERY hop, not just the first
                    if !edge_pat.types.is_empty() {
                        let t = e.edge_type.clone().unwrap_or_default();
                        if !edge_pat.types.iter().any(|x| x == &t) { continue; }
                    }
                    let next_id = if e.source == cur_anchor { e.target } else { e.source };
                    let mut deeper = cur_row.clone();
                    if let Some(var) = &edge_pat.variable {
                        deeper.insert(format!("{}_d{}", var, depth), edge_to_value(e));
                    }
                    frontier.push((deeper, next_id, depth + 1));
                }
            }
            for r in accepted { out.push(r); }
        } else {
            if let Some(rows) = extend_pattern(storage, &segs[1..], other, &new_row) {
                out.extend(rows);
            }
        }
    }
    if out.is_empty() { None } else { Some(out) }
}

fn merge(base: &Row, ext: &Row) -> Row {
    let mut m = base.clone();
    for (k, v) in ext { m.insert(k.clone(), v.clone()); }
    m
}

fn node_to_value(n: Node) -> Value {
    serde_json::json!({
        "node": {
            "id": n.id.0,
            "labels": n.labels,
            "properties": n.properties,
        }
    })
}

fn edge_to_value(e: crate::edge::Edge) -> Value {
    serde_json::json!({
        "id": e.id.0,
        "source": e.source.0,
        "target": e.target.0,
        "edge_type": e.edge_type,
        "properties": e.properties,
    })
}

fn json_from_properties(props: &[(String, Expr)], row: &Row) -> Result<Value> {
    let mut map = serde_json::Map::new();
    for (k, v) in props { map.insert(k.clone(), eval(v, row)); }
    Ok(Value::Object(map))
}

fn apply_assign(storage: &Storage, row: &mut Row, a: &Assign, val: Value) -> Result<()> {
    use crate::sql::SqlValue;
    let bound = row.get(&a.variable).cloned();
    // Node binding: {"node": {"id": ..., "labels": ..., "properties": ...}}
    if let Some(Value::Object(node_obj)) = &bound {
        if let Some(Value::Object(node)) = node_obj.get("node") {
            if let Some(id) = node.get("id").and_then(|x| x.as_i64()) {
                let mut n = storage.get_node(NodeId(id))?;
                let new_props = if let Some(prop) = &a.property {
                    let mut m = match n.properties {
                        Value::Object(map) => map,
                        _ => serde_json::Map::new(),
                    };
                    m.insert(prop.clone(), val.clone());
                    Value::Object(m)
                } else { val.clone() };
                let labels = n.labels.clone();
                storage.exec.execute(
                    &format!("UPDATE {}_nodes SET labels = ?1, properties = ?2 WHERE id = ?3", storage.prefix),
                    &[SqlValue::Text(serde_json::to_string(&labels)?), SqlValue::Text(serde_json::to_string(&new_props)?), SqlValue::from(n.id.0)],
                )?;
                n.properties = new_props;
                row.insert(a.variable.clone(), node_to_value(n));
                return Ok(());
            }
        }
    }
    // Edge binding: {"id": ..., "source": ..., "target": ..., "edge_type": ..., "properties": ...}
    if let Some(Value::Object(edge_obj)) = &bound {
        if let Some(id) = edge_obj.get("id").and_then(|x| x.as_i64()) {
            if edge_obj.get("source").is_some() && edge_obj.get("target").is_some() {
                // This is an edge binding. Update its properties.
                let old_props: Value = edge_obj.get("properties").cloned().unwrap_or(Value::Object(Default::default()));
                let new_props = if let Some(prop) = &a.property {
                    let mut m = match old_props {
                        Value::Object(map) => map,
                        _ => serde_json::Map::new(),
                    };
                    m.insert(prop.clone(), val.clone());
                    Value::Object(m)
                } else { val.clone() };
                storage.exec.execute(
                    &format!("UPDATE {}_edges SET properties = ?1 WHERE id = ?2", storage.prefix),
                    &[SqlValue::Text(serde_json::to_string(&new_props)?), SqlValue::from(id)],
                )?;
                let mut updated = edge_obj.clone();
                updated.insert("properties".into(), new_props);
                row.insert(a.variable.clone(), Value::Object(updated));
                return Ok(());
            }
        }
    }
    // Fallback: just record the value in the row (unbound or scalar variable).
    row.insert(a.variable.clone(), val);
    Ok(())
}

fn eval(e: &Expr, row: &Row) -> Value {
    match e {
        Expr::Literal(v) => v.clone(),
        Expr::Variable(v, prop) => {
            if let Some(bound) = row.get(v) {
                if let Some(p) = prop {
                    // Node binding: {node: {id, labels, properties, ...}}
                    if let Some(node) = bound.get("node") {
                        if let Some(props) = node.get("properties") {
                            if let Some(r) = resolve_path(props, p) { return r.clone(); }
                        }
                        if let Some(r) = resolve_path(node, p) { return r.clone(); }
                    }
                    // Edge binding: {id, source, target, edge_type, properties}
                    if let Some(props) = bound.get("properties") {
                        if let Some(r) = resolve_path(props, p) { return r.clone(); }
                    }
                    // Direct path on the binding itself
                    if let Some(r) = resolve_path(bound, p) { return r.clone(); }
                    return Value::Null;
                }
                bound.clone()
            } else { Value::Null }
        }
        Expr::Cmp(a, op, b) => {
            let av = eval(a, row);
            let bv = eval(b, row);
            Value::Bool(compare(&av, *op, &bv))
        }
        Expr::And(a, b) => Value::Bool(eval(a, row).as_bool().unwrap_or(false) && eval(b, row).as_bool().unwrap_or(false)),
        Expr::Or(a, b) => Value::Bool(eval(a, row).as_bool().unwrap_or(false) || eval(b, row).as_bool().unwrap_or(false)),
        Expr::Not(a) => Value::Bool(!eval(a, row).as_bool().unwrap_or(false)),
        Expr::VectorNn { .. } => Value::Bool(true),
    }
}

fn compare(a: &Value, op: CmpOp, b: &Value) -> bool {
    use Value::*;
    match (a, b) {
        (Null, _) | (_, Null) => matches!(op, CmpOp::Eq) && a == b,
        (Bool(x), Bool(y)) => match op { CmpOp::Eq => x == y, CmpOp::Ne => x != y, _ => false },
        (Number(x), Number(y)) => {
            let xf = x.as_f64().unwrap_or(0.0);
            let yf = y.as_f64().unwrap_or(0.0);
            match op {
                CmpOp::Eq => xf == yf, CmpOp::Ne => xf != yf,
                CmpOp::Lt => xf < yf, CmpOp::Le => xf <= yf, CmpOp::Gt => xf > yf, CmpOp::Ge => xf >= yf,
            }
        }
        (String(x), String(y)) => match op {
            CmpOp::Eq => x == y, CmpOp::Ne => x != y,
            CmpOp::Lt => x < y, CmpOp::Le => x <= y, CmpOp::Gt => x > y, CmpOp::Ge => x >= y,
        },
        _ => false,
    }
}

fn materialize(rows: &[Row], ret: &Return) -> Result<ResultSet> {
    let columns: Vec<String> = ret.items.iter().enumerate().map(|(i, it)| {
        it.alias.clone().unwrap_or_else(|| match &it.expr {
            Expr::Variable(v, p) => match p { Some(p) => format!("{}.{}", v, p), None => v.clone() },
            _ => format!("col{}", i),
        })
    }).collect();
    let mut out_rows: Vec<Vec<Value>> = Vec::new();
    for row in rows {
        let mut out_row = Vec::new();
        for it in &ret.items { out_row.push(eval(&it.expr, row)); }
        out_rows.push(out_row);
    }
    // simple single-key ORDER BY evaluated against the bound rows
    if let Some(ob) = ret.order_by.first() {
        let desc = ob.desc;
        let mut indexed: Vec<(Vec<Value>, Row)> = out_rows.iter().cloned().zip(rows.iter().cloned()).collect();
        indexed.sort_by(|x, y| {
            let va = eval(&ob.expr, &x.1);
            let vb = eval(&ob.expr, &y.1);
            let ord = cmp_json(&va, &vb);
            if desc { ord.reverse() } else { ord }
        });
        out_rows = indexed.into_iter().map(|(r, _)| r).collect();
    }
    if let Some(n) = ret.skip { if n < out_rows.len() { out_rows.drain(..n); } else { out_rows.clear(); } }
    if let Some(n) = ret.limit { out_rows.truncate(n); }
    Ok(ResultSet { columns, rows: out_rows })
}

fn cmp_json(a: &Value, b: &Value) -> std::cmp::Ordering {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64().partial_cmp(&y.as_f64()).unwrap_or(std::cmp::Ordering::Equal),
        (Value::String(x), Value::String(y)) => x.cmp(y),
        (Value::Bool(x), Value::Bool(y)) => x.cmp(y),
        (Value::Null, Value::Null) => std::cmp::Ordering::Equal,
        (Value::Null, _) => std::cmp::Ordering::Less,
        (_, Value::Null) => std::cmp::Ordering::Greater,
        _ => std::cmp::Ordering::Equal,
    }
}

impl fmt::Display for ResultSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} rows", self.rows.len())?;
        for r in &self.rows {
            for (i, v) in r.iter().enumerate() { if i > 0 { write!(f, " | ")?; } write!(f, "{}", v)?; }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::{SqlExecutor, SqlValue};
    use serde_json::json;

    /// Minimal in-memory `SqlExecutor` that returns results as JSON arrays.
    /// This keeps `gvec-core` free of any database dependency.
    pub struct MockExec(std::sync::Mutex<MockState>);
    pub struct MockState {
        rows: std::collections::BTreeMap<(String, i64), Vec<SqlValue>>,
        next_id: i64,
        schemas: std::collections::HashSet<String>,
    }

    impl MockExec {
        pub fn new() -> Self { Self(std::sync::Mutex::new(MockState { rows: Default::default(), next_id: 0, schemas: Default::default() })) }
    }

    fn sqlvalue_to_json(v: &SqlValue) -> serde_json::Value {
        match v {
            SqlValue::Null => serde_json::Value::Null,
            SqlValue::Integer(n) => serde_json::Value::from(*n),
            SqlValue::Real(f) => serde_json::json!(*f),
            SqlValue::Text(s) => serde_json::Value::String(s.clone()),
            SqlValue::Blob(b) => serde_json::Value::String(format!("<{} bytes>", b.len())),
        }
    }

        impl SqlExecutor for MockExec {
        fn execute(&self, sql: &str, params: &[SqlValue]) -> Result<usize> {
            let mut s = self.0.lock().unwrap();
            let sql = sql.trim();
            if sql.starts_with("CREATE TABLE") { return Ok(0); }
            if sql.starts_with("DROP TABLE") { return Ok(0); }
            if sql.starts_with("INSERT INTO") {
                if let Some(idx) = sql.find("INTO ") {
                    let after = &sql[idx + 5..];
                    let name: String = after.chars().take_while(|c| !c.is_whitespace() && *c != '(').collect();
                    s.next_id += 1;
                    let id = s.next_id;
                    // store the row with id prepended so SELECT returns [id, ...params]
                    let mut row = vec![SqlValue::Integer(id)];
                    row.extend(params.iter().cloned());
                    s.rows.insert((name, id), row);
                    return Ok(1);
                }
            }
            if sql.starts_with("UPDATE") { return Ok(1); }
            if sql.starts_with("DELETE FROM") { return Ok(1); }
            Ok(0)
        }
        fn query_one_json(&self, sql: &str, params: &[SqlValue]) -> Result<Option<Vec<serde_json::Value>>> {
            let s = self.0.lock().unwrap();
            if sql.contains("COUNT(*)") {
                let name = extract_table(sql).unwrap_or_default();
                let count = s.rows.iter().filter(|((t, _), _)| t == &name).count() as i64;
                return Ok(Some(vec![serde_json::Value::from(count)]));
            }
            if sql.to_ascii_uppercase().contains("SELECT 1 FROM") {
                // existence check: any row in this table?
                let name = extract_table(sql).unwrap_or_default();
                let exists = s.rows.iter().any(|((t, _), _)| t == &name);
                if exists { return Ok(Some(vec![serde_json::Value::from(1i64)])); }
                return Ok(None);
            }
            let name = extract_table(sql).unwrap_or_default();
            let id = match params.first() { Some(SqlValue::Integer(n)) => *n, _ => 0 };
            for ((t, k), row) in &s.rows {
                if t == &name && *k == id {
                    return Ok(Some(row.iter().map(sqlvalue_to_json).collect()));
                }
            }
            Ok(None)
        }
        fn query_all_json(&self, sql: &str, _params: &[SqlValue]) -> Result<Vec<Vec<serde_json::Value>>> {
            let s = self.0.lock().unwrap();
            let name = extract_table(sql).unwrap_or_default();
            let mut out = Vec::new();
            for ((t, _k), row) in &s.rows {
                if t == &name {
                    out.push(row.iter().map(sqlvalue_to_json).collect());
                }
            }
            Ok(out)
        }
        fn last_insert_rowid(&self) -> i64 { self.0.lock().unwrap().next_id }
    }

    fn extract_table(sql: &str) -> Option<String> {
        let upper = sql.to_ascii_uppercase();
        // For SELECT COUNT(*) FROM <tbl>, extract the table from after FROM.
        for kw in ["FROM ", "INTO ", "UPDATE "] {
            if let Some(i) = upper.find(kw) {
                let after = &sql[i + kw.len()..];
                // skip words until we find the next non-keyword identifier after COUNT / WHERE / etc.
                let name: String = after.chars()
                    .skip_while(|c| c.is_whitespace())
                    .take_while(|c| !c.is_whitespace() && *c != '(' && *c != ';')
                    .collect();
                // filter out COUNT / WHERE / etc.
                if !matches!(name.to_ascii_uppercase().as_str(), "COUNT"|"WHERE"|"SET"|"DISTINCT"|"ALL"|"SELECT") {
                    return Some(name);
                }
            }
        }
        None
    }

    use std::sync::atomic::{AtomicUsize, Ordering};
    static TEST_ID: AtomicUsize = AtomicUsize::new(0);

    fn fresh_storage() -> Storage {
        // Per-test exec. Different tests use different prefixes so they don't collide.
        let id = TEST_ID.fetch_add(1, Ordering::SeqCst);
        Storage::new(MockExec::new(), &format!("t{id}"))
    }

    #[test]
    fn create_simple_node() {
        let s = fresh_storage();
        s.ensure_schema().unwrap();
        let q = Parser::new(r#"CREATE (a:Person {name: "Alice", age: 30})"#).unwrap().parse_query().unwrap();
        execute(&s, &q).unwrap();
        assert_eq!(s.count_nodes().unwrap(), 1);
        let n = s.get_node(NodeId(1)).unwrap();
        assert_eq!(n.labels, vec!["Person"]);
        assert_eq!(n.properties["name"], "Alice");
        assert_eq!(n.properties["age"], 30);
    }

    #[test]
    fn match_with_where_and_return() {
        let s = fresh_storage();
        s.ensure_schema().unwrap();
        let q = Parser::new(r#"CREATE (a:Person {name: "Alice", age: 30})"#).unwrap().parse_query().unwrap();
        execute(&s, &q).unwrap();
        let q2 = Parser::new(r#"MATCH (n:Person) WHERE n.age > 25 RETURN n.name AS name"#).unwrap().parse_query().unwrap();
        let rs = execute(&s, &q2).unwrap();
        assert_eq!(rs.rows.len(), 1);
        assert_eq!(rs.rows[0][0], json!("Alice"));
    }

    #[test]
    fn create_chain_and_match() {
        let s = fresh_storage();
        s.ensure_schema().unwrap();
        let q = Parser::new(r#"CREATE (a:Person {name: "Alice"})-[:KNOWS]->(b:Person {name: "Bob"})"#).unwrap().parse_query().unwrap();
        execute(&s, &q).unwrap();
        assert_eq!(s.count_nodes().unwrap(), 2);
        assert_eq!(s.count_edges().unwrap(), 1);
    }

    #[test]
    fn return_order_limit() {
        let s = fresh_storage();
        s.ensure_schema().unwrap();
        for (n, age) in [("A", 10), ("B", 40), ("C", 25), ("D", 50)] {
            let q = Parser::new(&format!(r#"CREATE (p:Person {{name: "{}", age: {}}})"#, n, age)).unwrap().parse_query().unwrap();
            execute(&s, &q).unwrap();
        }
        let q = Parser::new("MATCH (n:Person) RETURN n.name AS name ORDER BY n.age DESC LIMIT 2").unwrap().parse_query().unwrap();
        let rs = execute(&s, &q).unwrap();
        assert_eq!(rs.rows.len(), 2);
        assert_eq!(rs.rows[0][0], json!("D"));
        assert_eq!(rs.rows[1][0], json!("B"));
    }

    #[test]
    fn optional_match_keeps_null() {
        let s = fresh_storage();
        s.ensure_schema().unwrap();
        let q = Parser::new(r#"CREATE (a:Person {name: "Alice"})"#).unwrap().parse_query().unwrap();
        execute(&s, &q).unwrap();
        let q2 = Parser::new("MATCH (n:Person) OPTIONAL MATCH (n)-[:KNOWS]->(f) RETURN n.name AS name, f.name AS friend").unwrap().parse_query().unwrap();
        let rs = execute(&s, &q2).unwrap();
        assert_eq!(rs.rows.len(), 1);
        assert_eq!(rs.rows[0][1], Value::Null);
    }
}
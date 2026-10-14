//! End-to-end proof of the staging DDL pilot against a real, disposable SQLite
//! file. No external database, credentials, or secrets are involved: the
//! "database" is a temp file created and deleted by the test.
//!
//! Proves:
//! * `dbx_prepare_ddl` executes nothing and returns a proposal + digest.
//! * `dbx_apply_approved_ddl` executes the prepared DDL exactly once, but ONLY
//!   when an external approver has approved the digest. The MCP client holds a
//!   proposal id, never an approval, so it cannot authorize its own write.
//! * Without an approver (deny-all default) nothing executes.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use dbx_core::{models::connection::ConnectionConfig, storage::Storage};
use dbx_mcp::{
    ddl_pilot::{DdlApproval, DdlApprover},
    DbxMcpServer, LocalBackend, McpScope,
};
use rmcp::{model::CallToolRequestParams, ServiceExt};
use serde_json::{json, Map, Value};
use tempfile::tempdir;

/// Stands in for the external authorization boundary. It approves whatever
/// digest the test explicitly records — mirroring an out-of-band approver that
/// saw the proposal — and nothing else.
#[derive(Default)]
struct RecordingApprover {
    approved: Mutex<Vec<String>>,
}

impl RecordingApprover {
    fn approve(&self, digest: &str) {
        self.approved.lock().unwrap().push(digest.to_string());
    }
}

#[async_trait]
impl DdlApprover for RecordingApprover {
    async fn approval_for(&self, digest: &str) -> Option<DdlApproval> {
        self.approved.lock().unwrap().iter().any(|approved| approved == digest).then(|| DdlApproval {
            approver: "external-approver".to_string(),
        })
    }
}

fn args(value: Value) -> Map<String, Value> {
    value.as_object().cloned().unwrap_or_default()
}

fn text_of(result: &rmcp::model::CallToolResult) -> String {
    result.content[0].as_text().map(|text| text.text.clone()).unwrap_or_default()
}

/// Boot a real MCP server over a temp SQLite connection, wired to `approver`.
async fn boot(
    directory: &tempfile::TempDir,
    approver: Arc<dyn DdlApprover>,
) -> (rmcp::service::RunningService<rmcp::RoleClient, ()>, tokio::task::JoinHandle<()>) {
    let db_path = directory.path().join("dbx.db");
    let sqlite_path = directory.path().join("staging.sqlite");
    std::fs::File::create(&sqlite_path).expect("create disposable SQLite file");
    let storage = Storage::open(&db_path).await.expect("open storage");
    let connection: ConnectionConfig = serde_json::from_value(json!({
        "id": "staging-sqlite",
        "name": "staging-sqlite",
        "db_type": "sqlite",
        "host": sqlite_path.to_string_lossy(),
        "port": 0,
        "username": "",
        "password": "",
        "database": "main",
        "ssl": false
    }))
    .expect("connection config");
    storage.save_connections(std::slice::from_ref(&connection)).await.expect("save connection");
    storage
        .save_mcp_global_policy(&dbx_core::storage::McpGlobalPolicy {
            read_only: false,
            allow_dangerous_sql: true,
            ..Default::default()
        })
        .await
        .expect("save policy");
    drop(storage);

    let backend = Arc::new(LocalBackend::open(&db_path).await.expect("open local backend"));
    let server = DbxMcpServer::with_runtime_options(backend, McpScope::default(), false).with_ddl_approver(approver);
    let (server_transport, client_transport) = tokio::io::duplex(64 * 1024);
    let server_task = tokio::spawn(async move {
        let service = server.serve(server_transport).await.expect("start MCP server");
        service.waiting().await.expect("run MCP server");
    });
    let client = ().serve(client_transport).await.expect("initialize MCP client");
    (client, server_task)
}

async fn table_exists(client: &rmcp::service::RunningService<rmcp::RoleClient, ()>, table: &str) -> bool {
    let result = client
        .peer()
        .call_tool(CallToolRequestParams::new("dbx_execute_query").with_arguments(args(json!({
            "connection_id": "staging-sqlite",
            "sql": format!("SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"),
        }))))
        .await
        .expect("query sqlite_master");
    text_of(&result).contains(table)
}

#[tokio::test]
async fn approved_ddl_creates_the_table_exactly_once_and_unapproved_does_not() {
    let directory = tempdir().expect("temp dir");
    let approver = Arc::new(RecordingApprover::default());
    let (client, server_task) = boot(&directory, approver.clone()).await;

    // 1. Prepare executes nothing.
    let prepared = client
        .peer()
        .call_tool(CallToolRequestParams::new("dbx_prepare_ddl").with_arguments(args(json!({
            "connection_id": "staging-sqlite",
            "sql": "CREATE TABLE ddl_pilot_proof (id INTEGER)",
        }))))
        .await
        .expect("prepare ddl");
    assert_ne!(prepared.is_error, Some(true), "prepare failed: {}", text_of(&prepared));
    let structured = prepared.structured_content.as_ref().expect("structured content");
    let proposal_id = structured["proposal_id"].as_str().expect("proposal_id").to_string();
    let digest = structured["digest"].as_str().expect("digest").to_string();
    assert!(
        !table_exists(&client, "ddl_pilot_proof").await,
        "prepare must not execute anything"
    );

    // 2. Applying before any external approval is refused and creates nothing.
    let unapproved = client
        .peer()
        .call_tool(CallToolRequestParams::new("dbx_apply_approved_ddl").with_arguments(args(json!({
            "proposal_id": proposal_id,
        }))))
        .await
        .expect("apply without approval");
    assert_eq!(unapproved.is_error, Some(true), "unapproved apply must fail");
    assert!(!table_exists(&client, "ddl_pilot_proof").await, "unapproved apply must not execute");

    // 3. Re-prepare, record the external approval, then apply.
    let prepared = client
        .peer()
        .call_tool(CallToolRequestParams::new("dbx_prepare_ddl").with_arguments(args(json!({
            "connection_id": "staging-sqlite",
            "sql": "CREATE TABLE ddl_pilot_proof (id INTEGER)",
        }))))
        .await
        .expect("re-prepare ddl");
    let structured = prepared.structured_content.as_ref().expect("structured content");
    let proposal_id = structured["proposal_id"].as_str().expect("proposal_id").to_string();
    let next_digest = structured["digest"].as_str().expect("digest");
    assert_ne!(next_digest, digest, "each proposal requires its own approval");
    approver.approve(next_digest);

    let applied = client
        .peer()
        .call_tool(CallToolRequestParams::new("dbx_apply_approved_ddl").with_arguments(args(json!({
            "proposal_id": proposal_id,
        }))))
        .await
        .expect("apply approved ddl");
    assert_ne!(applied.is_error, Some(true), "approved apply failed: {}", text_of(&applied));
    assert!(table_exists(&client, "ddl_pilot_proof").await, "approved apply must create the table");

    // 4. The proposal is single-use: a replay executes nothing new.
    let replay = client
        .peer()
        .call_tool(CallToolRequestParams::new("dbx_apply_approved_ddl").with_arguments(args(json!({
            "proposal_id": proposal_id,
        }))))
        .await
        .expect("replay apply");
    assert_eq!(replay.is_error, Some(true), "a proposal is single-use");

    client.cancel().await.expect("close client");
    server_task.abort();
}

#[tokio::test]
async fn deny_all_default_approver_executes_nothing() {
    let directory = tempdir().expect("temp dir");
    // No approver wired: the server defaults to deny-all.
    let (client, server_task) = boot(&directory, Arc::new(dbx_mcp::ddl_pilot::DenyAllApprover)).await;

    let prepared = client
        .peer()
        .call_tool(CallToolRequestParams::new("dbx_prepare_ddl").with_arguments(args(json!({
            "connection_id": "staging-sqlite",
            "sql": "CREATE TABLE deny_all_proof (id INTEGER)",
        }))))
        .await
        .expect("prepare ddl");
    let structured = prepared.structured_content.as_ref().expect("structured content");
    let proposal_id = structured["proposal_id"].as_str().expect("proposal_id").to_string();

    let applied = client
        .peer()
        .call_tool(CallToolRequestParams::new("dbx_apply_approved_ddl").with_arguments(args(json!({
            "proposal_id": proposal_id,
        }))))
        .await
        .expect("apply with deny-all approver");
    assert_eq!(applied.is_error, Some(true), "deny-all must refuse");
    assert!(!table_exists(&client, "deny_all_proof").await, "deny-all must not execute");

    client.cancel().await.expect("close client");
    server_task.abort();
}

//! End-to-end proof of the out-of-band DDL approval flow (`DBX_DDL_*`) against a
//! disposable SQLite file. Lives in its own test binary because the wiring is
//! read from process-global environment variables.
//!
//! Proves:
//! * The stored MCP policy stays read-only: the generic query tool cannot run DDL.
//! * `dbx_prepare_ddl` targets only the configured DDL connection, executes
//!   nothing, and publishes the server-resolved proposal to `pending/`.
//! * An apply before approval is refused and leaves the proposal pending.
//! * Only a record in `approved/` whose digest matches authorizes execution;
//!   it runs exactly once and the approval is moved to `used/`.

use std::sync::Arc;

use dbx_core::{models::connection::ConnectionConfig, storage::Storage};
use dbx_mcp::{DbxMcpServer, LocalBackend, McpScope};
use rmcp::{model::CallToolRequestParams, ServiceExt};
use serde_json::{json, Map, Value};
use tempfile::tempdir;

fn args(value: Value) -> Map<String, Value> {
    value.as_object().cloned().unwrap_or_default()
}

fn text_of(result: &rmcp::model::CallToolResult) -> String {
    result.content[0].as_text().map(|text| text.text.clone()).unwrap_or_default()
}

fn sqlite_connection(id: &str, path: &std::path::Path) -> ConnectionConfig {
    serde_json::from_value(json!({
        "id": id,
        "name": id,
        "db_type": "sqlite",
        "host": path.to_string_lossy(),
        "port": 0,
        "username": "",
        "password": "",
        "database": "main",
        "ssl": false
    }))
    .expect("connection config")
}

async fn call(
    client: &rmcp::service::RunningService<rmcp::RoleClient, ()>,
    tool: &'static str,
    arguments: Value,
) -> rmcp::model::CallToolResult {
    client.peer().call_tool(CallToolRequestParams::new(tool).with_arguments(args(arguments))).await.expect(tool)
}

async fn table_exists(client: &rmcp::service::RunningService<rmcp::RoleClient, ()>, table: &str) -> bool {
    let result = call(
        client,
        "dbx_execute_query",
        json!({
            "connection_id": "staging-ro",
            "sql": format!("SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"),
        }),
    )
    .await;
    assert_ne!(result.is_error, Some(true), "read query failed: {}", text_of(&result));
    text_of(&result).contains(table)
}

fn write_approval(dir: &std::path::Path, digest: &str, record_digest: &str) {
    let approved = dir.join("approved");
    std::fs::create_dir_all(&approved).expect("approved dir");
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    std::fs::write(
        approved.join(format!("{digest}.json")),
        serde_json::to_vec(&json!({
            "digest": record_digest,
            "approver": "telegram:1",
            "approved_at_unix": now,
        }))
        .unwrap(),
    )
    .expect("write approval");
}

#[tokio::test]
async fn out_of_band_approval_gates_ddl_on_the_dedicated_connection() {
    let directory = tempdir().expect("temp dir");
    let db_path = directory.path().join("dbx.db");
    let sqlite_path = directory.path().join("staging.sqlite");
    std::fs::File::create(&sqlite_path).expect("create disposable SQLite file");
    let approval_dir = directory.path().join("approvals");

    let storage = Storage::open(&db_path).await.expect("open storage");
    storage
        .save_connections(&[sqlite_connection("staging-ro", &sqlite_path), sqlite_connection("staging-ddl", &sqlite_path)])
        .await
        .expect("save connections");
    // The stored policy is what every generic tool sees: read-only.
    storage
        .save_mcp_global_policy(&dbx_core::storage::McpGlobalPolicy { read_only: true, ..Default::default() })
        .await
        .expect("save policy");
    drop(storage);

    std::env::set_var("DBX_DDL_CONNECTION_ID", "staging-ddl");
    std::env::set_var("DBX_DDL_APPROVAL_DIR", &approval_dir);
    let backend = Arc::new(LocalBackend::open(&db_path).await.expect("open local backend"));
    let server = DbxMcpServer::with_runtime_options(backend, McpScope::default(), false);
    let (server_transport, client_transport) = tokio::io::duplex(64 * 1024);
    let server_task = tokio::spawn(async move {
        let service = server.serve(server_transport).await.expect("start MCP server");
        service.waiting().await.expect("run MCP server");
    });
    let client = ().serve(client_transport).await.expect("initialize MCP client");

    // 1. Generic tools stay read-only, even against the DDL connection.
    for connection_id in ["staging-ro", "staging-ddl"] {
        let blocked =
            call(&client, "dbx_execute_query", json!({ "connection_id": connection_id, "sql": "CREATE TABLE bypass (id INTEGER)" }))
                .await;
        assert_eq!(blocked.is_error, Some(true), "generic DDL must be blocked on {connection_id}");
    }
    assert!(!table_exists(&client, "bypass").await);

    // 2. Prepare ignores the caller's selector, executes nothing, publishes pending.
    let prepared = call(
        &client,
        "dbx_prepare_ddl",
        json!({ "connection_id": "staging-ro", "database": "main", "sql": "CREATE TABLE approved_proof (id INTEGER)" }),
    )
    .await;
    assert_ne!(prepared.is_error, Some(true), "prepare failed: {}", text_of(&prepared));
    let structured = prepared.structured_content.clone().expect("structured content");
    let proposal_id = structured["proposal_id"].as_str().unwrap().to_string();
    let digest = structured["digest"].as_str().unwrap().to_string();
    assert_eq!(structured["connection_id"], "staging-ddl");
    assert!(!table_exists(&client, "approved_proof").await, "prepare must not execute");
    let pending: Value = serde_json::from_slice(
        &std::fs::read(approval_dir.join("pending").join(format!("{proposal_id}.json"))).expect("pending record"),
    )
    .unwrap();
    assert_eq!(pending["sql"], "CREATE TABLE approved_proof (id INTEGER)");
    assert_eq!(pending["digest"], digest.as_str());
    assert_eq!(pending["connection_id"], "staging-ddl");

    // 3. Early apply is refused and does NOT burn the proposal.
    let early = call(&client, "dbx_apply_approved_ddl", json!({ "proposal_id": proposal_id })).await;
    assert_eq!(early.is_error, Some(true));
    assert!(text_of(&early).contains("DDL_NOT_APPROVED") || text_of(&early).contains("not been approved"), "{}", text_of(&early));

    // 4. A record whose content names another digest is not an approval.
    write_approval(&approval_dir, &digest, &"0".repeat(64));
    let forged = call(&client, "dbx_apply_approved_ddl", json!({ "proposal_id": proposal_id })).await;
    assert_eq!(forged.is_error, Some(true), "mismatched approval must be rejected");
    assert!(!table_exists(&client, "approved_proof").await);

    // 5. The real approval executes exactly once and is spent.
    write_approval(&approval_dir, &digest, &digest);
    let applied = call(&client, "dbx_apply_approved_ddl", json!({ "proposal_id": proposal_id })).await;
    assert_ne!(applied.is_error, Some(true), "approved apply failed: {}", text_of(&applied));
    assert!(table_exists(&client, "approved_proof").await);
    assert!(!approval_dir.join("approved").join(format!("{digest}.json")).exists(), "approval must be spent");
    assert!(approval_dir.join("used").join(format!("{digest}.json")).exists());

    let replay = call(&client, "dbx_apply_approved_ddl", json!({ "proposal_id": proposal_id })).await;
    assert_eq!(replay.is_error, Some(true), "a proposal is single-use");

    client.cancel().await.expect("close client");
    server_task.abort();
}

//! Staging-only prepare/apply seam for single-statement DDL.
//!
//! This module deliberately implements **no approval of its own**. The MCP
//! client (an agent) can *prepare* a proposal and *request* execution, but it
//! can never mint the approval that `apply` requires: that comes only from a
//! [`DdlApprover`], which is the external authorization boundary. The default
//! approver denies everything, so shipping this tool without wiring a real
//! approver executes nothing.
//!
//! The general production-write guard in `validate_sql_policy` is untouched and
//! still runs on every apply, so a production target stays `PRODUCTION_WRITE_BLOCKED`
//! even with an approval.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use uuid::Uuid;

use async_trait::async_trait;
use dbx_core::{
    models::connection::DatabaseType,
    sql_risk::{classify_sql_risk_for_database, SqlRisk},
};

/// How long a prepared proposal stays applicable. Long enough for an agent to
/// present it and for an out-of-band approver to answer, short enough that a
/// stale handle cannot be replayed much later.
pub const DDL_PROPOSAL_TTL: Duration = Duration::from_secs(600);
/// Ceiling on simultaneously pending proposals so a prepare loop cannot grow
/// the in-memory map without bound.
pub const DDL_PROPOSAL_LIMIT: usize = 64;

/// One prepared DDL statement, bound server-side to its exact target and bytes.
#[derive(Debug, Clone)]
pub struct DdlProposal {
    pub connection_id: String,
    pub database: String,
    /// The exact statement text that was prepared. Apply re-sends these bytes
    /// verbatim, so what an approver saw is what the database receives.
    pub sql: String,
    /// SHA-256 over `connection_id`, `database` and the literal SQL. This is the
    /// value an external approver signs; a changed target or statement yields a
    /// different digest and therefore no approval.
    pub digest: String,
    pub created_at: Instant,
}

/// A recorded external approval for one digest. The agent cannot construct one
/// of these — only a [`DdlApprover`] implementation can.
#[derive(Debug, Clone)]
pub struct DdlApproval {
    /// Who approved (e.g. an SSO subject). Carried into the audit trail.
    pub approver: String,
}

/// The external authorization boundary. Implementations live outside the MCP
/// client's trust domain (a privileged service, a signed approval store, an
/// out-of-band workflow). The agent never holds an approval token.
#[async_trait]
pub trait DdlApprover: Send + Sync {
    /// Return the approval recorded for `digest`, if any and still valid.
    async fn approval_for(&self, digest: &str) -> Option<DdlApproval>;

    /// Mark the approval for `digest` as spent once its proposal is claimed, so
    /// a recorded approval can never authorize a second execution.
    async fn consume(&self, _digest: &str) {}
}

/// Fail-closed default: approves nothing. The DDL tools are inert until a real
/// approver is wired, which is the intended state before the external auth
/// boundary and OS isolation exist.
#[derive(Debug, Default)]
pub struct DenyAllApprover;

#[async_trait]
impl DdlApprover for DenyAllApprover {
    async fn approval_for(&self, _digest: &str) -> Option<DdlApproval> {
        None
    }
}

/// In-process store of prepared proposals. The returned proposal id is a
/// *lookup handle*, not an approval: possessing it grants nothing without a
/// matching [`DdlApprover`] entry.
#[derive(Debug)]
pub struct PendingDdlProposals {
    ttl: Duration,
    entries: tokio::sync::Mutex<HashMap<String, DdlProposal>>,
}

impl PendingDdlProposals {
    pub fn new(ttl: Duration) -> Arc<Self> {
        Arc::new(Self { ttl, entries: tokio::sync::Mutex::new(HashMap::new()) })
    }

    /// The process-wide store. Streamable HTTP builds a fresh server for every
    /// request, so a per-instance store would lose a proposal between
    /// `dbx_prepare_ddl` and `dbx_apply_approved_ddl`. Ids are random UUIDs, so
    /// sharing never lets one caller claim another's proposal by guessing.
    pub fn shared() -> Arc<Self> {
        static SHARED: std::sync::OnceLock<Arc<PendingDdlProposals>> = std::sync::OnceLock::new();
        SHARED.get_or_init(|| Self::new(DDL_PROPOSAL_TTL)).clone()
    }

    pub fn ttl_secs(&self) -> u64 {
        self.ttl.as_secs()
    }

    fn is_expired(&self, proposal: &DdlProposal) -> bool {
        proposal.created_at.elapsed() >= self.ttl
    }

    /// Store a proposal under its id, dropping expired entries first and then
    /// the oldest ones until the cap has room.
    pub async fn insert(&self, id: String, proposal: DdlProposal) {
        let mut entries = self.entries.lock().await;
        entries.retain(|_, entry| !self.is_expired(entry));
        while entries.len() >= DDL_PROPOSAL_LIMIT {
            let Some(oldest) =
                entries.iter().min_by_key(|(_, entry)| entry.created_at).map(|(id, _)| id.clone())
            else {
                break;
            };
            entries.remove(&oldest);
        }
        entries.insert(id, proposal);
    }

    /// Atomically claim a proposal. Single-use by design: a replay, a retry, or
    /// two concurrent applies resolve to at most one execution.
    pub async fn take(&self, id: &str) -> Option<DdlProposal> {
        let entry = self.entries.lock().await.remove(id.trim())?;
        (!self.is_expired(&entry)).then_some(entry)
    }

    /// Look at a pending proposal without claiming it. Apply checks approval
    /// against this copy first, so an early apply (before the human approved)
    /// leaves the proposal in place instead of burning it.
    pub async fn peek(&self, id: &str) -> Option<DdlProposal> {
        let entries = self.entries.lock().await;
        entries.get(id.trim()).filter(|entry| !self.is_expired(entry)).cloned()
    }
}

/// Runtime wiring for the approval flow, read from the environment of the
/// privileged MCP process. Both variables are required; with either missing the
/// pilot stays deny-all.
///
/// * `DBX_DDL_CONNECTION_IDS` — comma-separated connections (DDL-capable DB
///   accounts) the DDL tools may execute against. The single-value
///   `DBX_DDL_CONNECTION_ID` is still read and merged in. These connections are
///   deliberately *not* part of the MCP connection scope, so the generic query
///   tools cannot resolve or list them.
/// * `DBX_DDL_APPROVAL_DIR` — directory shared with the out-of-band approver,
///   writable only by the service account. `pending/` receives proposals from
///   this process; `approved/` receives decisions from the approver; spent
///   approvals move to `used/`.
#[derive(Debug, Clone)]
pub struct DdlPilotConfig {
    pub connection_ids: Vec<String>,
    pub approval_dir: PathBuf,
}

impl DdlPilotConfig {
    pub fn from_env() -> Option<Self> {
        let mut connection_ids = Vec::new();
        for name in ["DBX_DDL_CONNECTION_IDS", "DBX_DDL_CONNECTION_ID"] {
            for id in std::env::var(name).unwrap_or_default().split(',').map(str::trim).filter(|id| !id.is_empty()) {
                if !connection_ids.iter().any(|known| known == id) {
                    connection_ids.push(id.to_string());
                }
            }
        }
        let approval_dir = std::env::var_os("DBX_DDL_APPROVAL_DIR").filter(|value| !value.is_empty())?;
        (!connection_ids.is_empty()).then(|| Self { connection_ids, approval_dir: PathBuf::from(approval_dir) })
    }

    pub fn allows(&self, connection_id: &str) -> bool {
        self.connection_ids.iter().any(|id| id == connection_id)
    }
}

pub fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|elapsed| elapsed.as_secs()).unwrap_or(0)
}

/// Digests are lowercase SHA-256 hex. Anything else never becomes a path.
fn is_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Proposal ids are server-minted `ddl-<uuid>` handles. Anything else never
/// becomes a path.
fn is_proposal_id(value: &str) -> bool {
    value.strip_prefix("ddl-").is_some_and(|rest| Uuid::parse_str(rest).is_ok())
}

/// Write `value` to `dir/name` via a temp file and rename. Owner read/write,
/// group read: the approver runs as a separate account in the shared group.
fn write_record(dir: &Path, name: &str, value: &serde_json::Value) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|error| format!("create {}: {error}", dir.display()))?;
    let temp = dir.join(format!(".{name}.tmp"));
    let body = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o640);
        let mut file = options.open(&temp).map_err(|error| format!("write {}: {error}", temp.display()))?;
        std::io::Write::write_all(&mut file, &body).map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
    }
    std::fs::rename(&temp, dir.join(name)).map_err(|error| format!("publish {name}: {error}"))
}

/// Publish a prepared proposal for the out-of-band approver. The approver shows
/// the human exactly these server-resolved fields — never text relayed by the
/// agent.
pub fn publish_pending(dir: &Path, record: &serde_json::Value) -> Result<(), String> {
    let id = record.get("proposal_id").and_then(|value| value.as_str()).unwrap_or_default();
    if !is_proposal_id(id) {
        return Err("invalid proposal id".to_string());
    }
    write_record(&dir.join("pending"), &format!("{id}.json"), record)
}

/// Approver backed by records an out-of-band service writes to
/// `<approval_dir>/approved/<digest>.json`. The MCP client cannot write there:
/// the directory belongs to the service account, not to the agent's OS user.
#[derive(Debug)]
pub struct FileDdlApprover {
    dir: PathBuf,
    ttl: Duration,
}

impl FileDdlApprover {
    pub fn new(dir: PathBuf, ttl: Duration) -> Self {
        Self { dir, ttl }
    }

    fn approved_path(&self, digest: &str) -> PathBuf {
        self.dir.join("approved").join(format!("{digest}.json"))
    }
}

#[async_trait]
impl DdlApprover for FileDdlApprover {
    async fn approval_for(&self, digest: &str) -> Option<DdlApproval> {
        if !is_digest(digest) {
            return None;
        }
        let body = tokio::fs::read(self.approved_path(digest)).await.ok()?;
        let record: serde_json::Value = serde_json::from_slice(&body).ok()?;
        if record.get("digest").and_then(|value| value.as_str()) != Some(digest) {
            return None;
        }
        let approved_at = record.get("approved_at_unix").and_then(|value| value.as_u64())?;
        if unix_now().saturating_sub(approved_at) > self.ttl.as_secs() {
            return None;
        }
        let approver = record.get("approver").and_then(|value| value.as_str())?.trim();
        (!approver.is_empty()).then(|| DdlApproval { approver: approver.to_string() })
    }

    async fn consume(&self, digest: &str) {
        if !is_digest(digest) {
            return;
        }
        let used = self.dir.join("used");
        let _ = tokio::fs::create_dir_all(&used).await;
        if tokio::fs::rename(self.approved_path(digest), used.join(format!("{digest}.json"))).await.is_err() {
            // Never leave a spendable approval behind.
            let _ = tokio::fs::remove_file(self.approved_path(digest)).await;
        }
    }
}

/// Compute the approval digest an external approver must sign. Target and bytes
/// are length-delimited so no field boundary can be shifted to collide.
pub fn ddl_digest(proposal_id: &str, connection_id: &str, database: &str, sql: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for field in [proposal_id, connection_id, database, sql] {
        hasher.update((field.len() as u64).to_le_bytes());
        hasher.update(field.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Validate that `sql` is exactly one DDL statement the engine can execute, and
/// return it trimmed of surrounding whitespace. Rejects empty input, multiple
/// statements, and anything that is not DDL.
pub fn single_statement_ddl(sql: &str, db_type: DatabaseType) -> Result<String, String> {
    let trimmed = sql.trim();
    if trimmed.is_empty() {
        return Err("No SQL provided.".to_string());
    }
    let plan = dbx_core::sql::sql_execution_plan_for_database(trimmed, db_type);
    if plan.statements.len() != 1 {
        return Err(format!(
            "DDL proposals must contain exactly one statement; got {}. Split the work into separate proposals.",
            plan.statements.len()
        ));
    }
    match classify_sql_risk_for_database(trimmed, db_type) {
        Ok(SqlRisk::Ddl) => Ok(trimmed.to_string()),
        Ok(risk) => Err(format!("Only DDL statements are accepted here; this is {risk}.")),
        Err(error) => Err(format!("Could not parse the statement: {error}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_is_stable_and_bound_to_every_field() {
        let base = ddl_digest("p1", "conn", "app", "CREATE TABLE t (id INT)");
        assert_eq!(base, ddl_digest("p1", "conn", "app", "CREATE TABLE t (id INT)"));
        assert_ne!(base, ddl_digest("p2", "conn", "app", "CREATE TABLE t (id INT)"));
        assert_ne!(base, ddl_digest("p1", "other", "app", "CREATE TABLE t (id INT)"));
        assert_ne!(base, ddl_digest("p1", "conn", "other", "CREATE TABLE t (id INT)"));
        assert_ne!(base, ddl_digest("p1", "conn", "app", "CREATE TABLE t (id BIGINT)"));
    }

    #[test]
    fn only_a_single_ddl_statement_is_accepted() {
        assert!(single_statement_ddl("CREATE TABLE t (id INT)", DatabaseType::Sqlite).is_ok());
        assert!(single_statement_ddl("  DROP TABLE t  ", DatabaseType::Sqlite).is_ok());

        assert!(single_statement_ddl("", DatabaseType::Sqlite).is_err());
        assert!(single_statement_ddl("SELECT 1", DatabaseType::Sqlite).is_err());
        assert!(single_statement_ddl("DELETE FROM t", DatabaseType::Sqlite).is_err());
        assert!(single_statement_ddl("CREATE TABLE a (id INT); DROP TABLE b", DatabaseType::Sqlite).is_err());
    }

    #[tokio::test]
    async fn proposals_are_single_use_and_expire() {
        let store = PendingDdlProposals::new(DDL_PROPOSAL_TTL);
        let proposal = || DdlProposal {
            connection_id: "c".into(),
            database: "d".into(),
            sql: "CREATE TABLE t (id INT)".into(),
            digest: "digest".into(),
            created_at: Instant::now(),
        };
        store.insert("p1".into(), proposal()).await;
        assert!(store.take("p1").await.is_some());
        assert!(store.take("p1").await.is_none(), "a proposal is single-use");

        let expired = PendingDdlProposals::new(Duration::ZERO);
        expired.insert("p2".into(), proposal()).await;
        assert!(expired.take("p2").await.is_none(), "an expired proposal is not claimable");
    }

    #[tokio::test]
    async fn proposals_stay_bounded() {
        let store = PendingDdlProposals::new(DDL_PROPOSAL_TTL);
        for index in 0..(DDL_PROPOSAL_LIMIT + 10) {
            store
                .insert(
                    format!("p{index}"),
                    DdlProposal {
                        connection_id: "c".into(),
                        database: "d".into(),
                        sql: "CREATE TABLE t (id INT)".into(),
                        digest: "digest".into(),
                        created_at: Instant::now(),
                    },
                )
                .await;
        }
        assert_eq!(store.entries.lock().await.len(), DDL_PROPOSAL_LIMIT);
    }

    #[tokio::test]
    async fn deny_all_approver_never_approves() {
        let approver = DenyAllApprover;
        assert!(approver.approval_for("any-digest").await.is_none());
    }
}

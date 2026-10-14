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
    sync::Arc,
    time::{Duration, Instant},
};

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

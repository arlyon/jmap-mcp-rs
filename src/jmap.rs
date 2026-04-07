use anyhow::{Context, Result};
use directories::ProjectDirs;
use jmap_client::client::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedFilter {
    pub from: Option<Vec<String>>,
    pub to: Option<Vec<String>>,
    pub subject: Option<String>,
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NamedFiltersStore {
    pub filters: HashMap<String, NamedFilter>,
}

impl NamedFiltersStore {
    pub fn load() -> Result<Self> {
        let path = Self::filters_path()?;

        if !path.exists() {
            return Ok(Self::default());
        }

        let content = fs::read_to_string(&path)
            .context("Failed to read filters file")?;

        serde_json::from_str(&content)
            .context("Failed to parse filters JSON")
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::filters_path()?;

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .context("Failed to create config directory")?;
        }

        let content = serde_json::to_string_pretty(self)
            .context("Failed to serialize filters")?;

        fs::write(&path, content)
            .context("Failed to write filters file")
    }

    fn filters_path() -> Result<PathBuf> {
        // Allow env var override for config path
        if let Ok(path) = env::var("JMAP_CONFIG_DIR") {
            return Ok(PathBuf::from(path).join("filters.json"));
        }

        let project_dirs = ProjectDirs::from("", "", "jmap-mcp")
            .context("Could not determine config directory")?;

        Ok(project_dirs.config_dir().join("filters.json"))
    }
}

pub struct JmapConfig {
    pub session_url: String,
    pub access_token: String,
    pub account_id: Option<String>,
    pub http_timeout_secs: u64,
}

impl JmapConfig {
    pub fn from_env() -> Result<Self> {
        let session_url = env::var("JMAP_SESSION_URL")
            .context("JMAP_SESSION_URL must be set")?;
        let access_token = env::var("JMAP_BEARER_TOKEN")
            .context("JMAP_BEARER_TOKEN must be set")?;
        let account_id = env::var("JMAP_ACCOUNT_ID").ok();

        let http_timeout_secs = env::var("JMAP_HTTP_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60);

        Ok(Self {
            session_url,
            access_token,
            account_id,
            http_timeout_secs,
        })
    }
}

pub async fn init_client(config: &JmapConfig) -> Result<Client> {
    // Using explicit connect pattern with timeout configuration
    let mut client = Client::new()
        .credentials(config.access_token.as_str())
        .timeout(std::time::Duration::from_secs(config.http_timeout_secs))
        .connect(&config.session_url)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to connect: {}", e))?;

    // Set default account ID if provided
    if let Some(account_id) = &config.account_id {
        client.set_default_account_id(account_id);
    }

    Ok(client)
}

use anyhow::{Context, Result};
use jmap_client::client::Client;
use std::env;

pub struct JmapConfig {
    pub session_url: String,
    pub access_token: String,
    pub account_id: Option<String>,
}

impl JmapConfig {
    pub fn from_env() -> Result<Self> {
        let session_url = env::var("JMAP_SESSION_URL")
            .context("JMAP_SESSION_URL must be set")?;
        let access_token = env::var("JMAP_BEARER_TOKEN")
            .context("JMAP_BEARER_TOKEN must be set")?;
        let account_id = env::var("JMAP_ACCOUNT_ID").ok();

        Ok(Self {
            session_url,
            access_token,
            account_id,
        })
    }
}

pub async fn init_client(config: &JmapConfig) -> Result<Client> {
    // Using explicit connect pattern as suggested
    // jmap-client 0.4.0 might behave slightly differently but let's try this
    let client = Client::new()
        .credentials(config.access_token.as_str())
        .connect(&config.session_url)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to connect: {}", e))?;
        
    Ok(client)
}

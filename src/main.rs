use anyhow::Result;
use clap::Parser;
use rmcp::{transport::stdio, ServiceExt};
use std::sync::{Arc, RwLock};
use tracing_subscriber;

use jmap_mcp_rs::jmap::{init_client, JmapConfig, NamedFiltersStore};
use jmap_mcp_rs::tools::JmapServer;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let _cli = Cli::parse();

    // Load config
    let config = Arc::new(JmapConfig::from_env()?);

    // Init client
    let client = Arc::new(init_client(&config).await?);

    // Load named filters
    let filters = Arc::new(RwLock::new(NamedFiltersStore::load()?));
    tracing::info!("Loaded {} named filters", filters.read().unwrap().filters.len());

    // Create Server
    let server = JmapServer::new(client, config.account_id.clone(), config, filters);

    // Start Server
    tracing::info!("Starting JMAP MCP Server...");
    let service = server.serve(stdio()).await?;

    service.waiting().await?;

    Ok(())
}

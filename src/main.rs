use anyhow::Result;
use clap::Parser;
use rmcp::{transport::stdio, ServiceExt};
use std::sync::Arc;
use tracing_subscriber;

mod jmap;
mod tools;

use jmap::init_client;
use tools::JmapServer;

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
    let config = jmap::JmapConfig::from_env()?;
    
    // Init client
    let client = Arc::new(init_client(&config).await?);
    
    // Create Server
    let server = JmapServer::new(client, config.account_id);
    
    // Start Server
    tracing::info!("Starting JMAP MCP Server...");
    let service = server.serve(stdio()).await?;
    
    service.waiting().await?;
    
    Ok(())
}

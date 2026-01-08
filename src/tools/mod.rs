use jmap_client::client::Client as JmapClient;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::model::{Implementation, ServerCapabilities, ServerInfo};
use rmcp::{tool_handler, ServerHandler};
use std::sync::Arc;

pub mod email;
pub mod submission;

#[derive(Clone)]
pub struct JmapServer {
    pub client: Arc<JmapClient>,
    pub account_id: Option<String>,
    pub tool_router: ToolRouter<JmapServer>,
}

impl JmapServer {
    pub fn new(client: Arc<JmapClient>, account_id: Option<String>) -> Self {
        Self {
            client,
            account_id,
            tool_router: {
                let mut r = Self::public_email_router();
                r.merge(Self::public_submission_router());
                r
            },
        }
    }
}

// Implement ServerHandler to expose capabilities
#[tool_handler]
impl ServerHandler for JmapServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            server_info: Implementation {
                name: "jmap-mcp-rs".into(),
                version: "0.1.0".into(),
                ..Default::default()
            },
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            instructions: Some("JMAP Email Management Server".into()),
            protocol_version: rmcp::model::ProtocolVersion::V_2025_06_18,
        }
    }
}

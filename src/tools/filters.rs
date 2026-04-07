use super::JmapServer;
use crate::jmap::NamedFilter;
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::CallToolResult;
use rmcp::schemars::JsonSchema;
use rmcp::{tool, tool_router, ErrorData};
use serde::Deserialize;
use serde_json::json;

// --- Arguments ---

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddFilterArgs {
    pub name: String,
    pub from: Option<Vec<String>>,
    pub to: Option<Vec<String>>,
    pub subject: Option<String>,
    pub text: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RemoveFilterArgs {
    pub name: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListFiltersArgs {}

// --- Implementation ---

impl JmapServer {
    pub fn public_filters_router() -> ToolRouter<Self> {
        Self::filters_router()
    }
}

#[tool_router(router = filters_router)]
impl JmapServer {
    #[tool(description = "Add or update a named filter for easy email searching. Example: add_filter(name=\"vercel_alerts\", from=[\"noreply@vercel.com\"], subject=\"deployment\")")]
    async fn add_filter(
        &self,
        Parameters(args): Parameters<AddFilterArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let filter = NamedFilter {
            from: args.from,
            to: args.to,
            subject: args.subject,
            text: args.text,
        };

        {
            let mut store = self.filters.write()
                .map_err(|e| ErrorData::internal_error(format!("Lock error: {}", e), None))?;

            store.filters.insert(args.name.clone(), filter);

            store.save()
                .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        }

        Ok(CallToolResult::success(vec![rmcp::model::Content::text(
            format!("Filter '{}' saved successfully", args.name),
        )]))
    }

    #[tool(description = "Remove a named filter by name.")]
    async fn remove_filter(
        &self,
        Parameters(args): Parameters<RemoveFilterArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        {
            let mut store = self.filters.write()
                .map_err(|e| ErrorData::internal_error(format!("Lock error: {}", e), None))?;

            if store.filters.remove(&args.name).is_none() {
                return Err(ErrorData::internal_error(
                    format!("Filter '{}' not found", args.name),
                    None,
                ));
            }

            store.save()
                .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        }

        Ok(CallToolResult::success(vec![rmcp::model::Content::text(
            format!("Filter '{}' removed successfully", args.name),
        )]))
    }

    #[tool(description = "List all saved named filters.")]
    async fn list_filters(
        &self,
        Parameters(_args): Parameters<ListFiltersArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let store = self.filters.read()
            .map_err(|e| ErrorData::internal_error(format!("Lock error: {}", e), None))?;

        let filters_data: Vec<_> = store.filters.iter()
            .map(|(name, filter)| {
                json!({
                    "name": name,
                    "filter": filter
                })
            })
            .collect();

        Ok(CallToolResult::success(vec![rmcp::model::Content::json(
            json!({ "filters": filters_data }),
        )
        .unwrap()]))
    }
}

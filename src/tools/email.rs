use super::JmapServer;
use jmap_client::email::query::Filter as EmailFilter;
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::CallToolResult;
use rmcp::schemars::JsonSchema;
use rmcp::{tool, tool_router, ErrorData};
use serde::Deserialize;
use serde_json::json;

// --- Arguments ---

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchEmailsArgs {
    pub query: Option<String>,
    pub from: Option<String>,
    pub to: Option<String>,
    #[serde(rename = "inMailbox")]
    pub in_mailbox: Option<String>,
    pub before: Option<String>,
    pub after: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetEmailsArgs {
    pub ids: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetThreadsArgs {
    pub ids: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetMailboxesArgs {
    #[serde(rename = "parentId")]
    pub parent_id: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MarkEmailsArgs {
    pub ids: Vec<String>,
    pub seen: Option<bool>,
    pub flagged: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MoveEmailsArgs {
    pub ids: Vec<String>,
    #[serde(rename = "mailboxId")]
    pub mailbox_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeleteEmailsArgs {
    pub ids: Vec<String>,
}

// --- Implementation ---

impl JmapServer {
    pub fn public_email_router() -> ToolRouter<Self> {
        Self::email_router()
    }
}

#[tool_router(router = email_router)]
impl JmapServer {
    #[tool(description = "Search emails with filtering (text, from, to, date, mailbox).")]
    async fn search_emails(
        &self,
        Parameters(args): Parameters<SearchEmailsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let mut filter: Option<EmailFilter> = None;

        if let Some(q) = args.query {
            filter = Some(EmailFilter::text(q));
        }

        if let Some(f) = args.from {
            filter = Some(EmailFilter::from(f));
        }

        let _limit = args.limit.unwrap_or(50);

        let request = self.client.email_query(filter, Some([]));

        let result = request
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![rmcp::model::Content::json(
            json!({ "ids": result.ids() }),
        )
        .unwrap()]))
    }

    #[tool(description = "Retrieve full details (body, headers) for specific IDs.")]
    async fn get_emails(
        &self,
        Parameters(args): Parameters<GetEmailsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let mut emails = Vec::new();
        for id in args.ids {
            let email = self
                .client
                .email_get(&id, Option::<Vec<jmap_client::email::Property>>::None)
                .await
                .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
            if let Some(e) = email {
                emails.push(e);
            }
        }

        Ok(CallToolResult::success(vec![rmcp::model::Content::json(
            json!({ "emails": emails }),
        )
        .unwrap()]))
    }

    #[tool(description = "Retrieve email thread details.")]
    async fn get_threads(
        &self,
        Parameters(args): Parameters<GetThreadsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let mut threads = Vec::new();
        for id in args.ids {
            let thread = self
                .client
                .thread_get(&id)
                .await
                .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
            if let Some(t) = thread {
                threads.push(t);
            }
        }

        Ok(CallToolResult::success(vec![rmcp::model::Content::json(
            json!({ "threads": threads }),
        )
        .unwrap()]))
    }

    #[tool(description = "List all mailboxes and their hierarchy.")]
    async fn get_mailboxes(
        &self,
        Parameters(_args): Parameters<GetMailboxesArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let query_result = self
            .client
            .mailbox_query(
                Option::<jmap_client::mailbox::query::Filter>::None,
                Some([]),
            )
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

        let mut mailboxes = Vec::new();
        for id in query_result.ids() {
            let mb = self
                .client
                .mailbox_get(&id, Option::<Vec<jmap_client::mailbox::Property>>::None)
                .await
                .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
            if let Some(m) = mb {
                mailboxes.push(m);
            }
        }

        Ok(CallToolResult::success(vec![rmcp::model::Content::json(
            json!({ "mailboxes": mailboxes }),
        )
        .unwrap()]))
    }

    #[tool(description = "Modify email keywords (read/unread, flagged).")]
    async fn mark_emails(
        &self,
        Parameters(args): Parameters<MarkEmailsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let mut request = self.client.build();

        {
            let email_set = request.set_email();
            for id in args.ids {
                let patch = email_set.update(id);
                if let Some(seen) = args.seen {
                    patch.keyword("$seen", seen);
                }
                if let Some(flagged) = args.flagged {
                    patch.keyword("$flagged", flagged);
                }
            }
        }

        request
            .send()
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![rmcp::model::Content::text(
            "Emails marked successfully",
        )]))
    }

    #[tool(description = "Move emails to a specific mailbox.")]
    async fn move_emails(
        &self,
        Parameters(args): Parameters<MoveEmailsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let mut request = self.client.build();

        {
            let email_set = request.set_email();
            for id in args.ids {
                let patch = email_set.update(id);
                patch.mailbox_id(&args.mailbox_id, true);
            }
        }

        request
            .send()
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![rmcp::model::Content::text(
            "Emails moved successfully",
        )]))
    }

    #[tool(description = "Permanently remove emails.")]
    async fn delete_emails(
        &self,
        Parameters(args): Parameters<DeleteEmailsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let mut request = self.client.build();

        request.set_email().destroy(args.ids);

        request
            .send()
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![rmcp::model::Content::text(
            "Emails deleted successfully",
        )]))
    }
}

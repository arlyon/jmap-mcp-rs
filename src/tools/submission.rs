use super::JmapServer;
use jmap_client::core::set::SetObject;
use jmap_client::email::EmailBodyPart;
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::CallToolResult;
use rmcp::schemars::JsonSchema;
use rmcp::{tool, tool_router, ErrorData};
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SendEmailArgs {
    pub from: String,
    pub to: Vec<String>,
    pub cc: Option<Vec<String>>,
    pub subject: Option<String>,
    #[serde(rename = "textBody")]
    pub text_body: Option<String>,
    #[serde(rename = "htmlBody")]
    pub html_body: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReplyToEmailArgs {
    #[serde(rename = "emailId")]
    pub email_id: String,
    #[serde(rename = "replyAll")]
    pub reply_all: Option<bool>,
    #[serde(rename = "textBody")]
    pub text_body: String,
}

impl JmapServer {
    pub fn public_submission_router() -> ToolRouter<Self> {
        Self::submission_router()
    }

    /// Get the identity ID for a given email address by making a raw JMAP request
    /// The jmap-client library's identity methods don't work, so we use raw HTTP
    async fn get_identity_id_for_email(&self, email_address: &str) -> anyhow::Result<String> {
        use serde_json::json;

        let account_id = self
            .account_id
            .as_deref()
            .or_else(|| Some(self.client.default_account_id()))
            .ok_or_else(|| anyhow::anyhow!("No account ID available"))?;

        tracing::debug!(
            "Getting identity for email: {}, account: {}",
            email_address,
            account_id
        );

        // Make raw JMAP request to get identities
        let request_body = json!({
            "using": ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:submission"],
            "methodCalls": [[
                "Identity/get",
                {
                    "accountId": account_id
                },
                "0"
            ]]
        });

        let session = self.client.session();
        let api_url = session.api_url();

        tracing::debug!("Making request to: {}", api_url);

        let http_client = reqwest::Client::new();
        let auth_header = format!("Bearer {}", self.config.access_token);

        let response = http_client
            .post(api_url)
            .header("Authorization", &auth_header)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("HTTP request failed: {}", e))?;

        let status = response.status();
        let response_text = response
            .text()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read response: {}", e))?;

        tracing::debug!("Response status: {}, body: {}", status, response_text);

        let response_json: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|e| anyhow::anyhow!("Failed to parse JSON: {}", e))?;

        // Parse the response to find matching identity
        if let Some(method_responses) = response_json.get("methodResponses") {
            if let Some(first_response) = method_responses.get(0) {
                if let Some(list) = first_response.get(1).and_then(|r| r.get("list")) {
                    if let Some(identities) = list.as_array() {
                        tracing::debug!("Found {} identities", identities.len());

                        // Find identity matching the email address
                        for identity in identities {
                            if let Some(email) = identity.get("email").and_then(|e| e.as_str()) {
                                if let Some(id) = identity.get("id").and_then(|i| i.as_str()) {
                                    tracing::debug!("Found identity: {} -> {}", email, id);
                                    if email == email_address {
                                        tracing::info!("Using identity {} for {}", id, email);
                                        return Ok(id.to_string());
                                    }
                                }
                            }
                        }

                        // If no exact match, return first identity
                        if let Some(first_identity) = identities.first() {
                            if let Some(id) = first_identity.get("id").and_then(|i| i.as_str()) {
                                if let Some(email) =
                                    first_identity.get("email").and_then(|e| e.as_str())
                                {
                                    tracing::warn!(
                                        "No identity found for {}, using first identity: {} ({})",
                                        email_address,
                                        id,
                                        email
                                    );
                                    return Ok(id.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }

        Err(anyhow::anyhow!(
            "No identities found in response: {}",
            response_text
        ))
    }

    /// Get the Drafts mailbox ID
    /// Emails must be in a mailbox before they can be sent via EmailSubmission
    async fn get_drafts_mailbox_id(&self) -> anyhow::Result<String> {
        use jmap_client::mailbox::query::Filter as MailboxFilter;
        use jmap_client::mailbox::Role;

        let query_result = self
            .client
            .mailbox_query(Some(MailboxFilter::role(Role::Drafts)), Some([]))
            .await?;

        let drafts_id = query_result
            .ids()
            .first()
            .ok_or_else(|| anyhow::anyhow!("No Drafts mailbox found"))?
            .to_string();

        Ok(drafts_id)
    }
}

#[tool_router(router = submission_router)]
impl JmapServer {
    #[tool(description = "Compose and send a new message.")]
    async fn send_email(
        &self,
        Parameters(args): Parameters<SendEmailArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        // Get the proper identity ID for the from address
        let identity_id = self
            .get_identity_id_for_email(&args.from)
            .await
            .map_err(|e| {
                ErrorData::internal_error(format!("Failed to get identity: {}", e), None)
            })?;

        // Get the Drafts mailbox ID - required for sending
        let drafts_mailbox_id = self.get_drafts_mailbox_id().await.map_err(|e| {
            ErrorData::internal_error(format!("Failed to get Drafts mailbox: {}", e), None)
        })?;

        let mut request = self.client.build();

        let draft = request.set_email().create();
        let draft_ref = draft.create_id().unwrap();

        draft.from(vec![args.from.as_str()]);
        draft.to(args.to.clone());
        if let Some(cc) = args.cc {
            draft.cc(cc);
        }
        if let Some(subject) = args.subject {
            draft.subject(subject);
        }

        // CRITICAL: Email must be in a mailbox to be sent
        draft.mailbox_ids([&drafts_mailbox_id]);

        // Mark as draft
        draft.keywords(["$draft"]);

        if let Some(body) = args.text_body {
            let part_id = "body";
            draft.body_value(part_id.to_string(), body);

            let part = EmailBodyPart::new()
                .part_id(part_id)
                .content_type("text/plain");
            draft.text_body(part);
        } else if let Some(html) = args.html_body {
            let part_id = "body";
            draft.body_value(part_id.to_string(), html);

            let part = EmailBodyPart::new()
                .part_id(part_id)
                .content_type("text/html");
            draft.html_body(part);
        }

        let submission = request.set_email_submission().create();
        submission.email_id(format!("#{}", &draft_ref));
        submission.identity_id(identity_id);

        // Send the request
        request
            .send()
            .await
            .map_err(|e| ErrorData::internal_error(format!("Failed to send email: {}", e), None))?;

        Ok(CallToolResult::success(vec![rmcp::model::Content::text(
            "Email sent successfully",
        )]))
    }

    #[tool(description = "Reply to an existing message.")]
    async fn reply_to_email(
        &self,
        Parameters(args): Parameters<ReplyToEmailArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let original_email = self
            .client
            .email_get(
                &args.email_id,
                Option::<Vec<jmap_client::email::Property>>::None,
            )
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?
            .ok_or_else(|| ErrorData::internal_error("Original email not found", None))?;

        // Determine reply recipients
        // For reply: from -> original from
        // For reply-all: from -> original from, cc -> original to + cc (excluding self)
        let reply_to = original_email
            .reply_to()
            .or_else(|| original_email.from())
            .ok_or_else(|| {
                ErrorData::internal_error("Original email has no from/reply-to", None)
            })?;

        let reply_to_email = reply_to
            .first()
            .ok_or_else(|| ErrorData::internal_error("No reply-to address found", None))?
            .email()
            .to_string();

        // Get the recipient address from the original email to determine which identity to use
        let our_address = original_email
            .to()
            .and_then(|addrs| addrs.first())
            .map(|addr| addr.email().to_string())
            .ok_or_else(|| {
                ErrorData::internal_error("Could not determine recipient address", None)
            })?;

        // Get the proper identity ID
        let identity_id = self
            .get_identity_id_for_email(&our_address)
            .await
            .map_err(|e| {
                ErrorData::internal_error(format!("Failed to get identity: {}", e), None)
            })?;

        // Get the Drafts mailbox ID - required for sending
        let drafts_mailbox_id = self.get_drafts_mailbox_id().await.map_err(|e| {
            ErrorData::internal_error(format!("Failed to get Drafts mailbox: {}", e), None)
        })?;

        let mut request = self.client.build();
        let draft = request.set_email().create();
        let draft_ref = draft.create_id().unwrap();

        // Set reply headers
        let subject = original_email.subject().unwrap_or("");
        let reply_subject = if subject.to_lowercase().starts_with("re:") {
            subject.to_string()
        } else {
            format!("Re: {}", subject)
        };
        draft.subject(reply_subject);

        // Set from/to
        draft.from(vec![our_address.as_str()]);
        draft.to(vec![reply_to_email.as_str()]);

        // CRITICAL: Email must be in a mailbox to be sent
        draft.mailbox_ids([&drafts_mailbox_id]);

        // Mark as draft
        draft.keywords(["$draft"]);

        // Handle reply-all
        if args.reply_all.unwrap_or(false) {
            let mut cc_addrs = Vec::new();

            // Add original To addresses (excluding ourselves)
            if let Some(to_addrs) = original_email.to() {
                for addr in to_addrs {
                    if addr.email() != our_address {
                        cc_addrs.push(addr.email().to_string());
                    }
                }
            }

            // Add original CC addresses (excluding ourselves)
            if let Some(cc) = original_email.cc() {
                for addr in cc {
                    if addr.email() != our_address {
                        cc_addrs.push(addr.email().to_string());
                    }
                }
            }

            if !cc_addrs.is_empty() {
                draft.cc(cc_addrs);
            }
        }

        // Set In-Reply-To and References headers
        if let Some(msg_id) = original_email.message_id() {
            draft.in_reply_to(
                msg_id
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<String>>(),
            );

            // References should include both the original references and the message ID
            let mut references: Vec<String> = original_email
                .references()
                .map(|refs| refs.iter().map(|s| s.to_string()).collect())
                .unwrap_or_default();
            references.extend(msg_id.iter().map(|s| s.to_string()));
            draft.references(references);
        }

        let part_id = "body";
        draft.body_value(part_id.to_string(), args.text_body);
        let part = EmailBodyPart::new()
            .part_id(part_id)
            .content_type("text/plain");
        draft.text_body(part);

        let submission = request.set_email_submission().create();
        submission.email_id(format!("#{}", &draft_ref));
        submission.identity_id(identity_id);

        // Send the request
        request
            .send()
            .await
            .map_err(|e| ErrorData::internal_error(format!("Failed to send reply: {}", e), None))?;

        Ok(CallToolResult::success(vec![rmcp::model::Content::text(
            "Reply sent successfully",
        )]))
    }
}

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
}

#[tool_router(router = submission_router)]
impl JmapServer {
    #[tool(description = "Compose and send a new message.")]
    async fn send_email(
        &self,
        Parameters(args): Parameters<SendEmailArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let identity_id = args.from.clone();

        let mut request = self.client.build();

        let draft = request.set_email().create();
        let draft_id = draft.create_id().unwrap();

        draft.from(vec![args.from.as_str()]);
        draft.to(args.to.clone());
        if let Some(cc) = args.cc {
            draft.cc(cc);
        }
        if let Some(subject) = args.subject {
            draft.subject(subject);
        }

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
        submission.email_id(format!("#{}", draft_id));
        submission.identity_id(identity_id);

        request
            .send()
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

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

        let identity_id = if let Some(to) = original_email.to() {
            to.iter()
                .next()
                .map(|addr| addr.email().to_string())
                .unwrap_or_else(|| "default".to_string())
        } else {
            "default".to_string()
        };

        let mut request = self.client.build();
        let draft = request.set_email().create();
        let draft_id = draft.create_id().unwrap();

        let subject = format!("Re: {}", original_email.subject().unwrap_or(""));
        draft.subject(subject);

        if let Some(msg_id) = original_email.message_id() {
            draft.in_reply_to(
                msg_id
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<String>>(),
            );
            draft.references(
                msg_id
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<String>>(),
            );
        }

        let part_id = "body";
        draft.body_value(part_id.to_string(), args.text_body);
        let part = EmailBodyPart::new()
            .part_id(part_id)
            .content_type("text/plain");
        draft.text_body(part);

        let submission = request.set_email_submission().create();
        submission.email_id(format!("#{}", draft_id));
        submission.identity_id(identity_id);

        request
            .send()
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![rmcp::model::Content::text(
            "Reply sent successfully",
        )]))
    }
}

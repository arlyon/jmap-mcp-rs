use jmap_mcp_rs::jmap::{init_client, JmapConfig};
use jmap_mcp_rs::tools::email::SearchEmailsArgs;
use jmap_mcp_rs::tools::JmapServer;
use std::sync::Arc;

async fn setup_client() -> Option<Arc<jmap_client::client::Client>> {
    dotenvy::dotenv().ok();

    let config = match JmapConfig::from_env() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("Skipping tests: JMAP credentials not configured");
            return None;
        }
    };

    match init_client(&config).await {
        Ok(client) => Some(Arc::new(client)),
        Err(e) => {
            eprintln!("Failed to connect to JMAP server: {}", e);
            None
        }
    }
}

#[tokio::test]
async fn test_search_emails_respects_limit() {
    let Some(client) = setup_client().await else {
        return;
    };

    // Test with limit of 5
    let args = SearchEmailsArgs {
        query: None,
        from: None,
        to: None,
        in_mailbox: None,
        before: None,
        after: None,
        limit: Some(5),
    };

    let result = JmapServer::search_emails_impl(&client, args).await;
    assert!(result.is_ok(), "Search should succeed");

    let ids = result.unwrap();
    assert!(
        ids.len() <= 5,
        "Should return at most 5 results, got {}",
        ids.len()
    );
}

#[tokio::test]
async fn test_search_emails_default_limit() {
    let Some(client) = setup_client().await else {
        return;
    };

    // Test with no limit (should default to 50)
    let args = SearchEmailsArgs {
        query: None,
        from: None,
        to: None,
        in_mailbox: None,
        before: None,
        after: None,
        limit: None,
    };

    let result = JmapServer::search_emails_impl(&client, args).await;
    assert!(result.is_ok(), "Search should succeed");

    let ids = result.unwrap();
    assert!(
        ids.len() <= 50,
        "Should return at most 50 results (default), got {}",
        ids.len()
    );
}

#[tokio::test]
async fn test_search_emails_with_limit_1() {
    let Some(client) = setup_client().await else {
        return;
    };

    // Test with limit of 1 to verify the limit is actually being applied
    let args = SearchEmailsArgs {
        query: None,
        from: None,
        to: None,
        in_mailbox: None,
        before: None,
        after: None,
        limit: Some(1),
    };

    let result = JmapServer::search_emails_impl(&client, args).await;
    assert!(result.is_ok(), "Search should succeed");

    let ids = result.unwrap();
    assert!(
        ids.len() <= 1,
        "Should return at most 1 result, got {}",
        ids.len()
    );
}

#[tokio::test]
async fn test_search_emails_with_query_filter() {
    let Some(client) = setup_client().await else {
        return;
    };

    // Test with a text query filter
    let args = SearchEmailsArgs {
        query: Some("test".to_string()),
        from: None,
        to: None,
        in_mailbox: None,
        before: None,
        after: None,
        limit: Some(10),
    };

    let result = JmapServer::search_emails_impl(&client, args).await;
    assert!(result.is_ok(), "Search with query filter should succeed");

    let ids = result.unwrap();
    assert!(
        ids.len() <= 10,
        "Should respect limit of 10, got {}",
        ids.len()
    );
}

#[tokio::test]
async fn test_get_mailboxes() {
    let Some(client) = setup_client().await else {
        return;
    };

    let result = JmapServer::get_mailboxes_impl(&client).await;
    assert!(result.is_ok(), "Getting mailboxes should succeed");

    let mailboxes = result.unwrap();
    assert!(!mailboxes.is_empty(), "Should have at least one mailbox");
}

#[tokio::test]
async fn test_get_emails_empty_list() {
    let Some(client) = setup_client().await else {
        return;
    };

    // Test with empty ID list
    let result = JmapServer::get_emails_impl(&client, vec![]).await;
    assert!(
        result.is_ok(),
        "Getting emails with empty list should succeed"
    );

    let emails = result.unwrap();
    assert!(
        emails.is_empty(),
        "Should return empty list for empty input"
    );
}

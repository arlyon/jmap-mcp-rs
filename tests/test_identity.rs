use jmap_client::client::Client;
use jmap_client::core::get::GetResponse;
use jmap_client::identity::Identity;

#[tokio::test]
async fn test_get_identities() {
    // Load .env.test file
    dotenvy::from_filename(".env.test").ok();

    // Load credentials from environment
    let session_url = std::env::var("JMAP_SESSION_URL").expect("JMAP_SESSION_URL must be set");
    let bearer_token = std::env::var("JMAP_BEARER_TOKEN").expect("JMAP_BEARER_TOKEN must be set");
    let account_id = std::env::var("JMAP_ACCOUNT_ID").ok();

    println!("Session URL: {}", session_url);
    println!("Account ID: {:?}", account_id);

    // Connect to JMAP server
    let mut client = Client::new()
        .credentials(bearer_token.as_str())
        .timeout(std::time::Duration::from_secs(60))
        .connect(&session_url)
        .await
        .expect("Failed to connect to JMAP server");

    println!("✓ Connected successfully");

    // Set default account ID if provided
    if let Some(account_id) = &account_id {
        client.set_default_account_id(account_id);
        println!("✓ Set default account ID: {}", account_id);
    } else {
        // Try to get default account from session
        let default_account = client.default_account_id();
        println!("✓ Default account from session: {}", default_account);
    }

    // Print session capabilities
    let session = client.session();
    println!("\nSession capabilities:");
    let caps: Vec<_> = session.capabilities().collect();
    println!("{:#?}", caps);

    // Try to get identities using the direct client method
    println!("\n--- Attempting to get identities (using client.identity_get) ---");
    match client
        .identity_get("", Option::<Vec<jmap_client::identity::Property>>::None)
        .await
    {
        Ok(Some(identity)) => {
            println!("✓ Successfully retrieved identity");
            println!("  ID: {:?}", identity.id());
            println!("  Name: {:?}", identity.name());
            println!("  Email: {:?}", identity.email());
        }
        Ok(None) => {
            eprintln!("✗ No identity found");
            panic!("No identity returned");
        }
        Err(e) => {
            eprintln!("✗ Failed to get identity: {}", e);
            eprintln!("Error details: {:?}", e);

            // Try the request builder approach as fallback
            println!("\n--- Trying request builder approach ---");
            let mut request = client.build();
            request.get_identity();

            match request.send_single::<GetResponse<Identity>>().await {
                Ok(mut response) => {
                    let identities = response.take_list();
                    println!(
                        "✓ Successfully retrieved {} identities via builder",
                        identities.len()
                    );

                    for (i, identity) in identities.iter().enumerate() {
                        println!("\nIdentity {}:", i + 1);
                        println!("  ID: {:?}", identity.id());
                        println!("  Name: {:?}", identity.name());
                        println!("  Email: {:?}", identity.email());
                    }
                }
                Err(e2) => {
                    eprintln!("✗ Request builder also failed: {}", e2);
                    panic!("Both approaches failed");
                }
            }
        }
    }
}

#[tokio::test]
async fn test_send_email_with_email_as_identity() {
    use jmap_client::core::set::SetObject;
    use jmap_client::email::EmailBodyPart;

    // Load .env.test file
    dotenvy::from_filename(".env.test").ok();

    // Load credentials from environment
    let session_url = std::env::var("JMAP_SESSION_URL").expect("JMAP_SESSION_URL must be set");
    let bearer_token = std::env::var("JMAP_BEARER_TOKEN").expect("JMAP_BEARER_TOKEN must be set");
    let account_id = std::env::var("JMAP_ACCOUNT_ID").ok();

    // Connect and configure client
    let mut client = Client::new()
        .credentials(bearer_token.as_str())
        .timeout(std::time::Duration::from_secs(60))
        .connect(&session_url)
        .await
        .expect("Failed to connect");

    if let Some(account_id) = &account_id {
        client.set_default_account_id(account_id);
    }

    println!("Connected to JMAP server");

    // Use actual Fastmail identity ID (from Identity/get response)
    let from_email = "arlyon@fastmail.com";
    let identity_id = "94225816"; // Actual identity ID from Fastmail

    println!("\n--- Using actual identity ID ---");
    println!("  Email: {}", from_email);
    println!("  Identity ID: {}", identity_id);

    // Get Drafts mailbox
    println!("\n--- Getting Drafts mailbox ---");
    use jmap_client::mailbox::query::Filter as MailboxFilter;
    use jmap_client::mailbox::Role;

    let query_result = client
        .mailbox_query(Some(MailboxFilter::role(Role::Drafts)), Some([]))
        .await
        .expect("Failed to query mailboxes");

    let drafts_id = query_result
        .ids()
        .first()
        .expect("No Drafts mailbox found")
        .to_string();

    println!("  Drafts mailbox ID: {}", drafts_id);

    // Try to send a test email
    println!("\n--- Attempting to send email ---");
    let to_email = "arlyon@me.com"; // Send to self for testing

    let mut request = client.build();

    // Create draft email
    let draft = request.set_email().create();
    let draft_id = draft.create_id().unwrap();

    draft.from(vec![from_email]);
    draft.to(vec![to_email]);
    draft.subject("Test email from JMAP MCP Unit Test");

    // CRITICAL: Email must be in a mailbox to be sent
    draft.mailbox_ids([&drafts_id]);

    // Mark as draft
    draft.keywords(["$draft"]);

    let part_id = "body";
    draft.body_value(
        part_id.to_string(),
        "This is a test email sent via JMAP to verify the send_email tool works correctly.",
    );

    let part = EmailBodyPart::new()
        .part_id(part_id)
        .content_type("text/plain");
    draft.text_body(part);

    // Create submission with email as identity
    let submission = request.set_email_submission().create();
    submission.email_id(format!("#{}", draft_id));
    submission.identity_id(identity_id);

    // Send the request and inspect the response
    match request.send().await {
        Ok(response) => {
            println!("✓ Request completed!");
            println!("\n--- Server Response ---");
            println!("{:#?}", response);
            println!("\n✓ Check {} and {} for the test email", from_email, to_email);
        }
        Err(e) => {
            eprintln!("✗ Failed to send email: {}", e);
            eprintln!("Error details: {:?}", e);
            panic!("Email send failed");
        }
    }
}

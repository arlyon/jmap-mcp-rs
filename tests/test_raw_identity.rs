use serde_json::json;

#[tokio::test]
async fn test_raw_identity_query() {
    dotenvy::from_filename(".env.test").ok();

    let session_url = std::env::var("JMAP_SESSION_URL").expect("JMAP_SESSION_URL required");
    let bearer_token = std::env::var("JMAP_BEARER_TOKEN").expect("JMAP_BEARER_TOKEN required");

    // Make a raw HTTP request to get identities
    let api_url = "https://api.fastmail.com/jmap/api/";

    let request_body = json!({
        "using": [
            "urn:ietf:params:jmap:core",
            "urn:ietf:params:jmap:submission"
        ],
        "methodCalls": [[
            "Identity/get",
            {
                "accountId": "uecf140ef"
            },
            "0"
        ]]
    });

    let client = reqwest::Client::new();
    let response = client
        .post(api_url)
        .header("Authorization", format!("Bearer {}", bearer_token))
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()
        .await
        .expect("Failed to send request");

    let response_text = response.text().await.expect("Failed to read response");
    println!("\n--- Raw Identity/get Response ---");
    println!("{}", response_text);

    let response_json: serde_json::Value = serde_json::from_str(&response_text)
        .expect("Failed to parse JSON");
    println!("\n--- Formatted Response ---");
    println!("{:#}", serde_json::to_string_pretty(&response_json).unwrap());
}

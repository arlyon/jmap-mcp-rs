use super::JmapServer;
use jmap_client::core::query::{Filter, QueryResponse};
use jmap_client::email::query::Filter as EmailFilter;
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::CallToolResult;
use rmcp::schemars::JsonSchema;
use rmcp::{tool, tool_router, ErrorData};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

#[cfg(not(feature = "qwen-embeddings"))]
use std::collections::HashSet;

#[cfg(feature = "qwen-embeddings")]
use candle_core::quantized::gguf_file;
#[cfg(feature = "qwen-embeddings")]
use candle_core::{Device, Tensor};
#[cfg(feature = "qwen-embeddings")]
use candle_transformers::models::quantized_qwen3_5::ModelWeights as Qwen3_5Model;
#[cfg(feature = "qwen-embeddings")]
use std::sync::{Arc, Mutex, OnceLock};
#[cfg(feature = "qwen-embeddings")]
use tokenizers::Tokenizer;

// --- Arguments ---

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct EmailPreview {
    pub id: String,
    pub from: Option<String>,
    pub subject: Option<String>,
    pub preview: Option<String>,
    #[serde(rename = "receivedAt")]
    pub received_at: Option<String>,
    pub seen: bool,
    pub flagged: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchEmailsArgs {
    pub query: Option<String>,
    pub from: Option<String>,
    pub to: Option<String>,
    #[serde(rename = "inMailbox")]
    pub in_mailbox: Option<String>,
    /// Filter emails before a specific date in ISO-8601 format.
    pub before: Option<String>,
    /// Filter emails after a specific date in ISO-8601 format.
    pub after: Option<String>,
    pub limit: Option<usize>,
    /// Use a named filter defined in JMAP_NAMED_FILTERS
    pub filter: Option<String>,
    /// Filter by seen status
    pub seen: Option<bool>,
    /// Filter by flagged status
    pub flagged: Option<bool>,
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

#[derive(Debug, Deserialize, JsonSchema)]
pub struct BulkActionByQueryArgs {
    /// Search query (same filters as search_emails)
    pub query: Option<String>,
    pub from: Option<String>,
    pub to: Option<String>,
    #[serde(rename = "inMailbox")]
    pub in_mailbox: Option<String>,
    pub before: Option<String>,
    pub after: Option<String>,
    pub filter: Option<String>,
    pub seen: Option<bool>,
    pub flagged: Option<bool>,
    /// Action to perform: "delete", "move", "mark_seen", "mark_unseen"
    pub action: String,
    /// Target mailbox ID (required for "move" action)
    #[serde(rename = "targetMailboxId")]
    pub target_mailbox_id: Option<String>,
    /// Maximum number of emails to process (default: 100)
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetInboxSummaryArgs {
    /// Hours duration for relative time ranges. Behavior depends on what else is specified:
    /// - With 'before': calculates 'after' as (before - hours)
    /// - With 'after': calculates 'before' as (after + hours)
    /// - Alone: shows emails from (now - hours) to now
    /// - Default: 24 if nothing specified
    pub hours: Option<u32>,
    /// Mailbox to summarize (optional, defaults to all)
    #[serde(rename = "inMailbox")]
    pub in_mailbox: Option<String>,
    /// Filter emails before a specific date in ISO-8601 format.
    /// - With 'after': defines explicit range
    /// - With 'hours': end point, calculates start as (before - hours)
    /// - Alone: all emails before this time
    pub before: Option<String>,
    /// Filter emails after a specific date in ISO-8601 format.
    /// - With 'before': defines explicit range
    /// - With 'hours': start point, calculates end as (after + hours)
    /// - Alone: all emails after this time (capped at now)
    pub after: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SenderSummary {
    pub sender: String,
    pub email_ids: Vec<String>,
    pub unread_count: usize,
    pub subjects: Vec<String>,
}

// --- Implementation ---

// Helper functions for subject diversity

// TF-IDF implementation (default, fast)
#[cfg(not(feature = "qwen-embeddings"))]
mod diversity {
    use super::*;

    fn calculate_tfidf_vectors(subjects: &[String]) -> Vec<HashMap<String, f64>> {
        if subjects.is_empty() {
            return vec![];
        }

        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let num_docs = subjects.len() as f64;

        for subject in subjects {
            let lowercase = subject.to_lowercase();
            let words: HashSet<_> = lowercase.split_whitespace().collect();
            for word in words {
                *doc_freq.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        subjects
            .iter()
            .map(|subject| {
                let lowercase = subject.to_lowercase();
                let words: Vec<_> = lowercase.split_whitespace().collect();
                let word_count = words.len() as f64;

                if word_count == 0.0 {
                    return HashMap::new();
                }

                let mut term_freq: HashMap<String, f64> = HashMap::new();
                for word in &words {
                    *term_freq.entry(word.to_string()).or_insert(0.0) += 1.0;
                }

                let mut tfidf: HashMap<String, f64> = HashMap::new();
                for (word, tf) in term_freq {
                    let df = *doc_freq.get(&word).unwrap_or(&1) as f64;
                    let idf = (num_docs / df).ln();
                    tfidf.insert(word, (tf / word_count) * idf);
                }

                tfidf
            })
            .collect()
    }

    pub fn select_diverse_subjects(subjects: Vec<String>, limit: usize) -> Vec<String> {
        if subjects.len() <= limit {
            return subjects;
        }

        let tfidf_vectors = calculate_tfidf_vectors(&subjects);

        // Convert sparse TF-IDF vectors to dense normalized vectors for K-Means
        // First, collect all unique words across all documents
        let mut all_words: HashSet<String> = HashSet::new();
        for vec in &tfidf_vectors {
            all_words.extend(vec.keys().cloned());
        }
        let word_list: Vec<String> = all_words.into_iter().collect();

        // Convert to dense vectors
        let mut dense_vectors: Vec<Vec<f32>> = tfidf_vectors
            .iter()
            .map(|sparse_vec| {
                word_list
                    .iter()
                    .map(|word| *sparse_vec.get(word).unwrap_or(&0.0) as f32)
                    .collect()
            })
            .collect();

        // L2-Normalize all vectors
        for vec in &mut dense_vectors {
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                vec.iter_mut().for_each(|x| *x /= norm);
            }
        }

        // --- SPHERICAL K-MEANS CLUSTERING ---
        let n_dims = dense_vectors[0].len();

        // Initialize K centroids (using first 'limit' subjects for determinism)
        let mut centroids: Vec<Vec<f32>> = dense_vectors.iter().take(limit).cloned().collect();
        let mut assignments = vec![0; subjects.len()];

        // Lloyd's Algorithm (10 iterations)
        for _ in 0..10 {
            // Step A: Assign each point to nearest centroid
            for (i, vec) in dense_vectors.iter().enumerate() {
                let mut best_cluster = 0;
                let mut max_sim = f32::MIN;

                for (c_idx, centroid) in centroids.iter().enumerate() {
                    let sim: f32 = vec.iter().zip(centroid.iter()).map(|(a, b)| a * b).sum();
                    if sim > max_sim {
                        max_sim = sim;
                        best_cluster = c_idx;
                    }
                }
                assignments[i] = best_cluster;
            }

            // Step B: Recompute centroids
            let mut new_centroids = vec![vec![0.0; n_dims]; limit];
            let mut counts = vec![0; limit];

            for (i, &cluster) in assignments.iter().enumerate() {
                for j in 0..n_dims {
                    new_centroids[cluster][j] += dense_vectors[i][j];
                }
                counts[cluster] += 1;
            }

            // Step C: L2-Normalize centroids
            for (c_idx, centroid) in new_centroids.iter_mut().enumerate() {
                if counts[c_idx] > 0 {
                    let norm: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        centroid.iter_mut().for_each(|x| *x /= norm);
                    }
                    centroids[c_idx] = centroid.clone();
                }
            }
        }

        // Find actual email closest to each centroid
        let mut selected_indices = Vec::new();
        for centroid in &centroids {
            let mut best_idx = 0;
            let mut max_sim = f32::MIN;

            for (i, vec) in dense_vectors.iter().enumerate() {
                if selected_indices.contains(&i) {
                    continue;
                }

                let sim: f32 = vec.iter().zip(centroid.iter()).map(|(a, b)| a * b).sum();
                if sim > max_sim {
                    max_sim = sim;
                    best_idx = i;
                }
            }
            selected_indices.push(best_idx);
        }

        selected_indices
            .iter()
            .map(|&idx| subjects[idx].clone())
            .collect()
    }
}

// Qwen3.5 embeddings implementation (optional, slower but more accurate)
#[cfg(feature = "qwen-embeddings")]
mod diversity {
    use super::*;

    use candle_core::IndexOp;

    struct EmbeddingModel {
        model: Qwen3_5Model,
        tokenizer: Arc<Tokenizer>,
        device: Device,
    }

    static EMBEDDING_MODEL: OnceLock<Option<Arc<Mutex<EmbeddingModel>>>> = OnceLock::new();

    fn get_embedding_model() -> Option<&'static Arc<Mutex<EmbeddingModel>>> {
        EMBEDDING_MODEL
            .get_or_init(|| match load_embedding_model() {
                Ok(model) => Some(Arc::new(Mutex::new(model))),
                Err(e) => {
                    tracing::error!("Failed to load embedding model: {}", e);
                    None
                }
            })
            .as_ref()
    }

    fn load_embedding_model() -> anyhow::Result<EmbeddingModel> {
        let device = Device::Cpu;

        // Load model: check env var for local path, otherwise download from HF
        let model_path = if let Ok(path) = std::env::var("QWEN3_MODEL_PATH") {
            // Use local file if specified
            std::path::PathBuf::from(path)
        } else {
            println!("downloading");
            // Download from HF and cache in proper cache directory
            let api = hf_hub::api::sync::Api::new()
                .map_err(|e| anyhow::anyhow!("Failed to initialize HF API: {}", e))?;
            let repo = api.model("unsloth/Qwen3.5-0.8B-GGUF".to_string());
            repo.get("Qwen3.5-0.8B-Q4_K_M.gguf")
                .map_err(|e| anyhow::anyhow!("Failed to download model from HF: {}", e))?
        };
        println!("downloaded");

        let mut file = std::fs::File::open(&model_path)
            .map_err(|e| anyhow::anyhow!("Failed to open model file at {:?}: {}", model_path, e))?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("Failed to read GGUF file: {}", e))?;
        println!("loading model from gguf");
        let model = Qwen3_5Model::from_gguf(content, &mut file, &device)?;

        // Load tokenizer from HF (will use HF cache)
        let tokenizer = {
            println!("downloading tokenizer");
            let api = hf_hub::api::sync::Api::new()
                .map_err(|e| anyhow::anyhow!("Failed to initialize HF API for tokenizer: {}", e))?;
            let repo = api.model("Qwen/Qwen3.5-0.8B".to_string());
            let tokenizer_path = repo
                .get("tokenizer.json")
                .map_err(|e| anyhow::anyhow!("Failed to download tokenizer from HF: {}", e))?;
            println!("loading tokenizer");
            Tokenizer::from_file(tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?
        };

        Ok(EmbeddingModel {
            model,
            tokenizer: Arc::new(tokenizer),
            device,
        })
    }

    fn generate_embedding(model: &mut EmbeddingModel, text: &str) -> anyhow::Result<Vec<f32>> {
        // 1. Guard against empty strings to prevent Rank 0 tensor errors
        if text.trim().is_empty() {
            // Qwen3.5-0.8B uses a hidden dimension of 1536.
            // We return a tiny non-zero dummy vector to prevent divide-by-zero in cosine similarity.
            return Ok(vec![0.0001; 1536]);
        }

        let encoding = model
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let tokens = encoding.get_ids();

        // Secondary guard just in case the tokenizer strips out all characters
        if tokens.is_empty() {
            return Ok(vec![0.0001; 1536]);
        }

        let token_ids = Tensor::new(tokens, &model.device)?.unsqueeze(0)?;

        // 2. CRITICAL FIX: Clear the Key-Value cache!
        // This ensures the model treats each subject as a completely independent sequence
        // rather than trying to continue the sentence from the previous subject.
        model.model.clear_kv_cache();

        let output = model.model.forward(&token_ids, 0)?;
        let dims = output.dims();

        // The model returns [batch_size, flattened_output]
        // We need to extract the last hidden_size (1536) elements as the embedding
        const HIDDEN_SIZE: usize = 1536;

        if dims.len() != 2 {
            anyhow::bail!("Expected 2D output, got {:?}", dims);
        }

        let total_len = dims[1];
        if total_len < HIDDEN_SIZE {
            anyhow::bail!("Output too small: {} < {}", total_len, HIDDEN_SIZE);
        }

        // Extract the last HIDDEN_SIZE elements
        let embedding = output
            .narrow(0, 0, 1)?              // Select batch 0: [1, total_len]
            .narrow(1, total_len - HIDDEN_SIZE, HIDDEN_SIZE)?  // Last 1536 elements: [1, 1536]
            .squeeze(0)?;                   // Remove batch dim: [1536]

        let embedding_vec = embedding.to_vec1::<f32>()?;

        Ok(embedding_vec)
    }

    fn cosine_similarity(vec_a: &[f32], vec_b: &[f32]) -> f64 {
        if vec_a.len() != vec_b.len() {
            return 0.0;
        }

        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for i in 0..vec_a.len() {
            dot_product += (vec_a[i] * vec_b[i]) as f64;
            norm_a += (vec_a[i] * vec_a[i]) as f64;
            norm_b += (vec_b[i] * vec_b[i]) as f64;
        }

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a.sqrt() * norm_b.sqrt())
    }

    pub fn select_diverse_subjects(subjects: Vec<String>, limit: usize) -> Vec<String> {
        if subjects.len() <= limit {
            return subjects;
        }

        let Some(model_mutex) = get_embedding_model() else {
            tracing::warn!("Embedding model not available, falling back to simple selection");
            return subjects.into_iter().take(limit).collect();
        };

        let mut model = match model_mutex.lock() {
            Ok(m) => m,
            Err(e) => {
                tracing::error!("Failed to lock embedding model: {}", e);
                return subjects.into_iter().take(limit).collect();
            }
        };

        println!("running embeddings");

        let embeddings: Vec<Vec<f32>> = subjects
            .iter()
            .map(|s| {
                generate_embedding(&mut model, s).unwrap_or_else(|e| {
                    eprintln!("Failed to generate embedding: {}", e);
                    vec![0.0001; 1536] // Match Qwen's hidden size and avoid 0-vector math
                })
            })
            .collect();

        println!("embeddings: {:?}", embeddings);

        drop(model);

        // --- SPHERICAL K-MEANS CLUSTERING ---
        let n_dims = embeddings[0].len();

        // 1. L2-Normalize all embeddings so Dot Product == Cosine Similarity
        let mut normalized_embeddings = embeddings.clone();
        for emb in &mut normalized_embeddings {
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                emb.iter_mut().for_each(|x| *x /= norm);
            }
        }

        // 2. Initialize K centroids (using the first 'limit' subjects as starting points for determinism)
        let mut centroids: Vec<Vec<f32>> = normalized_embeddings.iter().take(limit).cloned().collect();
        let mut assignments = vec![0; subjects.len()];

        // 3. Lloyd's Algorithm (10 iterations is usually plenty for < 100 items)
        for _ in 0..10 {
            // Step A: Assign each point to the nearest centroid
            for (i, emb) in normalized_embeddings.iter().enumerate() {
                let mut best_cluster = 0;
                let mut max_sim = f32::MIN;

                for (c_idx, centroid) in centroids.iter().enumerate() {
                    let sim: f32 = emb.iter().zip(centroid.iter()).map(|(a, b)| a * b).sum();
                    if sim > max_sim {
                        max_sim = sim;
                        best_cluster = c_idx;
                    }
                }
                assignments[i] = best_cluster;
            }

            // Step B: Recompute centroids as the mean of assigned points
            let mut new_centroids = vec![vec![0.0; n_dims]; limit];
            let mut counts = vec![0; limit];

            for (i, &cluster) in assignments.iter().enumerate() {
                for j in 0..n_dims {
                    new_centroids[cluster][j] += normalized_embeddings[i][j];
                }
                counts[cluster] += 1;
            }

            // Step C: L2-Normalize the new centroids back to the sphere surface
            for (c_idx, centroid) in new_centroids.iter_mut().enumerate() {
                if counts[c_idx] > 0 {
                    // Prevent division by zero for empty clusters
                    let norm: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        centroid.iter_mut().for_each(|x| *x /= norm);
                    }
                    centroids[c_idx] = centroid.clone();
                }
            }
        }

        // 4. Find the actual email subject closest to each finalized centroid
        let mut selected_indices = Vec::new();
        for centroid in &centroids {
            let mut best_idx = 0;
            let mut max_sim = f32::MIN;

            for (i, emb) in normalized_embeddings.iter().enumerate() {
                if selected_indices.contains(&i) {
                    continue;
                } // Don't pick duplicates

                let sim: f32 = emb.iter().zip(centroid.iter()).map(|(a, b)| a * b).sum();
                if sim > max_sim {
                    max_sim = sim;
                    best_idx = i;
                }
            }
            selected_indices.push(best_idx);
        }

        selected_indices
            .iter()
            .map(|&idx| subjects[idx].clone())
            .collect()
    }
}

// Business logic functions (testable, independent of MCP)
impl JmapServer {
    pub async fn search_emails_impl(
        &self,
        args: SearchEmailsArgs,
    ) -> anyhow::Result<Vec<EmailPreview>> {
        let mut conditions = Vec::new();

        // Apply named filter if specified
        if let Some(filter_name) = &args.filter {
            let store = self
                .filters
                .read()
                .map_err(|e| anyhow::anyhow!("Lock error: {}", e))?;

            if let Some(named_filter) = store.filters.get(filter_name) {
                if let Some(ref froms) = named_filter.from {
                    for from in froms {
                        conditions.push(EmailFilter::from(from.clone()));
                    }
                }
                if let Some(ref tos) = named_filter.to {
                    for to in tos {
                        conditions.push(EmailFilter::to(to.clone()));
                    }
                }
                if let Some(ref subj) = named_filter.subject {
                    conditions.push(EmailFilter::subject(subj.clone()));
                }
                if let Some(ref text) = named_filter.text {
                    conditions.push(EmailFilter::text(text.clone()));
                }
            }
        }

        if let Some(q) = args.query {
            conditions.push(EmailFilter::text(q));
        }

        if let Some(f) = args.from {
            conditions.push(EmailFilter::from(f));
        }

        if let Some(t) = args.to {
            conditions.push(EmailFilter::to(t));
        }

        if let Some(mb) = args.in_mailbox {
            conditions.push(EmailFilter::in_mailbox(mb));
        }

        if let Some(before) = args.before {
            let before = chrono::DateTime::parse_from_rfc3339(&before)
                .map_err(|e| anyhow::anyhow!("Invalid date: {}", e))?;
            conditions.push(EmailFilter::before(before.timestamp()));
        }

        if let Some(after) = args.after {
            let after = chrono::DateTime::parse_from_rfc3339(&after)
                .map_err(|e| anyhow::anyhow!("Invalid date: {}", e))?;
            conditions.push(EmailFilter::after(after.timestamp()));
        }

        if let Some(seen) = args.seen {
            if seen {
                conditions.push(EmailFilter::has_keyword("$seen"));
            } else {
                conditions.push(EmailFilter::not_keyword("$seen"));
            }
        }

        if let Some(flagged) = args.flagged {
            if flagged {
                conditions.push(EmailFilter::has_keyword("$flagged"));
            } else {
                conditions.push(EmailFilter::not_keyword("$flagged"));
            }
        }

        let limit = args.limit.unwrap_or(50);

        let mut request = self.client.build();
        let query_request = request.query_email().limit(limit);

        if !conditions.is_empty() {
            if conditions.len() == 1 {
                query_request.filter(conditions.pop().unwrap());
            } else {
                query_request.filter(Filter::and(conditions));
            }
        }

        let result = request.send_single::<QueryResponse>().await?;
        let ids: Vec<String> = result.ids().to_vec();

        // Now fetch preview data for each email
        let mut previews = Vec::new();
        for id in ids {
            if let Some(email) = self
                .client
                .email_get(&id, None::<Vec<jmap_client::email::Property>>)
                .await?
            {
                let from_str = email
                    .from()
                    .and_then(|addrs| addrs.first())
                    .map(|addr| addr.email().to_string());

                let keywords = email.keywords();
                let seen = keywords.iter().any(|k| *k == "$seen");
                let flagged = keywords.iter().any(|k| *k == "$flagged");

                previews.push(EmailPreview {
                    id: id.clone(),
                    from: from_str,
                    subject: email.subject().map(|s| s.to_string()),
                    preview: email.preview().map(|s| s.to_string()),
                    received_at: email.received_at().map(|ts| {
                        chrono::DateTime::from_timestamp(ts, 0)
                            .map(|dt| dt.to_rfc3339())
                            .unwrap_or_default()
                    }),
                    seen,
                    flagged,
                });
            }
        }

        Ok(previews)
    }

    pub async fn get_emails_impl(
        client: &jmap_client::client::Client,
        ids: Vec<String>,
    ) -> anyhow::Result<Vec<jmap_client::email::Email>> {
        let mut emails = Vec::new();
        for id in ids {
            let email = client
                .email_get(&id, Option::<Vec<jmap_client::email::Property>>::None)
                .await?;
            if let Some(e) = email {
                emails.push(e);
            }
        }
        Ok(emails)
    }

    pub async fn get_threads_impl(
        client: &jmap_client::client::Client,
        ids: Vec<String>,
    ) -> anyhow::Result<Vec<jmap_client::thread::Thread>> {
        let mut threads = Vec::new();
        for id in ids {
            let thread = client.thread_get(&id).await?;
            if let Some(t) = thread {
                threads.push(t);
            }
        }
        Ok(threads)
    }

    pub async fn get_mailboxes_impl(
        client: &jmap_client::client::Client,
    ) -> anyhow::Result<Vec<jmap_client::mailbox::Mailbox>> {
        let query_result = client
            .mailbox_query(
                Option::<jmap_client::mailbox::query::Filter>::None,
                Some([]),
            )
            .await?;

        let mut mailboxes = Vec::new();
        for id in query_result.ids() {
            let mb = client
                .mailbox_get(&id, Option::<Vec<jmap_client::mailbox::Property>>::None)
                .await?;
            if let Some(m) = mb {
                mailboxes.push(m);
            }
        }
        Ok(mailboxes)
    }

    pub async fn bulk_action_by_query_impl(
        &self,
        args: BulkActionByQueryArgs,
    ) -> anyhow::Result<usize> {
        // First, search for matching emails
        let search_args = SearchEmailsArgs {
            query: args.query,
            from: args.from,
            to: args.to,
            in_mailbox: args.in_mailbox,
            before: args.before,
            after: args.after,
            limit: args.limit,
            filter: args.filter,
            seen: args.seen,
            flagged: args.flagged,
        };

        let previews = self.search_emails_impl(search_args).await?;
        let ids: Vec<String> = previews.iter().map(|p| p.id.clone()).collect();

        if ids.is_empty() {
            return Ok(0);
        }

        let count = ids.len();

        // Perform the requested action
        match args.action.as_str() {
            "delete" => {
                let mut request = self.client.build();
                request.set_email().destroy(ids);
                request.send().await?;
            }
            "move" => {
                let mailbox_id = args
                    .target_mailbox_id
                    .ok_or_else(|| anyhow::anyhow!("target_mailbox_id required for move action"))?;
                let mut request = self.client.build();
                {
                    let email_set = request.set_email();
                    for id in ids {
                        email_set.update(id).mailbox_id(&mailbox_id, true);
                    }
                }
                request.send().await?;
            }
            "mark_seen" => {
                let mut request = self.client.build();
                {
                    let email_set = request.set_email();
                    for id in ids {
                        email_set.update(id).keyword("$seen", true);
                    }
                }
                request.send().await?;
            }
            "mark_unseen" => {
                let mut request = self.client.build();
                {
                    let email_set = request.set_email();
                    for id in ids {
                        email_set.update(id).keyword("$seen", false);
                    }
                }
                request.send().await?;
            }
            _ => return Err(anyhow::anyhow!("Invalid action: {}", args.action)),
        }

        Ok(count)
    }

    pub async fn get_inbox_summary_impl(
        &self,
        args: GetInboxSummaryArgs,
    ) -> anyhow::Result<Vec<SenderSummary>> {
        let now = chrono::Utc::now();
        let hours = args.hours;

        // Determine the date range based on provided parameters
        let (before, after) = match (args.before, args.after, hours) {
            // Both before and after specified: use them directly
            (Some(before), Some(after), _) => (Some(before), Some(after)),

            // hours + before: calculate after as (before - hours)
            (Some(before), None, Some(h)) => {
                let before_dt = chrono::DateTime::parse_from_rfc3339(&before)
                    .map_err(|e| anyhow::anyhow!("Invalid before date: {}", e))?;
                let after_dt = before_dt - chrono::Duration::hours(h as i64);
                (Some(before), Some(after_dt.to_rfc3339()))
            }

            // Only before: all emails before this time
            (Some(before), None, None) => (Some(before), None),

            // hours + after: calculate before as (after + hours)
            (None, Some(after), Some(h)) => {
                let after_dt = chrono::DateTime::parse_from_rfc3339(&after)
                    .map_err(|e| anyhow::anyhow!("Invalid after date: {}", e))?;
                let before_dt = after_dt + chrono::Duration::hours(h as i64);
                (Some(before_dt.to_rfc3339()), Some(after))
            }

            // Only after: all emails after this time (cap at now)
            (None, Some(after), None) => (Some(now.to_rfc3339()), Some(after)),

            // Only hours: from (now - hours) to now
            (None, None, Some(h)) => {
                let after_dt = now - chrono::Duration::hours(h as i64);
                (Some(now.to_rfc3339()), Some(after_dt.to_rfc3339()))
            }

            // None specified: default to last 24 hours
            (None, None, None) => {
                let after_dt = now - chrono::Duration::hours(24);
                (Some(now.to_rfc3339()), Some(after_dt.to_rfc3339()))
            }
        };

        // Search for recent emails
        let search_args = SearchEmailsArgs {
            query: None,
            from: None,
            to: None,
            in_mailbox: args.in_mailbox,
            before,
            after,
            limit: Some(500), // Fetch more for summarization
            filter: None,
            seen: None,
            flagged: None,
        };

        let previews = self.search_emails_impl(search_args).await?;

        // Group by sender
        let mut sender_map: HashMap<String, Vec<EmailPreview>> = HashMap::new();
        for preview in previews {
            let sender = preview
                .from
                .clone()
                .unwrap_or_else(|| "Unknown".to_string());
            sender_map
                .entry(sender)
                .or_insert_with(Vec::new)
                .push(preview);
        }

        // Build summary
        let mut summaries: Vec<SenderSummary> = sender_map
            .into_iter()
            .map(|(sender, emails)| {
                let email_ids: Vec<String> = emails.iter().map(|e| e.id.clone()).collect();
                let unread_count = emails.iter().filter(|e| !e.seen).count();
                let all_subjects: Vec<String> =
                    emails.iter().filter_map(|e| e.subject.clone()).collect();
                let subjects = diversity::select_diverse_subjects(all_subjects, 10);

                SenderSummary {
                    sender,
                    email_ids,
                    unread_count,
                    subjects,
                }
            })
            .collect();

        // Sort by email count descending
        summaries.sort_by(|a, b| b.email_ids.len().cmp(&a.email_ids.len()));

        Ok(summaries)
    }
}

// MCP tool handlers
impl JmapServer {
    pub fn public_email_router() -> ToolRouter<Self> {
        Self::email_router()
    }
}

#[tool_router(router = email_router)]
impl JmapServer {
    #[tool(
        description = "Search emails with filtering (text, from, to, date, mailbox, keywords). Returns preview data including subject, from, preview text, and seen/flagged status."
    )]
    async fn search_emails(
        &self,
        Parameters(args): Parameters<SearchEmailsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let previews = self
            .search_emails_impl(args)
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![rmcp::model::Content::json(
            json!({ "emails": previews }),
        )
        .unwrap()]))
    }

    #[tool(description = "Retrieve full details (body, headers) for specific IDs.")]
    async fn get_emails(
        &self,
        Parameters(args): Parameters<GetEmailsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let emails = Self::get_emails_impl(&self.client, args.ids)
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

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
        let threads = Self::get_threads_impl(&self.client, args.ids)
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

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
        let mailboxes = Self::get_mailboxes_impl(&self.client)
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

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

    #[tool(
        description = "Perform bulk actions (delete, move, mark_seen, mark_unseen) on emails matching a search query. Useful for mass-deleting alerts or moving filtered emails."
    )]
    async fn bulk_action_by_query(
        &self,
        Parameters(args): Parameters<BulkActionByQueryArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let count = self
            .bulk_action_by_query_impl(args)
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![rmcp::model::Content::text(
            format!("Processed {} emails", count),
        )]))
    }

    #[tool(
        description = "Get a summary of recent emails grouped by sender. Shows count, unread count, and sample subjects for each sender. Supports date range filtering with 'before' and 'after' (ISO-8601 format) or 'hours' for relative time. Useful for quick inbox triage."
    )]
    async fn get_inbox_summary(
        &self,
        Parameters(args): Parameters<GetInboxSummaryArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let summary = self
            .get_inbox_summary_impl(args)
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![rmcp::model::Content::json(
            json!({ "summary": summary }),
        )
        .unwrap()]))
    }
}

#[cfg(test)]
mod tests {
    use super::diversity::select_diverse_subjects;

    #[test]
    fn test_select_diverse_subjects_empty() {
        let subjects = vec![];
        let result = select_diverse_subjects(subjects, 5);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_select_diverse_subjects_less_than_limit() {
        let subjects = vec![
            "Meeting tomorrow".to_string(),
            "Urgent: Review needed".to_string(),
        ];
        let result = select_diverse_subjects(subjects.clone(), 5);
        assert_eq!(result.len(), 2);
        assert_eq!(result, subjects);
    }

    #[test]
    fn test_select_diverse_subjects_exact_limit() {
        let subjects = vec![
            "Meeting tomorrow".to_string(),
            "Urgent: Review needed".to_string(),
            "Weekly report".to_string(),
        ];
        let result = select_diverse_subjects(subjects.clone(), 3);
        assert_eq!(result.len(), 3);
        assert_eq!(result, subjects);
    }

    #[test]
    fn test_select_diverse_subjects_diverse_selection() {
        let subjects = vec![
            "Meeting tomorrow at 3pm".to_string(),
            "Meeting tomorrow at 4pm".to_string(), // Similar to first
            "Meeting tomorrow at 5pm".to_string(), // Similar to first
            "Urgent: Security incident detected".to_string(), // Very different
            "Weekly financial report Q4".to_string(), // Different
            "Your package has been delivered".to_string(), // Different
        ];

        let result = select_diverse_subjects(subjects, 3);
        assert_eq!(result.len(), 3);

        // K-Means should prefer diverse subjects over similar ones
        // Count how many are from the similar "Meeting tomorrow" cluster
        let meeting_count = result.iter().filter(|s| s.contains("Meeting tomorrow")).count();

        // At least one diverse topic should be included
        let diverse_count = result.iter().filter(|s|
            s.contains("Security incident") ||
            s.contains("financial report") ||
            s.contains("package")
        ).count();

        assert!(diverse_count >= 1, "Should include at least one diverse topic, got {}", diverse_count);
        assert!(meeting_count < 3, "Should not select all 'Meeting tomorrow' variants, got {}", meeting_count);
    }

    #[test]
    fn test_select_diverse_subjects_identical_subjects() {
        let subjects = vec![
            "Same subject".to_string(),
            "Same subject".to_string(),
            "Same subject".to_string(),
            "Different subject entirely".to_string(),
        ];

        let result = select_diverse_subjects(subjects, 2);
        assert_eq!(result.len(), 2);

        // K-Means should identify 2 clusters and select one from each
        // Should include both "Same subject" and "Different subject entirely"
        assert!(result.contains(&"Same subject".to_string()));
        assert!(result.contains(&"Different subject entirely".to_string()));
    }

    #[test]
    fn test_select_diverse_subjects_single_word_subjects() {
        let subjects = vec![
            "Hello".to_string(),
            "World".to_string(),
            "Test".to_string(),
            "Message".to_string(),
        ];

        let result = select_diverse_subjects(subjects, 2);
        assert_eq!(result.len(), 2);
        // All single words are maximally different from each other
        assert!(result.contains(&"Hello".to_string()));
    }

    #[test]
    fn test_select_diverse_subjects_empty_strings() {
        let subjects = vec!["".to_string(), "Real subject".to_string(), "".to_string()];

        let result = select_diverse_subjects(subjects, 2);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_select_diverse_subjects_all_different() {
        let subjects = vec![
            "First".to_string(),
            "Second very different words here".to_string(),
            "Third completely unrelated content".to_string(),
            "Fourth another unique message".to_string(),
        ];

        let result = select_diverse_subjects(subjects.clone(), 3);
        assert_eq!(result.len(), 3);

        // K-Means should select 3 diverse subjects from the 4 available
        // All subjects are different, so all selected ones should be from the input
        for selected in &result {
            assert!(subjects.contains(selected), "Selected subject '{}' should be from input", selected);
        }

        // Verify all selected subjects are unique
        let unique_count = result.iter().collect::<std::collections::HashSet<_>>().len();
        assert_eq!(unique_count, 3, "All selected subjects should be unique");
    }

    #[test]
    fn test_select_diverse_subjects_topical_clustering() {
        // Test that the algorithm picks from different topics
        let subjects = vec![
            "Database migration v1.0".to_string(),
            "Database migration v1.1".to_string(), // Similar topic
            "Database migration v1.2".to_string(), // Similar topic
            "Python tutorial for beginners".to_string(), // Different topic
            "Python advanced features".to_string(), // Similar to above
            "Marketing campaign results".to_string(), // Different topic
            "Marketing strategy 2024".to_string(), // Similar to above
            "Lunch invitation Friday".to_string(), // Different topic
        ];

        let result = select_diverse_subjects(subjects, 4);
        assert_eq!(result.len(), 4);

        // Should spread across topics rather than picking multiple from same topic
        let topics = result
            .iter()
            .map(|s| {
                if s.contains("Database") {
                    "db"
                } else if s.contains("Python") {
                    "python"
                } else if s.contains("Marketing") {
                    "marketing"
                } else if s.contains("Lunch") {
                    "lunch"
                } else {
                    "other"
                }
            })
            .collect::<Vec<_>>();

        // Should have picked from at least 3 different topics
        let unique_topics: std::collections::HashSet<_> = topics.into_iter().collect();
        assert!(
            unique_topics.len() >= 3,
            "Expected diverse topics, got: {:?}",
            result
        );
    }

    #[test]
    fn test_select_diverse_subjects_limit_one() {
        let subjects = vec![
            "First".to_string(),
            "Second".to_string(),
            "Third".to_string(),
        ];

        let result = select_diverse_subjects(subjects.clone(), 1);
        assert_eq!(result.len(), 1);
        // K-Means with limit=1 should select one representative subject
        assert!(subjects.contains(&result[0]), "Selected subject should be from input");
    }

    // Tests specific to Qwen embeddings implementation
    #[cfg(feature = "qwen-embeddings")]
    mod qwen_tests {
        use super::*;

        #[test]
        fn test_qwen_select_diverse_subjects_semantic_similarity() {
            // Test that Qwen embeddings can distinguish semantic similarity
            // better than simple TF-IDF
            let subjects = vec![
                "The quick brown fox jumps over the lazy dog".to_string(),
                "A fast auburn fox leaps above a sleepy canine".to_string(), // Semantically similar
                "Database migration completed successfully".to_string(),     // Completely different
                "The weather is sunny today".to_string(),                    // Different topic
                "System backup finished without errors".to_string(), // Technical but different
            ];

            let result = select_diverse_subjects(subjects.clone(), 3);
            assert_eq!(result.len(), 3);

            // First should always be selected
            assert_eq!(result[0], subjects[0]);

            // Should prefer semantically different topics
            // The database and weather subjects should be more likely than the
            // semantically similar fox sentence
            let has_database = result.iter().any(|s| s.contains("Database"));
            let has_weather = result.iter().any(|s| s.contains("weather"));
            let has_similar_fox = result.iter().any(|s| s.contains("auburn"));

            // At least one of the semantically different subjects should be selected
            assert!(
                has_database || has_weather,
                "Expected semantically diverse subjects, got: {:?}",
                result
            );
        }

        #[test]
        fn test_qwen_model_graceful_fallback() {
            // Test that if model fails to load, we fallback gracefully
            // This test will pass regardless of whether the model loads successfully
            let subjects = vec![
                "Subject 1".to_string(),
                "Subject 2".to_string(),
                "Subject 3".to_string(),
                "Subject 4".to_string(),
            ];

            let result = select_diverse_subjects(subjects.clone(), 2);

            // Should return results even if model loading fails
            assert_eq!(result.len(), 2);
            assert!(result.iter().all(|s| subjects.contains(s)));
        }

        #[test]
        fn test_qwen_embedding_consistency() {
            // Test that running the same subjects twice gives the same results
            let subjects = vec![
                "Machine learning algorithms".to_string(),
                "Deep neural networks".to_string(),
                "Pizza delivery service".to_string(),
                "Italian cuisine recipes".to_string(),
                "Cloud computing infrastructure".to_string(),
            ];

            let result1 = select_diverse_subjects(subjects.clone(), 3);
            let result2 = select_diverse_subjects(subjects.clone(), 3);

            // Results should be deterministic
            assert_eq!(
                result1, result2,
                "Qwen embeddings should produce consistent results"
            );
        }

        #[test]
        fn test_qwen_technical_vs_casual() {
            // Test that Qwen can distinguish between technical and casual language
            let subjects = vec![
                "Kubernetes pod scheduling optimization".to_string(),
                "Docker container orchestration strategies".to_string(), // Technical, similar domain
                "Hey, let's grab lunch tomorrow!".to_string(),           // Casual
                "Birthday party invitation for Saturday".to_string(),    // Casual, events
                "Microservices architecture patterns".to_string(), // Technical, similar domain
                "Coffee break at 3pm?".to_string(),                // Casual
            ];

            let result = select_diverse_subjects(subjects.clone(), 3);
            assert_eq!(result.len(), 3);

            // Count how many technical vs casual subjects were selected
            let technical_count = result
                .iter()
                .filter(|s| {
                    s.contains("Kubernetes") || s.contains("Docker") || s.contains("Microservices")
                })
                .count();

            let casual_count = result
                .iter()
                .filter(|s| s.contains("lunch") || s.contains("party") || s.contains("Coffee"))
                .count();

            // Should select from both categories to maximize diversity
            assert!(
                technical_count >= 1 && casual_count >= 1,
                "Expected mix of technical and casual subjects, got: {:?}",
                result
            );
        }

        #[test]
        fn test_qwen_multilingual_fallback() {
            // Test behavior with mixed language content
            // Qwen should handle this gracefully even if embeddings quality varies
            let subjects = vec![
                "Hello world".to_string(),
                "Bonjour le monde".to_string(), // French
                "Hola mundo".to_string(),       // Spanish
                "System error occurred".to_string(),
            ];

            let result = select_diverse_subjects(subjects.clone(), 2);
            assert_eq!(result.len(), 2);

            // Should complete without panicking
            assert!(result.iter().all(|s| subjects.contains(s)));
        }

        #[test]
        fn test_qwen_empty_and_short_subjects() {
            // Test edge cases with Qwen embeddings
            let subjects = vec![
                "".to_string(),
                "A".to_string(),
                "This is a much longer subject with many words".to_string(),
                "B".to_string(),
            ];

            let result = select_diverse_subjects(subjects.clone(), 2);
            assert_eq!(result.len(), 2);
        }

        #[test]
        fn test_qwen_large_batch() {
            // Test that Qwen can handle a larger batch of subjects
            let subjects: Vec<String> = (0..20)
                .map(|i| format!("Subject number {} with some unique content", i))
                .collect();

            let result = select_diverse_subjects(subjects.clone(), 5);
            assert_eq!(result.len(), 5);

            // All selected subjects should be unique
            let unique_count = result
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len();
            assert_eq!(unique_count, 5);
        }

        #[test]
        fn test_qwen_specialized_domains() {
            // Test semantic understanding within specialized domains
            let subjects = vec![
                "PostgreSQL query optimization techniques".to_string(),
                "MySQL index performance tuning".to_string(), // Similar domain
                "MongoDB aggregation pipeline design".to_string(), // Same domain, NoSQL
                "Chocolate cake recipe instructions".to_string(), // Completely different
                "SQLite transaction handling best practices".to_string(), // Similar domain
                "Gardening tips for spring season".to_string(), // Completely different
            ];

            let result = select_diverse_subjects(subjects.clone(), 3);
            assert_eq!(result.len(), 3);

            // Should select from different domains
            let database_count = result
                .iter()
                .filter(|s| s.contains("SQL") || s.contains("MongoDB"))
                .count();

            // Should not select all database subjects - diversity should spread topics
            assert!(
                database_count <= 2,
                "Expected diverse domain selection, but got {} database subjects: {:?}",
                database_count,
                result
            );
        }

        #[test]
        fn test_qwen_billing_payment_vs_flowers() {
            // Test that Qwen understands "billing" and "payment" are semantically similar,
            // while "flowers" is dissimilar
            let subjects = vec![
                "Your monthly billing statement is ready".to_string(),
                "Payment confirmation for invoice #12345".to_string(), // Similar to billing
                "Payment processing completed successfully".to_string(), // Similar to billing/payment
                "Beautiful spring flowers are now in bloom".to_string(), // Completely different
                "Invoice payment due next week".to_string(), // Similar to billing/payment
            ];

            let result = select_diverse_subjects(subjects.clone(), 2);
            assert_eq!(result.len(), 2);

            // Should select the flowers subject since it's most dissimilar
            let has_flowers = result.iter().any(|s| s.contains("flowers"));
            assert!(
                has_flowers,
                "Expected flowers subject to be selected as most dissimilar, got: {:?}",
                result
            );

            // Should select only one billing/payment subject
            let billing_payment_count = result
                .iter()
                .filter(|s| s.contains("billing") || s.contains("Payment") || s.contains("Invoice"))
                .count();

            assert_eq!(
                billing_payment_count, 1,
                "Expected only one billing/payment subject when flowers is selected, got {}: {:?}",
                billing_payment_count, result
            );
        }

        #[test]
        fn test_qwen_billing_payment_similarity_strong() {
            // Stronger test: with 3 billing/payment and 1 flowers, flowers should be selected
            let subjects = vec![
                "Your monthly billing statement".to_string(),
                "Payment received thank you".to_string(), // Semantically similar
                "Flowers for sale at discount".to_string(), // Dissimilar
                "Invoice payment reminder".to_string(),   // Similar to first two
            ];

            let result = select_diverse_subjects(subjects.clone(), 2);
            assert_eq!(result.len(), 2);

            // First subject is always selected
            assert_eq!(result[0], "Your monthly billing statement");

            println!("{:?}", result);

            // Second should be flowers (most dissimilar)
            assert_eq!(
                result[1], "Flowers for sale at discount",
                "Expected 'flowers' to be selected as most dissimilar from 'billing', got: {:?}",
                result
            );
        }
    }
}

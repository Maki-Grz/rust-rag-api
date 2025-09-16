use crate::types::Passage;
use mongodb::bson::doc;
use mongodb::options::FindOptions;
use rayon::prelude::*;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

pub async fn search_top_k(
    question_embedding: &[f32],
    k: usize,
    client: &mongodb::Client,
    db_name: &str,
    collection_name: &str,
    fetch_limit: Option<i64>,
) -> Result<Vec<Passage>, Box<dyn std::error::Error>> {
    let docs_collection = client
        .database(db_name)
        .collection::<Passage>(collection_name);

    let find_opts = FindOptions::builder()
        .projection(doc! {
            "text": 1,
            "embedding": 1,
            "metadata": 1,
        })
        .limit(fetch_limit.unwrap_or(2000))
        .build();

    let mut cursor = docs_collection
        .find(doc! {})
        .with_options(find_opts)
        .await?;
    let mut passages = Vec::new();

    use futures::TryStreamExt;
    while let Some(p) = cursor.try_next().await? {
        if !p.embedding.is_empty() {
            passages.push(p);
        }
    }

    let mut scored_passages: Vec<_> = passages
        .par_iter()
        .map(|p| {
            let sim = cosine_similarity(question_embedding, &p.embedding);
            (p.clone(), sim)
        })
        .collect();

    scored_passages
        .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_passages: Vec<Passage> = scored_passages
        .into_iter()
        .take(k)
        .map(|(p, _)| p)
        .collect();

    Ok(top_passages)
}

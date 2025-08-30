use crate::types::Passage;
use mongodb::bson::doc;

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
) -> Result<Vec<Passage>, Box<dyn std::error::Error>> {
    let docs_collection = client
        .database(db_name)
        .collection::<Passage>(collection_name);

    let mut cursor = docs_collection.find(doc! {}).await?;
    let mut scored_passages = Vec::new();

    use futures::TryStreamExt;

    while let Some(passage) = cursor.try_next().await? {
        if !passage.embedding.is_empty() {
            let similarity = cosine_similarity(question_embedding, &passage.embedding);
            scored_passages.push((passage, similarity));
        }
    }

    scored_passages.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let threshold = scored_passages
        .first()
        .map(|(_, sim)| *sim * 0.95)
        .unwrap_or(0.0);

    let top_candidates: Vec<_> = scored_passages
        .into_iter()
        .filter(|(_, sim)| *sim >= threshold)
        .collect();

    let top_passages: Vec<Passage> = top_candidates.into_iter().take(k).map(|(p, _)| p).collect();

    Ok(top_passages)
}

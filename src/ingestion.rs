use crate::types::{Metadata, Passage};
use crate::utils::compute_hash;
use mongodb::bson::doc;
use mongodb::Client;

pub fn segment_text(
    text: &str,
    metadata: Option<Metadata>,
    passage_size: usize,
    step: usize,
) -> Vec<Passage> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut passages = Vec::new();
    let mut start = 0;

    while start < words.len() {
        let end = usize::min(start + passage_size, words.len());
        let slice = &words[start..end];
        let passage_text = slice.join(" ");

        let passage = Passage {
            id: uuid::Uuid::new_v4().to_string(),
            text: passage_text,
            embedding: vec![],
            metadata: metadata.clone(),
            hash: None,
        };

        passages.push(passage);
        start += step;
    }

    passages
}

pub async fn store_passage(
    mut passage: Passage,
    client: &Client,
) -> Result<String, Box<dyn std::error::Error>> {
    let hash = compute_hash(&passage.text);

    let database = std::env::var("DATABASE").expect("Set env variable DATABASE first!");
    let collection = std::env::var("COLLECTION").expect("Set env variable COLLECTION first!");

    let docs_collection = client
        .database(database.as_str())
        .collection::<Passage>(collection.as_str());

    let existing = docs_collection
        .find_one(doc! { "hash": hash as i64 })
        .await?;

    if let Some(existing_passage) = existing {
        return Ok(existing_passage.id);
    }

    passage.hash = Some(hash as i64);
    docs_collection.insert_one(&passage).await?;

    Ok(passage.id.clone())
}

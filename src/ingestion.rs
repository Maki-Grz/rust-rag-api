use crate::types::{Metadata, Passage};
use crate::utils::compute_hash;
use mongodb::bson::doc;
use mongodb::Client;

pub fn segment_text(text: &str, metadata: Option<Metadata>) -> Vec<Passage> {
    let max_words = 400;
    let overlap = 60;

    let mut passages = Vec::new();

    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .filter(|p| !p.trim().is_empty())
        .collect();

    for paragraph in paragraphs {
        let sentences: Vec<&str> = paragraph
            .split_terminator(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        let mut start = 0;
        while start < sentences.len() {
            let mut passage_words = Vec::new();
            let mut end = start;

            while end < sentences.len() {
                let sentence_words: Vec<&str> = sentences[end].split_whitespace().collect();
                if passage_words.len() + sentence_words.len() > max_words {
                    break;
                }
                passage_words.extend(sentence_words);
                end += 1;
            }

            if !passage_words.is_empty() {
                let passage_text = passage_words.join(" ");
                let passage = Passage {
                    id: uuid::Uuid::new_v4().to_string(),
                    text: passage_text,
                    embedding: vec![],
                    metadata: metadata.clone(),
                    hash: None,
                };
                passages.push(passage);
            }

            let step = if overlap >= passage_words.len() {
                1
            } else {
                passage_words.len() - overlap
            };
            start += step;
        }
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

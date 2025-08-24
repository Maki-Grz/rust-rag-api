use crate::types::Passage;
use crate::utils::{compute_hash, sigmoid};
use actix_web::web;
use mongodb::bson::doc;
use mongodb::Client;
use std::cmp::min;

pub fn segment_text(text: &str) -> Vec<Passage> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let passage_size = 300;
    let step = 250;

    let mut passages: Vec<Passage> = Vec::new();
    let mut start = 0;
    while start < words.len() {
        let end = min(start + passage_size, words.len());
        let slice = &words[start..end];
        let text = slice.join(" ");

        let passage_struct = Passage {
            id: uuid::Uuid::new_v4().to_string(),
            text: text.to_string(),
            embedding: vec![],
            metadata: None,
            hash: None,
        };

        passages.push(passage_struct);
        start += step;
    }

    passages
}

pub fn compute_embedding(passage: &Passage) -> Vec<f32> {
    const EMBEDDING_SIZE: usize = 128;
    let words: Vec<&str> = passage.text.split_whitespace().collect();
    let mut embedding: Vec<f32> = vec![0.0; EMBEDDING_SIZE];

    let mut temp_embedding: Vec<f32> = Vec::new();
    for word in words {
        let c = word.chars().next().unwrap_or(' ') as u8;
        temp_embedding.push(sigmoid(c as f32));
    }

    let len = min(temp_embedding.len(), EMBEDDING_SIZE);
    embedding[..len].copy_from_slice(&temp_embedding[..len]);

    if embedding.len() > EMBEDDING_SIZE {
        embedding.truncate(EMBEDDING_SIZE);
    } else {
        embedding.resize(EMBEDDING_SIZE, 0.0);
    }

    embedding
}

pub async fn store_passage(
    mut passage: Passage,
    client: &web::Data<Client>,
) -> Result<String, Box<dyn std::error::Error>> {
    let hash = compute_hash(&passage.text);

    let docs_collection = client
        .database("test")
        .collection::<Passage>("paladium-wiki");
    let existing = docs_collection
        .find_one(doc! { "hash": hash as i64 })
        .await?;

    if existing.is_some() {
        return Ok(existing.unwrap().id);
    }

    passage.hash = Option::from(hash as i64);

    docs_collection.insert_one(&passage).await?;

    Ok(passage.id.clone())
}

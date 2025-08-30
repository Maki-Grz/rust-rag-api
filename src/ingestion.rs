use crate::types::{Metadata, Passage};
use crate::utils::compute_hash;
use mongodb::bson::doc;
use mongodb::Client;
use regex::Regex;
use tokenizers::Tokenizer;
use uuid::Uuid;

fn count_tokens(tokenizer: &Tokenizer, text: &str) -> usize {
    tokenizer.encode(text, true).unwrap().len()
}

fn keep_last_tokens(tokenizer: &Tokenizer, text: &str, n: usize) -> String {
    let encoding = tokenizer.encode(text, true).unwrap();
    let ids = encoding.get_ids();
    let start = if ids.len() > n { ids.len() - n } else { 0 };
    let slice = &ids[start..];
    tokenizer.decode(&*slice.to_vec(), true).unwrap()
}

fn split_sections(text: &str) -> Vec<String> {
    let re = Regex::new(r"(?m)^(\d+\.\s.*)").unwrap();
    let mut sections = Vec::new();
    let mut current = String::new();

    for line in text.lines() {
        if re.is_match(line) {
            if !current.trim().is_empty() {
                sections.push(current.trim().to_string());
                current.clear();
            }
        }
        current.push_str(line);
        current.push('\n');
    }

    if !current.trim().is_empty() {
        sections.push(current.trim().to_string());
    }

    sections
}

fn make_passage(text: &str, metadata: &Option<Metadata>) -> Passage {
    Passage {
        id: Uuid::new_v4().to_string(),
        text: text.to_string(),
        embedding: vec![],
        metadata: metadata.clone(),
        hash: None,
    }
}

pub fn segment_text(text: &str, metadata: Option<Metadata>, tokenizer: &Tokenizer) -> Vec<Passage> {
    let max_tokens = 512;
    let overlap_tokens = 50;

    let mut passages = Vec::new();

    let sections = split_sections(text);

    for section in sections {
        let paragraphs: Vec<&str> = section
            .split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .collect();

        for paragraph in paragraphs {
            let sentences = split_sections(paragraph);

            let mut buffer = String::new();
            let mut token_count = 0;

            for sentence in sentences {
                let sentence_tokens = count_tokens(tokenizer, &sentence);

                if token_count + sentence_tokens > max_tokens {
                    if !buffer.is_empty() {
                        passages.push(make_passage(&buffer, &metadata));
                    }

                    let overlap_text = keep_last_tokens(tokenizer, &buffer, overlap_tokens);
                    buffer = format!("{} {}", overlap_text, sentence);
                    token_count = count_tokens(tokenizer, &buffer);
                } else {
                    if !buffer.is_empty() {
                        buffer.push(' ');
                    }
                    buffer.push_str(&sentence);
                    token_count += sentence_tokens;
                }
            }

            if !buffer.is_empty() {
                passages.push(make_passage(&buffer, &metadata));
            }
        }
    }

    passages
}

pub async fn store_passage(
    mut passage: Passage,
    client: &Client,
    db_name: &str,
    collection_name: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let hash = compute_hash(&passage.text);

    let docs_collection = client
        .database(db_name)
        .collection::<Passage>(collection_name);

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

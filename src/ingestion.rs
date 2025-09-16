use crate::types::{Metadata, Passage};
use crate::utils::compute_hash;
use mongodb::bson::{doc, Bson};
use mongodb::Client;
use regex::Regex;
use tokenizers::Tokenizer;

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
        id: None,
        text: text.to_string(),
        embedding: vec![],
        metadata: metadata.clone(),
        hash: None,
    }
}

fn clean_text(text: &str) -> String {
    text.replace("\r\n", "\n")
        .replace('\r', "\n")
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn split_sentences(text: &str) -> Vec<String> {
    let re = Regex::new(r"(?m)([^.!?]*[.!?](?:\s|$))").unwrap();
    let mut sentences = Vec::new();
    let mut last_index = 0;

    for mat in re.find_iter(text) {
        let sentence = text[last_index..mat.end()].trim();
        if !sentence.is_empty() && sentence.len() > 10 {
            sentences.push(sentence.to_string());
        }
        last_index = mat.end();
    }

    if last_index < text.len() {
        let rest = text[last_index..].trim();
        if !rest.is_empty() {
            sentences.push(rest.to_string());
        }
    }

    if sentences.is_empty() && !text.trim().is_empty() {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut current_sentence = String::new();
        let mut word_count = 0;

        for word in words {
            if word_count > 0 {
                current_sentence.push(' ');
            }
            current_sentence.push_str(word);
            word_count += 1;

            if word_count >= 25 {
                sentences.push(current_sentence.trim().to_string());
                current_sentence.clear();
                word_count = 0;
            }
        }

        if !current_sentence.trim().is_empty() {
            sentences.push(current_sentence.trim().to_string());
        }
    }

    sentences
}

fn split_large_text(
    text: &str,
    tokenizer: &Tokenizer,
    max_tokens: usize,
    overlap_tokens: usize,
) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();
    let mut start_idx = 0;

    while start_idx < words.len() {
        let mut end_idx = start_idx;
        let mut current_chunk = String::new();

        while end_idx < words.len() {
            let test_text = if current_chunk.is_empty() {
                words[end_idx].to_string()
            } else {
                format!("{} {}", current_chunk, words[end_idx])
            };

            if count_tokens(tokenizer, &test_text) > max_tokens && !current_chunk.is_empty() {
                break;
            }

            current_chunk = test_text;
            end_idx += 1;
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        if end_idx < words.len() {
            let overlap_text =
                keep_last_tokens(tokenizer, &chunks[chunks.len() - 1], overlap_tokens);
            let overlap_words: Vec<&str> = overlap_text.split_whitespace().collect();
            start_idx = end_idx.saturating_sub(overlap_words.len());
        } else {
            break;
        }
    }

    chunks
}

pub fn segment_text(text: &str, metadata: Option<Metadata>, tokenizer: &Tokenizer) -> Vec<Passage> {
    let max_tokens = 200;
    let overlap_tokens = 30;
    let min_tokens = 20;

    let text = clean_text(text);
    let sections = split_sections(&text);

    let mut passages = Vec::new();

    for section in sections {
        let section_token_count = count_tokens(tokenizer, &section);

        if section_token_count > max_tokens * 3 {
            let chunks = split_large_text(&section, tokenizer, max_tokens, overlap_tokens);
            for chunk in chunks {
                if count_tokens(tokenizer, &chunk) >= min_tokens {
                    passages.push(make_passage(&chunk, &metadata));
                }
            }
            continue;
        }

        let paragraphs: Vec<&str> = section
            .split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .collect();

        for paragraph in paragraphs {
            let paragraph_tokens = count_tokens(tokenizer, paragraph);

            if paragraph_tokens > max_tokens {
                let chunks = split_large_text(paragraph, tokenizer, max_tokens, overlap_tokens);
                for chunk in chunks {
                    if count_tokens(tokenizer, &chunk) >= min_tokens {
                        passages.push(make_passage(&chunk, &metadata));
                    }
                }
                continue;
            }

            let sentences = split_sentences(paragraph);

            let mut buffer = String::new();
            let mut token_count = 0;

            for sentence in sentences {
                let sentence_tokens = count_tokens(tokenizer, &sentence);

                if sentence_tokens > max_tokens {
                    if !buffer.is_empty() && token_count >= min_tokens {
                        passages.push(make_passage(&buffer, &metadata));
                        buffer.clear();
                        token_count = 0;
                    }

                    let sentence_chunks =
                        split_large_text(&sentence, tokenizer, max_tokens, overlap_tokens);
                    for chunk in sentence_chunks {
                        if count_tokens(tokenizer, &chunk) >= min_tokens {
                            passages.push(make_passage(&chunk, &metadata));
                        }
                    }
                    continue;
                }

                if token_count + sentence_tokens > max_tokens {
                    if token_count >= min_tokens {
                        passages.push(make_passage(&buffer, &metadata));
                    }

                    let overlap_text = keep_last_tokens(tokenizer, &buffer, overlap_tokens);
                    buffer = if overlap_text.is_empty() {
                        sentence.clone()
                    } else {
                        format!("{} {}", overlap_text, sentence)
                    };
                    token_count = count_tokens(tokenizer, &buffer);
                } else {
                    if !buffer.is_empty() {
                        buffer.push(' ');
                    }
                    buffer.push_str(&sentence);
                    token_count += sentence_tokens;
                }
            }

            if !buffer.is_empty() && token_count >= min_tokens {
                passages.push(make_passage(&buffer, &metadata));
            }
        }
    }

    if passages.len() < 3 && !text.trim().is_empty() {
        let aggressive_chunks =
            split_large_text(&text, tokenizer, max_tokens / 2, overlap_tokens / 2);
        passages.clear();
        for chunk in aggressive_chunks {
            if count_tokens(tokenizer, &chunk) >= min_tokens {
                passages.push(make_passage(&chunk, &metadata));
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

    if let Some(existing_passage) = docs_collection
        .find_one(doc! { "hash": hash as i64 })
        .await?
    {
        return Ok(existing_passage.id.unwrap().to_string());
    }

    passage.hash = Some(hash as i64);

    let insert_result = docs_collection.insert_one(&passage).await?;

    let id_str = match insert_result.inserted_id {
        Bson::ObjectId(oid) => oid.to_string(),
        Bson::String(s) => s,
        other => return Err(format!("Unexpected inserted_id type: {:?}", other).into()),
    };

    Ok(id_str)
}

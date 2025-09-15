use crate::types::{LLMMessage, LLMRequest, LLMStreamResponse, Passage};
use futures_util::Stream;
use futures_util::StreamExt;
use std::pin::Pin;

pub async fn generate_answer(
    question: &str,
    context: &[Passage],
) -> Result<
    Pin<Box<dyn Stream<Item = Result<String, Box<dyn std::error::Error>>> + Send>>,
    Box<dyn std::error::Error>,
> {
    let llm_uri = std::env::var("LLM_URI").expect("Set env variable LLM_URI first!");
    let model_hash = std::env::var("MODEL_HASH").expect("Set env variable MODEL_HASH first!");
    let system_prompt =
        std::env::var("SYSTEM_PROMPT").expect("Set env variable SYSTEM_PROMPT first!");

    let context_text = context
        .iter()
        .enumerate()
        .map(|(i, passage)| format!("Passage {}: {}", i + 1, passage.text))
        .collect::<Vec<_>>()
        .join("\n\n");

    let system_prompt = format!(
        "{} \n
        [PASSAGE] :\n{}",
        system_prompt, context_text
    );

    let user_prompt = format!("QUESTION: {}\n", question);

    let llm_request = LLMRequest {
        model: model_hash.to_string(),
        messages: vec![
            LLMMessage {
                role: "system".to_string(),
                content: system_prompt,
            },
            LLMMessage {
                role: "user".to_string(),
                content: user_prompt,
            },
        ],
        stream: true,
    };

    let client = reqwest::Client::new();
    let response = client.post(llm_uri).json(&llm_request).send().await?;

    if !response.status().is_success() {
        return Err(format!("Erreur LLM: {}", response.status()).into());
    }

    let byte_stream = response.bytes_stream();

    let mapped_stream = byte_stream.filter_map(|chunk| async {
        match chunk {
            Ok(bytes) => {
                let text = String::from_utf8_lossy(&bytes);
                let mut parts = Vec::new();

                for line in text.lines() {
                    if line.starts_with("data: ") {
                        let json_str = &line[6..];
                        if json_str == "[DONE]" {
                            return Some(Ok("[DONE]".to_string()));
                        }

                        if let Ok(event) = serde_json::from_str::<LLMStreamResponse>(json_str) {
                            if let Some(choice) = event.choices.first() {
                                let content = &choice.delta.content;
                                parts.push(content.clone());
                            }
                        }
                    }
                }

                if parts.is_empty() {
                    None
                } else {
                    Some(Ok(parts.join("")))
                }
            }
            Err(e) => Some(Err(Box::new(e) as Box<dyn std::error::Error>)),
        }
    });

    Ok(Box::pin(mapped_stream))
}

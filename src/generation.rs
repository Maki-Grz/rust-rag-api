use crate::types::{LLMMessage, LLMRequest, LLMResponse, Passage};

pub async fn generate_answer(
    question: &str,
    context: &[Passage],
) -> Result<String, Box<dyn std::error::Error>> {
    let llm_uri = std::env::var("LLM_URI").expect("Set env variable LLM_URI first!");

    let context_text = context
        .iter()
        .enumerate()
        .map(|(i, passage)| format!("Passage {}: {}", i + 1, passage.text))
        .collect::<Vec<_>>()
        .join("\n\n");

    let system_prompt = format!(
        "Vous êtes un assistant qui répond uniquement en utilisant les informations fournies dans les passages ci-dessous. \
        Ne jamais inventer d'informations ou utiliser vos connaissances générales. \
        Si les passages ne contiennent pas suffisamment d'informations pour répondre à la question, dites-le clairement.\n\n\
        Passages disponibles:\n{}",
        context_text
    );

    let user_prompt = format!(
        "Question: {}\n\n\
        Répondez en utilisant uniquement les informations des passages fournis ci-dessus. \
        Citez les numéros des passages utilisés dans votre réponse.",
        question
    );

    let llm_request = LLMRequest {
        model: "ai/smollm2".to_string(),
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
        max_tokens: Some(500),
        temperature: Some(0.1),
    };

    let client = reqwest::Client::new();
    let response = client
        .post(llm_uri.as_str())
        .json(&llm_request)
        .send()
        .await?;

    if !response.status().is_success() {
        return Err(format!("Erreur LLM: {}", response.status()).into());
    }

    let llm_response: LLMResponse = response.json().await?;

    if let Some(choice) = llm_response.choices.first() {
        Ok(choice.message.content.clone())
    } else {
        Err("Aucune réponse générée par le LLM".into())
    }
}

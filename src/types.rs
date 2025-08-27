use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Passage {
    pub id: String,
    pub text: String,
    pub embedding: Vec<f32>,
    pub metadata: Option<Metadata>,
    pub hash: Option<i64>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Metadata {
    pub title: Option<String>,
    pub source: Option<String>,
    pub date: Option<String>,
    pub url: Option<String>,
}

#[derive(Deserialize)]
pub struct QuestionRequest {
    pub question: String,
}

#[derive(Serialize)]
pub struct AnswerResponse {
    pub answer: Option<String>,
    pub passages: Option<Vec<Passage>>,
    pub fallback_reason: Option<String>,
}

#[derive(Deserialize)]
pub struct IngestRequest {
    pub text: String,
    pub metadata: Option<Metadata>,
}

#[derive(Serialize)]
pub struct IngestResponse {
    pub passage_ids: Vec<String>,
    pub count: usize,
}

#[derive(Serialize, Deserialize)]
pub struct LLMMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct LLMRequest {
    pub model: String,
    pub messages: Vec<LLMMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

#[derive(Deserialize)]
pub struct LLMResponse {
    pub choices: Vec<LLMChoice>,
}

#[derive(Deserialize)]
pub struct LLMChoice {
    pub message: LLMMessage,
}

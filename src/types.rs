use mongodb::bson::{doc, oid::ObjectId};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Passage {
    #[serde(rename = "_id", skip_serializing_if = "Option::is_none")]
    pub id: Option<ObjectId>,

    pub text: String,
    pub embedding: Vec<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<i64>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Metadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub date: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

#[derive(Deserialize)]
pub struct IngestRequest {
    pub text: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

#[derive(Serialize)]
pub struct IngestResponse {
    pub passage_ids: Vec<String>,
    pub count: usize,
}

#[derive(Deserialize)]
pub struct QuestionRequest {
    pub question: String,
}

#[derive(Serialize)]
pub struct AnswerResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub passages: Option<Vec<Passage>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LLMMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize, Clone, Debug)]
pub struct LLMRequest {
    pub model: String,
    pub messages: Vec<LLMMessage>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

#[derive(Deserialize, Debug)]
pub struct LLMResponse {
    pub choices: Vec<LLMChoice>,
}

#[derive(Deserialize, Debug)]
pub struct LLMChoice {
    pub message: LLMMessage,
}

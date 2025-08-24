#[derive(serde::Deserialize, serde::Serialize)]
pub struct Passage {
    pub id: String,
    pub text: String,
    pub embedding: Vec<f32>,
    pub metadata: Option<Metadata>,
    pub hash: Option<i64>,
}

#[derive(serde::Deserialize, serde::Serialize)]
pub struct Metadata {
    pub title: Option<String>,
    pub source: Option<String>,
    pub date: Option<String>,
}

#[derive(serde::Deserialize)]
pub struct QuestionRequest {
    pub question: String,
}

#[derive(serde::Serialize)]
pub struct AnswerResponse {
    pub answer: String,
}

#[derive(serde::Deserialize)]
pub struct IngestRequest {
    pub text: String,
    pub metadata: Option<Metadata>,
}

#[derive(serde::Serialize)]
pub struct IngestResponse {
    pub passage_ids: Vec<String>,
}

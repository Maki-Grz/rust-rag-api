use crate::generation::*;
use crate::ingestion::*;
use crate::retrieval::*;
use crate::types::{AnswerResponse, IngestRequest, IngestResponse, QuestionRequest};
use actix_web::{post, web, HttpResponse, Responder};
use mongodb::Client;

#[post("/ingest")]
pub async fn ingest(req: web::Json<IngestRequest>, client: web::Data<Client>) -> impl Responder {
    let passages = segment_text(&req.text);
    let mut passage_ids = Vec::new();
    for mut p in passages {
        p.embedding = compute_embedding(&p);
        if let Ok(id) = store_passage(p, &client).await {
            passage_ids.push(id);
        }
    }
    web::Json(IngestResponse { passage_ids })
}

#[post("/ask")]
pub async fn ask(req: web::Json<QuestionRequest>) -> impl Responder {
    let question_embedding = compute_question_embedding(&req.question);
    let top_passages = search_top_k(&question_embedding, 5); // top-5 par défaut
    if top_passages.is_empty() {
        return HttpResponse::Ok().body("Please provide a more precise question.");
    }
    let answer = generate_answer(&req.question, &top_passages);
    HttpResponse::Ok().json(AnswerResponse { answer })
}

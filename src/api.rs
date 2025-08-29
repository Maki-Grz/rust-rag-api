use crate::generation::generate_answer;
use crate::ingestion::{segment_text, store_passage};
use crate::retrieval::search_top_k;
use crate::types::{AnswerResponse, IngestRequest, IngestResponse, Passage, QuestionRequest};
use crate::utils::compute_text_embedding;
use actix_web::{post, web, HttpResponse, Responder};
use candle_core::Device;
use candle_transformers::models::bert::BertModel;
use futures::stream::{FuturesUnordered, StreamExt};
use mongodb::Client;
use std::sync::Arc;
use tokenizers::Tokenizer;

pub struct AppState {
    pub model: BertModel,
    pub tokenizer: Tokenizer,
    pub device: Device,
}

#[post("/ingest")]
pub async fn ingest(
    state: web::Data<AppState>,
    client: web::Data<Client>,
    req: web::Json<IngestRequest>,
) -> impl Responder {
    let passages = segment_text(&req.text, req.metadata.clone());

    let tasks = FuturesUnordered::new();
    let model = Arc::new(&state.model);
    let tokenizer = Arc::new(state.tokenizer.clone());
    let device = Arc::new(state.device.clone());
    let client_arc = Arc::new(client.clone());

    for mut p in passages {
        let model = Arc::clone(&model);
        let tokenizer = Arc::clone(&tokenizer);
        let device = Arc::clone(&device);
        let client = Arc::clone(&client_arc);

        tasks.push(async move {
            let embedding = compute_text_embedding(&model, &tokenizer, &device, &p.text)
                .await
                .ok()?;

            let vec2 = embedding.to_vec2().ok()?;
            p.embedding = vec2[0].clone();

            store_passage(p, &client).await.ok()
        });
    }

    let passage_ids: Vec<String> = tasks
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .filter_map(|res| res)
        .collect();

    HttpResponse::Ok().json(IngestResponse {
        count: passage_ids.len(),
        passage_ids,
    })
}

#[post("/ask")]
pub async fn ask(
    state: web::Data<AppState>,
    client: web::Data<Client>,
    req: web::Json<QuestionRequest>,
) -> impl Responder {
    let question_embedding_tensor =
        match compute_text_embedding(&state.model, &state.tokenizer, &state.device, &req.question)
            .await
        {
            Ok(t) => t,
            Err(e) => {
                return HttpResponse::InternalServerError()
                    .json(format!("Erreur embedding question: {}", e));
            }
        };

    let question_embedding: Vec<f32> = match question_embedding_tensor.to_vec2() {
        Ok(v) => v[0].clone(),
        Err(e) => {
            return HttpResponse::InternalServerError()
                .json(format!("Erreur conversion Tensor -> Vec: {}", e));
        }
    };

    match search_top_k(&question_embedding, 3, &client).await {
        Ok(top_passages_with_scores) => {
            if top_passages_with_scores.is_empty() {
                return HttpResponse::Ok().json(AnswerResponse {
                    answer: None,
                    passages: None,
                    fallback_reason: Some(
                        "Aucun passage pertinent trouvé pour votre question.".to_string(),
                    ),
                });
            }

            let top_passages: Vec<Passage> = top_passages_with_scores.into_iter().collect();

            match generate_answer(&req.question, &top_passages).await {
                Ok(answer) => HttpResponse::Ok().json(AnswerResponse {
                    answer: Some(answer),
                    passages: None,
                    fallback_reason: None,
                }),
                Err(e) => HttpResponse::Ok().json(AnswerResponse {
                    answer: None,
                    passages: Some(top_passages),
                    fallback_reason: Some(format!(
                        "LLM non disponible ({}). Voici les passages les plus pertinents :",
                        e
                    )),
                }),
            }
        }
        Err(e) => {
            HttpResponse::InternalServerError().json(format!("Erreur lors de la recherche: {}", e))
        }
    }
}

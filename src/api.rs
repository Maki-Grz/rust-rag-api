use crate::generation::generate_answer;
use crate::ingestion::{segment_text, store_passage};
use crate::retrieval::search_top_k;
use crate::types::{AnswerResponse, IngestRequest, IngestResponse, Passage, QuestionRequest};
use crate::utils::compute_text_embedding;
use crate::AppState;
use actix_web::{post, web, HttpResponse, Responder};
use futures::stream::{FuturesUnordered, StreamExt};
use std::sync::Arc;

#[post("/ingest")]
pub async fn ingest(state: web::Data<AppState>, req: web::Json<IngestRequest>) -> impl Responder {
    let db_name = &state.config.database_name;
    let collection_name = &state.config.collection_name;
    let client = &state.db_client;

    if req.text.len() > 1_000_000 {
        return HttpResponse::BadRequest().json("Texte trop volumineux");
    }

    if req.text.trim().is_empty() {
        return HttpResponse::BadRequest().json("Texte vide");
    }

    let passages = segment_text(&req.text, req.metadata.clone(), &state.tokenizer);

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
            let embedding = match compute_text_embedding(&model, &tokenizer, &device, &p.text).await
            {
                Ok(emb) => emb,
                Err(e) => {
                    eprintln!(
                        "Impossible de calculer l'embedding pour le passage {:?}: {}",
                        &p.id, e
                    );
                    return None;
                }
            };

            let vec2 = embedding.to_vec2().ok()?;
            p.embedding = vec2[0].clone();

            store_passage(p, &client, db_name, collection_name)
                .await
                .ok()
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
pub async fn ask(state: web::Data<AppState>, req: web::Json<QuestionRequest>) -> impl Responder {
    let db_name = &state.config.database_name;
    let collection_name = &state.config.collection_name;
    let client = &state.db_client;

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

    match search_top_k(
        &question_embedding,
        3,
        client,
        db_name,
        collection_name,
        Some(2000),
    )
    .await
    {
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

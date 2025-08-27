use actix_web::{web, App, HttpServer};
use anyhow::Result;
use candle_core::Device;
use mongodb::Client;

mod api;
mod generation;
mod ingestion;
mod retrieval;
mod types;
mod utils;

use crate::utils::load_bert_model_and_tokenizer;
use api::{ask, ingest, AppState};

#[actix_web::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();
    unsafe {
        std::env::set_var("RUST_LOG", "debug,actix_web=debug");
    }
    tracing_subscriber::fmt::init();

    let device = Device::Cpu;
    let (model, tokenizer) = load_bert_model_and_tokenizer(&device)?;

    let app_state = web::Data::new(AppState {
        model,
        tokenizer,
        device,
    });

    let uri = std::env::var("COSMOS_URI").unwrap_or_else(|_| "mongodb://localhost:27017".into());

    let client = Client::with_uri_str(uri).await.expect("failed to connect");

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .app_data(web::Data::new(client.clone()))
            .service(ingest)
            .service(ask)
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await?;

    Ok(())
}

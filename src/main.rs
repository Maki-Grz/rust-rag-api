use actix_web::{web, App, HttpServer};
use anyhow::Result;
use candle_core::Device;
use candle_transformers::models::bert::BertModel;
use mongodb::bson::doc;
use mongodb::{Client, IndexModel};
use tokenizers::Tokenizer;

mod api;
mod config;
mod generation;
mod ingestion;
mod retrieval;
mod types;
mod utils;

use crate::config::Config;
use crate::types::Passage;
use crate::utils::load_bert_model_and_tokenizer;
use api::{ask, ingest};

pub struct AppState {
    pub model: BertModel,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub db_client: Client,
    pub config: Config,
}

#[actix_web::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();

    let config = match Config::from_env() {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Erreur lors du chargement de la configuration: {}", e);
            std::process::exit(1);
        }
    };

    unsafe {
        std::env::set_var("RUST_LOG", "debug,actix_web=debug");
    }
    tracing_subscriber::fmt::init();

    let device = Device::Cpu;
    let (model, tokenizer) = load_bert_model_and_tokenizer(&device)?;

    let db_client = Client::with_uri_str(&config.cosmos_uri).await?;

    let index = IndexModel::builder().keys(doc! { "hash": 1 }).build();
    db_client
        .clone()
        .database(config.database_name.as_str())
        .collection::<Passage>(config.collection_name.as_str())
        .create_index(index)
        .await?;

    let app_state = web::Data::new(AppState {
        model,
        tokenizer,
        device,
        db_client,
        config,
    });

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .service(ingest)
            .service(ask)
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await?;

    Ok(())
}

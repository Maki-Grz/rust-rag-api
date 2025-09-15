use actix_cors::Cors;
use actix_web::{web, App, HttpServer};
use anyhow::Result;
use candle_core::Device;
use candle_transformers::models::bert::BertModel;
use mongodb::bson::doc;
use mongodb::options::Compressor;
use mongodb::{options::ClientOptions, Client, IndexModel};
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
        std::env::set_var("RUST_LOG", "info,actix_web=info");
    }
    tracing_subscriber::fmt::init();

    let device = Device::Cpu;
    let (model, tokenizer) = load_bert_model_and_tokenizer(&device)?;

    let mut client_opts = ClientOptions::parse(&config.cosmos_uri).await?;
    client_opts.compressors = Some(vec![Compressor::Zstd { level: Some(1) }]);
    client_opts.max_pool_size = Some(128);
    client_opts.min_pool_size = Some(16);
    client_opts.server_selection_timeout = Some(std::time::Duration::from_secs(2));
    client_opts.app_name = Some("rag-api".into());

    let db_client = Client::with_options(client_opts)?;

    let coll = db_client
        .database(config.database_name.as_str())
        .collection::<Passage>(config.collection_name.as_str());

    let index = IndexModel::builder()
        .keys(doc! { "hash": 1 })
        .options(Some(
            mongodb::options::IndexOptions::builder()
                .unique(true)
                .build(),
        ))
        .build();
    coll.create_index(index).await?;

    let app_state = web::Data::new(AppState {
        model,
        tokenizer,
        device,
        db_client,
        config,
    });

    HttpServer::new(move || {
        let cors = Cors::permissive();

        App::new()
            .wrap(cors)
            .app_data(app_state.clone())
            .service(ingest)
            .service(ask)
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await?;

    Ok(())
}

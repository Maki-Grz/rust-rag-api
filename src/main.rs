mod api;
mod generation;
mod ingestion;
mod retrieval;
mod types;
mod utils;

use actix_web::{web, App, HttpServer};
use mongodb::Client;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let uri = std::env::var("COSMOS_URI").unwrap_or_else(|_| "mongodb://localhost:27017".into());

    let client = Client::with_uri_str(uri).await.expect("failed to connect");

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(client.clone()))
            .service(api::ingest)
            .service(api::ask)
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}

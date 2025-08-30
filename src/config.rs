use std::env;

#[derive(Clone)]
pub struct Config {
    pub database_name: String,
    pub collection_name: String,
    pub llm_uri: String,
    pub cosmos_uri: String,
}

impl Config {
    pub fn from_env() -> Result<Self, env::VarError> {
        Ok(Self {
            database_name: env::var("DATABASE")?,
            collection_name: env::var("COLLECTION")?,
            llm_uri: env::var("LLM_URI")?,
            cosmos_uri: env::var("COSMOS_URI")?,
        })
    }
}

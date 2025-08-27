use anyhow::Error as E;
use candle_core::{Device, Tensor};
use candle_transformers::models::bert::BertModel;
use tokenizers::Tokenizer;
use twox_hash::XxHash3_64;

pub async fn compute_text_embedding(
    model: &BertModel,
    tokenizer: &Tokenizer,
    device: &Device,
    text: &str,
) -> anyhow::Result<Tensor> {
    let encoding = tokenizer.encode(text, true).map_err(E::msg)?;
    let mut tokens_padded = encoding.get_ids().to_vec();
    tokens_padded.resize(512, 0);

    let token_ids_tensor = Tensor::new(&*tokens_padded, device)?.unsqueeze(0)?;
    let token_type_ids = token_ids_tensor.zeros_like()?;
    let embeddings = model.forward(&token_ids_tensor, &token_type_ids, None)?;

    let (_batch, n_tokens, _hidden) = embeddings.dims3()?;
    let pooled = (embeddings.sum(1)? / (n_tokens as f64))?;

    Ok(pooled.broadcast_div(&pooled.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

pub fn load_bert_model_and_tokenizer(device: &Device) -> Result<(BertModel, Tokenizer), E> {
    use candle_nn::VarBuilder;
    use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
    use hf_hub::{api::sync::Api, Repo, RepoType};

    let model_id = "sentence-transformers/all-MiniLM-L6-v2";
    let revision = "refs/pr/21";

    let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, revision.parse()?);
    let api = Api::new()?.repo(repo);
    let config_file = api.get("config.json")?;
    let tokenizer_file = api.get("tokenizer.json")?;
    let weights_file = api.get("model.safetensors")?;

    let config_str = std::fs::read_to_string(config_file)?;
    let mut config: Config = serde_json::from_str(&config_str)?;
    config.hidden_act = HiddenAct::GeluApproximate;

    let tokenizer = match Tokenizer::from_file(tokenizer_file) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Erreur tokenizer: {:?}", e);
            return Err(E::msg("Erreur tokenizer"));
        }
    };

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_file], DTYPE, device)? };
    let model = BertModel::load(vb, &config)?;

    Ok((model, tokenizer))
}

pub fn compute_hash(s: &str) -> u64 {
    XxHash3_64::oneshot(s.as_bytes())
}

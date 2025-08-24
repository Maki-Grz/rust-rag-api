use crate::types::Passage;

pub fn compute_question_embedding(question: &str) -> Vec<f32> {
    // TODO: calculer l'embedding de la question
    vec![]
}

pub fn search_top_k(question_embedding: &Vec<f32>, k: usize) -> Vec<Passage> {
    // TODO: récupérer les passages les plus proches depuis Cosmos DB
    vec![]
}

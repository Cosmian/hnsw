

use rust_bert::{pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType}, RustBertError};

use crate::vector::VectorItem;

pub struct Node<TC,TV> {
    pub id: usize,
    pub connections: Vec<Vec<TC>>,
    pub item: VectorItem<TV>,
    pub layer: usize,
}

impl Node<usize,f64> {
    pub fn _create(document : String) -> Result<usize,RustBertError> {
        let embedder = SentenceEmbeddingsBuilder::remote(
            SentenceEmbeddingsModelType::AllMiniLmL12V2
        ).create_model()?;
        let document_v = embedder.encode(&[document])?;
        let _ : Vec<&f32>= document_v.iter().flatten().collect();
        Ok(0)
    }
}
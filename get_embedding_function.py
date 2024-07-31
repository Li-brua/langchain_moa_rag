from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from sentence_transformers import SentenceTransformer

class EmbeddingFunctionWrapper:
    def __init__(self, model):
        self.model = model

    def embed_query(self, query):
        """
        Embed a single query.

        Args:
            query (str): The text to be embedded.

        Returns:
            list of float: The embedding for the text.
        """
        return self.model.encode([query])[0].tolist()

    def embed_documents(self, documents):
        """
        Embed a list of documents.

        Args:
            documents (list of str): The texts to be embedded.

        Returns:
            list of list of float: The embeddings for each text.
        """
        return self.model.encode(documents).tolist()

#
# # Initialize the embedding function
def embedding_function():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embedding_function = EmbeddingFunctionWrapper(model)
    return embedding_function

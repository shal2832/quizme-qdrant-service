import os
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from qdrant_client.http import models as rest
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi import HTTPException


class qdrantService:

    collectionName = 'pdf_chunks'
    
    def __init__(self):
        self.file_name = None
        self.qdrantClient = QdrantClient(
            url= os.getenv("qdrant_cluster_url"),
            api_key= os.getenv("qdrant_api_key")
        )
        self.textSplitter = RecursiveCharacterTextSplitter(
            chunk_size= 1000,
            chunk_overlap=100
        )
        self.hf_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.check_collection_exists()
        self.vector_store = QdrantVectorStore(
            client=self.qdrantClient,
            collection_name=self.collectionName,
            embedding=self.hf_embeddings
        )
        # This tells Qdrant to build a "Keyword" index for your file_id field
        self.qdrantClient.create_payload_index(
            collection_name=self.collectionName,
            field_name="metadata.file_id",
            field_schema=rest.PayloadSchemaType.KEYWORD,
        )
        

    def set_file_name(self, file_name):
        """
        Get the file name from user input and set it to class variable file_name for all context retreival"

        Args:
            file_name: name of the file uploaded by the user
        """
        self.file_name = file_name
        print(f"File name set to: {self.file_name} for context retrieval.")

    def check_collection_exists(self):
        """
        Check if the collection exists in Qdrant, if not create it.
        
        """
        existing_collection_names = self.get_collections()
        print(f"Existing collections in Qdrant: {existing_collection_names}")

        if self.collectionName not in existing_collection_names:
            self.create_collection(self.collectionName)
            print(f"Collection {self.collectionName} created successfully:")
    
    def create_collection(self,collection_name):
        """
        Create a collection in Qdrant with the specified name.

        """
        try:
            result = self.qdrantClient.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE
                )
            )
            return {"successfully created - ":result}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    async def delete_collection(self, collection_name):
        """
        Delete a collection in Qdrant with the specified name.
        
        Args:
            collection_name (str): The name of the collection to delete.
        """

        result = self.qdrantClient.delete_collection(collection_name=collection_name)
        return {"Collection {collection_name} successfully deleted. result - ":result}
    
    def get_collections(self):
        """
        Retrieve the list of existing collections in Qdrant.
        
        Returns:
            list: A list of collection names.
        """
        collections = self.qdrantClient.get_collections().collections
        return [c.name for c in collections]
    
    def query_context_retrieval(self, query : str):
        """
        Retrieve relevant context from Qdrant based on the input query.

        Args:
            query (str): The input query for which to retrieve context.

        Returns:
            str: The concatenated context retrieved from the Qdrant collection.
        """
        try:
            self.vector_store.from_existing_collection(
                embedding=self.hf_embeddings,
                collection_name=self.collectionName,
                url=os.getenv("qdrant_cluster_url"),
                api_key= os.getenv("qdrant_api_key")
            )
            print(f"Vector store initialized with collection '{self.collectionName}' for context retrieval.")
            relevant_chunks = self.vector_store.similarity_search(query, k=5)
            return ".\n".join([chunk.page_content for chunk in relevant_chunks])
        except Exception as e:
            print(f"Error: {e}, Provided collection not present to fetch the context for the query.")
    
    def entire_context_retrieval(self):
        """
        Retrieve the entire context from Qdrant collection.

        Returns:
            str: The concatenated context retrieved from the Qdrant collection.
        """
        try:
            self.vector_store.from_existing_collection(
                embedding=self.hf_embeddings,
                collection_name=self.collectionName,
                url=os.getenv("qdrant_cluster_url"),
                api_key= os.getenv("qdrant_api_key")
            )
            print(f"Vector store initialized with collection '{self.collectionName}' for entire context retrieval.")
            all_chunks = self.vector_store.similarity_search(
                            query="", 
                            k=1000,
                            filter=rest.Filter(
                                must=[
                                    rest.FieldCondition(
                                        key="metadata.file_id", # Must use the 'metadata.' prefix
                                        match=rest.MatchValue(value="test.pdf")
                                    )
                                ]
                            )
                        )  # Retrieve all chunks
            return ".\n".join([chunk.page_content for chunk in all_chunks])
        except Exception as e:
            print(f"Error: {e}, Provided collection not present to fetch the entire context.")

# Create the instance here
qdrant_service_instance = qdrantService()
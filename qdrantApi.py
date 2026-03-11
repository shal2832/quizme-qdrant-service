from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from qdrantService import qdrant_service_instance

router = APIRouter(prefix="/qdrant", tags=["qdrant"])

@router.post("/set-file")
def set_file(file_name: str):
    qdrant_service_instance.set_file_name(file_name)
    return {"message": f"File set to {file_name}"}

@router.get("/collections")
def list_collections():
    return {"collections": qdrant_service_instance.get_collections()}

@router.post("/split-documents")
def split_documents(documents: list):
    try:
        documents = qdrant_service_instance.textSplitter.create_documents(documents) 
        return {"message": "Text splitter initialized", "status_code": 200, "documents": documents}
    except Exception as e:
        return {"message": "Error initializing text splitter", "status_code": 500, "error": str(e)}
    
@router.post("/store-documents")
def add_documents(documents: list):
    try:
        qdrant_service_instance.initialize_vector_store(documents)
        return {"message": "Documents added to vector store", "status_code": 200}
    except Exception as e:
        return {"message": "Error adding documents to vector store", "status_code": 500, "error": str(e)}

@router.post("/query")
def query_context(query: str):
    context = qdrant_service_instance.query_context_retrieval(query)
    if not context:
        raise HTTPException(status_code=404, detail="No context found")
    return {"context": context}

@router.get("/entire-context")
def get_all_context():
    context = qdrant_service_instance.entire_context_retrieval()
    return {"context": context}

@router.delete("/collection/{name}")
async def delete_col(name: str):
    return await qdrant_service_instance.delete_collection(name)
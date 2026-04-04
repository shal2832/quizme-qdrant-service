from fastapi import APIRouter, HTTPException, Body
from qdrantService import qdrant_service_instance
from langchain_core.documents import Document

router = APIRouter(prefix="/qdrant", tags=["qdrant"])

@router.post("/set-file")
def set_file(file_name: str = Body(..., embed=True)):
    qdrant_service_instance.set_file_name(file_name)
    return {"message": f"File set to {file_name}"}

@router.get("/collections")
def list_collections():
    return {"collections": qdrant_service_instance.get_collections()}

@router.post("/split-documents")
def split_documents(documents: list = Body(..., embed=True)):
    try:
        input_documents = [Document(page_content=document["page_content"], metadata= document["metadata"]) for document in documents]
        response_documents = qdrant_service_instance.textSplitter.split_documents(input_documents)
        print("Splitted the recieved documents with size: ", len(response_documents)) 
        qdrant_service_instance.initialize_vector_store(response_documents)
        return {"message": "Stored the splitted documents in vector store", "status_code": 200, "documents":response_documents}
    except Exception as e:
        return {"message": "Error initializing text splitter", "status_code": 500, "error": str(e)}

@router.post("/query")
def query_context(query: str = Body(...,embed=True)):
    context = qdrant_service_instance.query_context_retrieval(query)
    if not context:
        raise HTTPException(status_code=404, detail="No context found")
    return {"context": context}

@router.get("/entire-context")
def get_all_context():
    context = qdrant_service_instance.entire_context_retrieval()
    return {"context": context}

@router.delete("/collection/{name}")
def delete_col(name: str):
    response = qdrant_service_instance.delete_collection(name)
    return {"message": f"Collection {name} deleted", "response": response}
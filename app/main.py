from fastapi import FastAPI, HTTPException
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel
from google.genai import types
import chromadb
import os
from fastapi.staticfiles import StaticFiles

load_dotenv()
historial: list[types.Content] = []

app = FastAPI()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

cliente_chroma = chromadb.PersistentClient(path="./chroma_db")

class Mensaje(BaseModel):
    pregunta: str

UMBRAL_RELEVANCIA = 0.65

def buscar_contexto(pregunta_usuario, n_resultados = 6):
    coleccion = cliente_chroma.get_collection(name="libros_historia")
    
    resultado_query = client.models.embed_content(
        model = "gemini-embedding-001",
        contents = pregunta_usuario,
        config= types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    vector_pregunta = resultado_query.embeddings[0].values

    resultados = coleccion.query(
        query_embeddings=[vector_pregunta],
        n_results=n_resultados,
        include = ["documents", "metadatas", "distances"]
    )

    documentos_filtrados = []
    metadatas_filtradas = []
    for doc, meta, distancia in zip(
        resultados["documents"][0],
        resultados["metadatas"][0],
        resultados["distances"][0]
    ):
        if distancia <= UMBRAL_RELEVANCIA:
            documentos_filtrados.append(doc)
            metadatas_filtradas.append(meta)

    return documentos_filtrados, metadatas_filtradas

@app.post("/chat")
def chat(mensaje: Mensaje):
    try:
        documentos, metadatas = buscar_contexto(mensaje.pregunta)
    except Exception as e:
        raise HTTPException(status_code=503, detail = f"Error al buscar en la base de conocimiento: {str(e)}")

    if not documentos:
        return {
            "respuesta" : "No encontré información relevante en los libros disponibles para responder tu pregunta.",
            "fuentes" : []
        }
    
    contexto_texto = ""
    for doc, meta in zip(documentos, metadatas):
        contexto_texto += f"[Fuente: {meta['libro']}]\n{doc}\n\n"

    prompt_aumentado = f"""Usa el siguiente contexto para responder la pregunta. Si el contexto no contiene la respuesta, dilo explícitamente en vez de inventar información. Cuando uses información de una fuente específica, menciona de qué libro proviene.
    
    Contexto:
    {contexto_texto}
    
    Pregunta: {mensaje.pregunta}"""

    mensajes_para_el_modelo = historial + [
        types.Content(role="user", parts=[types.Part(text=prompt_aumentado)])
    ]

    try:
        respuesta = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=mensajes_para_el_modelo
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail = f"Error al generar la respuesta: {str(e)}")


    historial.append(
        types.Content(role="user", parts=[types.Part(text=mensaje.pregunta)])
    )
    
    historial.append(
        types.Content(role="model", parts=[types.Part(text=respuesta.text)])
    )

    fuentes_unicas = list(set(meta["libro"] for meta in metadatas))

    return {
        "respuesta": respuesta.text,
        "fuentes": fuentes_unicas
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")
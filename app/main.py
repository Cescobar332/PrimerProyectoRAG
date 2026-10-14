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

def recuperar(pregunta: str, n_resultados: int = 6) -> list[dict]:
    coleccion = cliente_chroma.get_collection(name="libros_historia")
    
    resultado_query = client.models.embed_content(
        model = "gemini-embedding-001",
        contents = pregunta,
        config= types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    vector_pregunta = resultado_query.embeddings[0].values

    resultados = coleccion.query(
        query_embeddings=[vector_pregunta],
        n_results=n_resultados,
        include = ["documents", "metadatas", "distances"]
    )

    return [
        {"texto": doc, "libro": meta["libro"], "distancia": dist}
        for doc, meta, dist in zip(
            resultados["documents"][0],
            resultados["metadatas"][0],
            resultados["distances"][0],
        )
    ]

def filtrar_por_umbral(fragmentos: list[dict], umbral: float = UMBRAL_RELEVANCIA) -> list[dict]:
    return [f for f in fragmentos if f["distancia"] <= umbral]

def generar_respuesta(pregunta: str, fragmentos: list[dict], historial_previo: list) -> str:
    contexto_texto = ""
    for f in fragmentos:
        contexto_texto += f"[Fuente: {f['libro']}]\n{f['texto']}\n\n"

    prompt_aumentado = f"""Usa el siguiente contexto para responder la pregunta. Si el contexto no contiene la respuesta, dilo explícitamente en vez de inventar información. Cuando uses información de una fuente específica, menciona de qué libro proviene.
    
    Contexto:
    {contexto_texto}
    
    Pregunta: {pregunta}"""

    mensajes_para_el_modelo = historial_previo + [
        types.Content(role="user", parts=[types.Part(text=prompt_aumentado)])
    ]
    respuesta = client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents=mensajes_para_el_modelo,
    )
    return respuesta.text

@app.post("/chat")
def chat(mensaje: Mensaje):
    try:
        fragmentos = filtrar_por_umbral(recuperar(mensaje.pregunta))
    except Exception as e:
        print(f"[ERROR recuperar] {type(e).__name__}: {e}")
        raise HTTPException(status_code=503, detail=f"Error al buscar en la base de conocimiento: {e}")

    if not fragmentos:
        return {"respuesta": "No encontré información relevante en los libros disponibles para responder tu pregunta.", "fuentes": []}

    try:
        texto = generar_respuesta(mensaje.pregunta, fragmentos, historial)
    except Exception as e:
        print(f"[ERROR generar] {type(e).__name__}: {e}")
        raise HTTPException(status_code=503, detail=f"Error al generar la respuesta: {e}")

    historial.append(types.Content(role="user", parts=[types.Part(text=mensaje.pregunta)]))
    historial.append(types.Content(role="model", parts=[types.Part(text=texto)]))

    return {"respuesta": texto, "fuentes": list({f["libro"] for f in fragmentos})}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
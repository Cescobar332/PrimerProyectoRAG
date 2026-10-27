import numpy as np
from google.genai import types
from app.main import client, cliente_chroma
from normalizacion import normalizar_texto

CASOS = [
    {
        "pregunta": "¿Cuándo se fundó Coltejer y cómo logró sortear la crisis de los años treinta?",
        "id_fragmento": "empresas-y-empresarios-en-la-historia-de-colombia-carlos-davila-l-de-guevara.pdf_chunk_149",
    },
    {
        "pregunta": "¿Qué hizo Langlois?",
        "id_fragmento": "empresas-y-empresarios-en-la-historia-de-colombia-carlos-davila-l-de-guevara.pdf_chunk_329",
    },
]

def embeber(texto, task_type):
    resultado = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texto,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return np.array(resultado.embeddings[0].values)

def distancia(a, b):
    return float(np.sum((a - b) ** 2))

coleccion = cliente_chroma.get_collection(name="libros_historia")

for caso in CASOS:
    guardado = coleccion.get(ids=[caso["id_fragmento"]], include=["documents", "embeddings"])
    texto_original = guardado["documents"][0]
    vector_original = np.array(guardado["embeddings"][0])
    texto_normalizado = normalizar_texto(texto_original)
    vector_normalizado = embeber(texto_normalizado, "RETRIEVAL_DOCUMENT")
    vector_pregunta = embeber(caso["pregunta"], "RETRIEVAL_QUERY")

    d_original = distancia(vector_pregunta, vector_original)
    d_normalizado = distancia(vector_pregunta, vector_normalizado)

    print(f"\nPregunta: {caso['pregunta']}")
    print(f"Fragmento: {caso['id_fragmento']}")
    print(f"  Distancia original:    {d_original:.4f}")
    print(f"  Distancia normalizado: {d_normalizado:.4f}")
    print(f"  Diferencia:            {d_normalizado - d_original:+.4f}")
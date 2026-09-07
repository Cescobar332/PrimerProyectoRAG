from google import genai
from google.genai import types, errors
from pypdf import PdfReader
from dotenv import load_dotenv
import os
import chromadb
import time
import json
import logging

logging.disable(logging.WARNING)

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

cliente_chroma = chromadb.PersistentClient(path="./chroma_db")
coleccion = cliente_chroma.get_or_create_collection(name="libros_historia")
archivo_registro = "libros_procesados.json"

def dividir_en_chunks(texto, tamano_chunk = 1000, solapamiento = 200):
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + tamano_chunk
        chunks.append(texto[inicio:fin])
        inicio += tamano_chunk - solapamiento
    return chunks

def embed_con_reintentos(texto, task_type, intentos_maximos = 3):
    for intento in range(intentos_maximos):
        try:
            resultado = client.models.embed_content(
                model = "gemini-embedding-001",
                contents = texto,
                config=types.EmbedContentConfig(task_type=task_type)
            )
            return resultado.embeddings[0].values
        except errors.ClientError as e:
            if e.code == 429:
                espera = 15 * (intento + 1)
                print(f"Límite de cuota alcanzado, esperando {espera}s...")
                time.sleep(espera)
            else:
                raise
    return None

def cargar_registro():
    if os.path.exists(archivo_registro):
        with open(archivo_registro, "r") as f:
            return json.load(f)
    return []

def guardar_en_registro(nombre_archivo):
    registro = cargar_registro()
    registro.append(nombre_archivo)
    with open(archivo_registro, "w") as f:
        json.dump(registro, f)

libros_ya_procesados = cargar_registro()
carpeta_libros = "libros"
for nombre_archivo in os.listdir(carpeta_libros):
    if not nombre_archivo.endswith(".pdf"):
        continue

    if nombre_archivo in libros_ya_procesados:
        print(f"Omitido (ya procesado): {nombre_archivo}")
        continue

    ruta_completa = os.path.join(carpeta_libros, nombre_archivo)

    try: 
        lector = PdfReader(ruta_completa)
        texto_completo = ""
        for pagina in lector.pages:
            texto_completo += pagina.extract_text() or ""
    except Exception as e:
        print(f"No se pudo leer '{nombre_archivo}': {e}. Saltando este libro.")
        continue

    chunks = dividir_en_chunks(texto_completo)

    ids_esperados = [f"{nombre_archivo}_chunk_{i}" for i in range(len(chunks))]
    existentes = coleccion.get(ids=ids_esperados)
    ids_ya_guardados = set(existentes["ids"])

    if ids_ya_guardados:
        print(f"Retomando '{nombre_archivo}': {len(ids_ya_guardados)}/{len(chunks)} chunks ya guardados.")

    cuota_agotada = False

    for i, chunk in enumerate(chunks):
        id_chunk = f"{nombre_archivo}_chunk_{i}"
        if id_chunk in ids_ya_guardados:
            continue

        vector = embed_con_reintentos(chunk, "RETRIEVAL_DOCUMENT")
        if vector is None:
            print(f"Cuota diaria probablemente agotada. Deteniendo en '{nombre_archivo}', chunk {i}.")
            print("Vuelve a correr el script mañana para continuar.")
            cuota_agotada = True
            break

        coleccion.upsert(
            ids = [id_chunk],
            embeddings=[vector],
            documents=[chunk],
            metadatas=[{"libro": nombre_archivo}]
        )

        time.sleep(0.7)

    if cuota_agotada:
        break

    guardar_en_registro(nombre_archivo)
    print(f"Procesado: {nombre_archivo} ({len(chunks)} chunks)")
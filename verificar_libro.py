import chromadb

cliente_chroma = chromadb.PersistentClient(path="./chroma_db")
coleccion = cliente_chroma.get_collection(name="libros_historia")

nombre_libro = "carlos-pizarro-leongomez-de-guerrillero-a-candidato-presidencial-varios-autores.pdf"

resultados = coleccion.get(
    where={"libro": nombre_libro},
    include=["documents"]
)

print(f"Chunks guardados para '{nombre_libro}': {len(resultados['documents'])}")

for i, doc in enumerate(resultados["documents"]):
    if "\ufffd" in doc or doc.strip() == "":
        print(f"⚠ Chunk {i} sospechoso (posible corrupción o vacío)")

print("\n--- Muestra de los primeros 3 chunks ---")
for i, doc in enumerate(resultados["documents"][:3]):
    print(f"\nChunk {i}:")
    print(doc[:300])
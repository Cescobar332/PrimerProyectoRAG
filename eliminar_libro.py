import sys
import chromadb

if len(sys.argv) != 2:
    print('Uso: python eliminar_libro.py "<nombre-del-libro.pdf>"')
    sys.exit(1)

nombre_libro = sys.argv[1]

cliente_chroma = chromadb.PersistentClient(path="./chroma_db")
coleccion = cliente_chroma.get_collection(name="libros_historia")

encontrados = coleccion.get(where={"libro": nombre_libro}, include=[])
cantidad = len(encontrados["ids"])

if cantidad == 0:
    print(f"No hay fragmentos de '{nombre_libro}' en la base. Nada que borrar.")
    sys.exit(0)

print(f"Se van a borrar {cantidad} fragmentos de '{nombre_libro}'.")
print(f"Total en la base antes de borrar: {coleccion.count()}")
if input("Escribe 'si' para confirmar: ").strip().lower() != "si":
    print("Cancelado.")
    sys.exit(0)

coleccion.delete(where={"libro": nombre_libro})

restantes = len(coleccion.get(where={"libro": nombre_libro}, include=[])["ids"])
print(f"Listo. Fragmentos restantes de ese libro: {restantes}. Total en la base: {coleccion.count()}")
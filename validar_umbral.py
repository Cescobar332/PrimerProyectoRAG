import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
cliente_chroma = chromadb.PersistentClient(path="./chroma_db")
coleccion = cliente_chroma.get_collection(name="libros_historia")

preguntas_prueba = [
    #Relevantes claras
    "¿Qué fue el M19?",
    "¿Cómo fue el proceso de conquista del territorio colombiano?",
    "¿Qué pasó con Jorge Eliécer Gaitán?",
    "¿Cuál fue el papel de la lógica en la filosofía del siglo XII, según los textos de Abelardo y los escolásticos?",
    "¿Cómo influyó el renacimiento del siglo XII en el desarrollo de las universidades en Europa?",
    "¿Qué consecuencias tuvo la competencia entre gremios artesanales en las ciudades medievales?",
    "¿De qué manera las ideas de Adam Smith y Quesnay sobre leyes económicas contribuyeron a la comprensión del comercio y la agricultura?",
    "¿Cuál fue la relación entre la revisión de Aristóteles y la crisis en la teología cristiana en la Edad Media?",
    "¿Qué caracteriza la narrativa y la literatura del siglo XVIII, según los textos?",
    "¿Cómo explican Spengler y Toynbee los ciclos de ascenso y caída de las civilizaciones?",
    "¿Qué aportes hizo Masaccio a la pintura renacentista en términos de realismo y perspectiva?",
    "¿Qué ideas fundamentales sustentó Rousseau en 'El contrato social', y cómo influyeron en la Revolución Francesa?",
    "¿Qué cambios artísticos y culturales caracterizaron el final de la Edad Media y el Renacimiento en Europa?",

    #Tangenciales / a medio camino
    "Qué opina la gente hoy en día sobre la Guerra de los Mil Días?",
    "¿Cómo influyó la Guerra Fría en América Latina en general?",
    "¿En qué medida la desilusión por el avance científico del siglo XX afectó la política internacional?",
    "¿Qué similitudes existen entre los movimientos artísticos del siglo XIX y la literatura de la misma época?",
    "¿Cómo influye la tradición filosófica del Renacimiento en las concepciones modernas del pensamiento social?",
    "¿Qué relación tendría la teoría de la relatividad de Einstein con las ideas filosóficas del romanticismo?",
    "¿De qué manera los conceptos de poder en las ciudades medievales prepararon el terreno para las democracias modernas?",
    "¿Podría establecerse un paralelo entre los ciclos de civilización de Toynbee y las transformaciones políticas actuales en Asia?",
    "¿Qué impacto tuvo la traducción de textos árabes en la ciencia moderna, en relación con la expansión del islám en Europa?",
    "¿Cómo se relaciona la evolución de la pintura renacentista con las ideas de individualismo y humanismo en la época?",

    # Claramente ajenas
    "¿Cuál es la capital de Japón?",
    "¿Cómo se hace una tortila española?",
    "¿Cuál es la mejor manera de preparar un pastel de chocolate?",
    "¿Qué estrategias de marketing son efectivas para vender productos tecnológicos?",
    "¿Cuáles son las ventajas y desventajas de aprender un segundo idioma en la infancia?"
]

for pregunta in preguntas_prueba:
    resultado_query = client.models.embed_content(
        model="gemini-embedding-001",
        contents=pregunta,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    vector = resultado_query.embeddings[0].values

    resultados = coleccion.query(
        query_embeddings=[vector],
        n_results=3,
        include=["distances", "metadatas"]
    )

    print(f"\nPregunta: {pregunta}")
    for dist, meta in zip(resultados["distances"][0], resultados["metadatas"][0]):
        print(f"  {dist:.4f} - {meta['libro']}")
*[Read this in English](README.md)*

# RAG sobre Historia Colombiana y Latinoamericana

API construida con FastAPI que permite hacer preguntas en lenguaje natural sobre un corpus de libros de historia (principalmente colombiana y latinoamericana) y recibe respuestas generadas por un LLM, fundamentadas exclusivamente en el contenido real de esos libros — con citación de fuentes garantizada y manejo explícito de los casos donde el sistema no tiene información suficiente para responder.

Proyecto de portafolio construido para demostrar habilidades de **AI Engineering**: integración de LLMs vía API, diseño de pipelines de RAG (Retrieval-Augmented Generation), manejo de bases de datos vectoriales, y decisiones de ingeniería tomadas y validadas con datos reales, no solo copiadas de un tutorial.

> **Estado actual:** carga del corpus completada — 48 de 50 libros objetivo indexados. Consulta la sección [Estado del corpus](#estado-del-corpus) para más detalle.

---

## Por qué este proyecto

Como ingeniero de sistemas recién graduado con experiencia en desarrollo backend (Laravel, .NET), busqué un proyecto que jugara a favor de esa base — construir software, no entrenar modelos desde cero — mientras demuestra la capa de habilidades que el mercado actual de **AI Engineer** realmente pide: integrar modelos de lenguaje ya existentes en aplicaciones reales, mediante APIs, bases de datos vectoriales y arquitecturas de recuperación de información.

Elegí un corpus de historia (en vez de, por ejemplo, documentación técnica genérica) porque exige resolver problemas reales de calidad de datos: libros de cientos de páginas, PDFs con codificaciones problemáticas, archivos corruptos, y contenido multi-idioma — condiciones mucho más cercanas a un caso de uso real que un dataset de juguete ya limpio.

---

## Arquitectura

El sistema está dividido en dos programas independientes, con responsabilidades separadas:

```
┌─────────────────┐         ┌──────────────────────┐
│   ingesta.py     │         │      main.py          │
│  (offline, se     │        │   (API en producción)  │
│  corre cuando se  │        │                        │
│  agregan libros)  │        │                        │
├─────────────────┤         ├──────────────────────┤
│ 1. Lee PDFs        │        │ 1. Recibe pregunta     │
│ 2. Extrae texto     │        │    del usuario (POST)  │
│ 3. Divide en chunks │        │ 2. Embebe la pregunta  │
│ 4. Genera embeddings│        │ 3. Busca chunks         │
│ 5. Guarda en Chroma │───────▶│    relevantes en Chroma│
└─────────────────┘         │ 4. Filtra por umbral    │
                              │    de relevancia        │
                              │ 5. Arma prompt con       │
                              │    contexto + pregunta   │
                              │ 6. Llama al LLM           │
                              │ 7. Devuelve respuesta +   │
                              │    fuentes citadas        │
                              └──────────────────────┘
```

Esta separación no es arbitraria: la ingesta es un proceso pesado, lento y limitado por cuota, que se corre esporádicamente. La API de consulta necesita ser rápida y estar siempre disponible. Mezclar ambas responsabilidades en un solo programa habría acoplado innecesariamente el ciclo de vida de uno con el del otro.

---

## Stack tecnológico

| Componente | Tecnología | Por qué |
|---|---|---|
| Backend / API | **FastAPI** | Async nativo, validación automática con Pydantic, documentación interactiva generada sola. Estándar de facto en el ecosistema de aplicaciones de IA — se eligió deliberadamente sobre Django, que trae complejidad innecesaria (ORM pesado, admin panel) para una API pura. |
| Lenguaje | **Python 3.12** | Se usó 3.12 en vez de una versión más nueva (3.14) instalada en paralelo, por madurez de compatibilidad del ecosistema de librerías de IA/ML con versiones de Python recién liberadas. |
| LLM (generación) | **Google Gemini API** (`gemini-flash-lite-latest`) | Capa gratuita real, sin tarjeta de crédito, con cuota diaria suficiente para desarrollo y demostración (a diferencia de Anthropic/OpenAI, que solo ofrecen un crédito de prueba único). Se usa el alias `-latest` en vez de fijar una versión exacta, para no depender de un nombre de modelo que puede ser retirado sin aviso. |
| Embeddings | **Google Gemini API** (`gemini-embedding-001`) | Con diferenciación explícita de `task_type` (`RETRIEVAL_DOCUMENT` para el corpus, `RETRIEVAL_QUERY` para las preguntas del usuario) — una optimización asimétrica que mejora la precisión de búsqueda semántica. |
| Base de datos vectorial | **ChromaDB** (persistente, local) | Sin costo de infraestructura, adecuada para el volumen del proyecto (decenas de libros). No requiere un servicio externo. |
| Extracción de PDF | **pypdf** | Librería pura en Python, sin dependencias compiladas, licencia permisiva (BSD) — relevante para un proyecto de portafolio público en GitHub, a diferencia de alternativas más rápidas pero con licencia AGPL. |

---

## Decisiones técnicas clave (y por qué importan)

**1. La memoria conversacional y el RAG son mecanismos independientes.**
El historial de conversación (lo que el usuario y el modelo se han dicho) se mantiene separado del contexto recuperado por búsqueda semántica en cada turno. El contexto RAG se inyecta solo en la llamada puntual de cada pregunta — nunca se guarda de forma permanente en el historial. Esto evita que la conversación acumule contexto irrelevante de preguntas anteriores, lo cual encarecería y ralentizaría cada turno sucesivo sin necesidad.

**2. El umbral de relevancia semántica se calibró empíricamente, no se adivinó.**
En vez de fijar un valor arbitrario de distancia vectorial para decidir qué chunks son "suficientemente relevantes", se probó el sistema con un conjunto variado de preguntas (relevantes, tangenciales y completamente ajenas al corpus) y se midieron las distancias reales devueltas por ChromaDB. El umbral se ajustó con esos datos, y **se documenta como un parámetro que requiere recalibración conforme el corpus crece** — con más libros y más diversidad temática, el punto de corte óptimo se desplaza, y esto se validó de forma práctica durante el desarrollo (pasó de 0.7 a 0.65 tras ampliar el corpus de 6 a 15 libros).

**3. Manejo explícito de cuota de API con reintentos y reanudación granular.**
El tier gratuito de la API de embeddings impone límites por minuto y por día. El pipeline de ingesta implementa reintentos con backoff progresivo ante error 429, y — más importante — **reanudación a nivel de chunk individual, no solo a nivel de libro**: si el proceso se corta a mitad de un libro grande por agotamiento de cuota diaria, el siguiente día retoma exactamente en el chunk donde quedó, en vez de reprocesar el libro completo y volver a agotar la cuota sin avanzar.

**4. Grounding explícito contra alucinaciones.**
El prompt instruye directamente al modelo a declarar cuando el contexto no contiene la respuesta, en vez de inventar información. Adicionalmente, el campo `fuentes` de la respuesta se calcula de forma determinística desde los metadatos de ChromaDB — no depende de que el modelo mencione correctamente sus fuentes en el texto libre.

**5. Manejo de errores diferenciado.**
Fallos en la búsqueda vectorial y fallos en la generación del LLM se capturan por separado, devolviendo códigos HTTP apropiados (503) con mensajes claros, en vez de un 500 genérico. La colección de ChromaDB se obtiene de forma perezosa (dentro de la función, no al arrancar el módulo), para que el servidor pueda levantar incluso si la ingesta no se ha corrido todavía.

**6. Resiliencia ante datos de entrada defectuosos.**
La ingesta aísla la lectura de cada PDF en su propio manejo de errores: un archivo corrupto (se encontró un caso real de un PDF truncado, irreparable incluso con herramientas especializadas como `qpdf`) se registra y se salta, sin detener el procesamiento del resto del corpus.

---

## Estado del corpus

**48 libros cargados** (2 de una meta inicial de 50 quedaron descartados por corrupción del archivo — uno de los PDFs estaba truncado de forma irreparable, incluso con herramientas especializadas como `qpdf`; esto queda registrado automáticamente por el script de ingesta, no se descarta en silencio). El corpus cubre historia colombiana y latinoamericana, además de varias referencias de historia universal.

**Limitación conocida:** algunos libros incluyen resúmenes o fragmentos en otros idiomas (ej. catalán, en documentos académicos con resumen multilingüe). Los embeddings de Gemini son multilingües y manejan esto razonablemente bien, pero no se implementó filtrado por idioma — queda documentado como mejora futura, no como defecto oculto.

---

## Instalación y uso local

### Requisitos previos
- Python 3.12
- Una API key gratuita de [Google AI Studio](https://aistudio.google.com/)

### Setup

```powershell
# Clonar el repositorio
git clone <url-del-repo>
cd mi-proyecto-rag

# Crear y activar entorno virtual
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
# Crea un archivo .env con:
# GEMINI_API_KEY=tu_key_aqui
```

### Cargar el corpus

Coloca los PDFs en la carpeta `libros/` y corre:

```powershell
python ingesta.py
```

El script es seguro de interrumpir y reanudar — respeta la cuota diaria gratuita de la API y retoma automáticamente donde quedó.

### Levantar la API

```powershell
uvicorn app.main:app --reload
```

Documentación interactiva disponible en `http://127.0.0.1:8000/docs`.

### Ejemplo de uso

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "¿Qué fue el M-19?"}'
```

```json
{
  "respuesta": "El M-19 fue un movimiento guerrillero colombiano...",
  "fuentes": ["carlos-pizarro-leongomez-de-guerrillero-a-candidato-presidencial-varios-autores.pdf"]
}
```

---

## Roadmap / mejoras futuras

- [ ] Recalibrar el umbral de relevancia periódicamente si se agregan más libros
- [ ] Filtrado o detección de idioma por chunk
- [ ] Frontend mínimo (decisión pendiente)
- [ ] Sesiones de conversación por usuario (actualmente el historial es global, adecuado para demo de un solo usuario)

---

## Autor

Proyecto desarrollado por Carlos Escobar como parte de portafolio para posiciones de AI Engineer.
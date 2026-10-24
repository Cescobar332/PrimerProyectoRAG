*[Read this in English](README.md)*

# RAG sobre Historia Colombiana y Latinoamericana

API construida con FastAPI, con interfaz web, que permite hacer preguntas en lenguaje natural sobre un corpus de libros de historia (principalmente colombiana y latinoamericana) y recibe respuestas generadas por un LLM, fundamentadas exclusivamente en el contenido real de esos libros — con citación de fuentes garantizada y manejo explícito de los casos donde el sistema no tiene información suficiente para responder.

Proyecto de portafolio construido para demostrar habilidades de **AI Engineering**: integración de LLMs vía API, diseño de pipelines de RAG (Retrieval-Augmented Generation), manejo de bases de datos vectoriales, y decisiones de ingeniería tomadas y validadas con datos reales, no solo copiadas de un tutorial.

> **Estado actual:** carga del corpus completada — 48 de 50 libros objetivo indexados. Consulta la sección [Estado del corpus](#estado-del-corpus) para más detalle.

---

## Evaluación del sistema

Evaluado con un banco de 30 preguntas escritas a mano ([`eval/preguntas.jsonl`](eval/preguntas.jsonl)) y un script reproducible ([`eval/evaluar.py`](eval/evaluar.py)). El detalle por pregunta, con la respuesta generada y la justificación del juez, está en [`eval/resultados.json`](eval/resultados.json).

| Métrica | Qué mide | Resultado |
|---|---|---|
| Recuperación: libro correcto en top-k | Hit@6 antes del umbral | 18/18 (100.0%) |
| Recuperación tras umbral | Hit@6 después del umbral (0.65) | 18/18 (100.0%) |
| Responde lo que el corpus cubre | Preguntas respondibles no rechazadas | 17/18 (94.4%) |
| Responde por completo | Preguntas respondibles con respuesta completa (no parcial) | 16/18 (88.9%) |
| Rechaza lo que el corpus no cubre | Preguntas fuera del corpus con abstención | 12/12 (100.0%) |
| Fidelidad al contexto | Respuestas sin afirmaciones fuera del contexto (juez: gemini-flash-latest) | 16/17 (94.1%) |

### ¿Cómo se midió?

- **Preguntas nuevas, no las de calibración.** El umbral de relevancia (0.65) se calibró con otro conjunto de preguntas ([`validar_umbral.py`](validar_umbral.py)). Usar esas mismas preguntas para medirlo inflaría los resultados (*data leakage*), así que este banco es independiente.
- **18 preguntas respondibles y 12 negativas cercanas.** Cada respondible tiene anotado el libro donde está la respuesta. Las negativas son de historia latinoamericana pero sobre temas que el corpus no cubre (por ejemplo, la masacre de El Mozote o el Caracazo), verificado buscando esos términos en el texto indexado. Una pregunta obviamente ajena, como "¿cuál es la capital de Japón?", no pone a prueba nada.
- **Recuperación:** Hit@6, es decir, si el libro esperado aparece entre los 6 fragmentos recuperados, medido antes y después de aplicar el umbral.
- **Generación:** un LLM juez (`gemini-flash-latest`, distinto y más capaz que el generador `gemini-flash-lite-latest`) clasifica cada respuesta como completa, parcial o abstención, y decide si todas sus afirmaciones están respaldadas por el contexto recuperado. Usa salida estructurada con esquema Pydantic y `temperature=0`. Los veredictos de los casos fallidos se revisaron manualmente.

### Hallazgos

- **Una respuesta con la fuente bien citada puede afirmar lo contrario de lo que dice esa fuente (`par-01`).** Uno de los libros menciona las Convivir como ejemplo de gobiernos que llegaron a auspiciar oficialmente el paramilitarismo. El modelo escribió que las Convivir se crearon "para enfrentar el paramilitarismo", citando ese mismo libro. El campo `fuentes` era correcto y el contenido no. Por eso la fidelidad se mide aparte de la citación.
- **La extracción del PDF rompió un dato que sí estaba en el corpus (`emp-01`).** La fecha de fundación de Coltejer aparece en el libro de Carlos Dávila, pero pypdf la extrajo como `1 9 0 7`. El sistema no la encontró y respondió que no tenía la información. Es un problema de calidad de datos en la ingesta, no del modelo ni del umbral, y solo apareció al evaluar.
- **El umbral funciona con poco margen, y el prompt actúa como segunda capa.** La pregunta respondible con mayor distancia quedó en 0.587 y la negativa más cercana en 0.605. Esa negativa (sobre el gobierno de Iván Duque) pasó el umbral con sus 6 fragmentos y la rechazó el LLM, siguiendo la instrucción de decir cuando el contexto no contiene la respuesta. Las otras 11 negativas las descartó el umbral. El umbral no se reajustó con estos resultados, para no contaminar la evaluación.
- **Respuestas parciales honestas (`ind-01`).** Cuando el contexto cubre solo una parte de la pregunta, el modelo responde lo que tiene y dice explícitamente qué le falta, en vez de completar con conocimiento propio.

### Limitaciones

- **Una sola corrida con un generador no determinista.** El generador usa la temperatura por defecto y, entre corridas hechas durante el desarrollo, las respuestas a una misma pregunta cambiaron. Los resultados pueden variar en uno o dos casos entre ejecuciones.
- **Muestra pequeña.** Con 18 preguntas respondibles, cada una vale unos 5.5 puntos porcentuales. Por eso los resultados se muestran como n/total.
- **Preguntas probablemente fáciles para el retriever.** Las respondibles se escribieron a partir del texto de los libros y comparten vocabulario con él, lo que favorece el 18/18 en Hit@6. Faltan preguntas parafraseadas.
- **Juez posiblemente indulgente.** En la revisión manual se encontraron casos donde el juez aceptó como fieles inferencias leves del modelo, así que la métrica de fidelidad probablemente es optimista.

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

La interfaz web queda disponible en `http://127.0.0.1:8000/` y la documentación interactiva en `http://127.0.0.1:8000/docs`.

### Ejecutar la evaluación

Desde la raíz del proyecto:

```powershell
python -m eval.evaluar
```

Genera `eval/resultados.json` y `eval/resultados.md`. Cada pregunta evaluada se guarda en `eval/cache.jsonl`: si la ejecución se interrumpe por cuota, al volver a correrla continúa donde quedó. Si cambias el umbral, el banco de preguntas o el prompt del juez, esas entradas del caché se invalidan solas.

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
- [x] Interfaz web mínima (HTML/CSS/JS servida por FastAPI)
- [x] Evaluación formal de recuperación, abstención y fidelidad
- [ ] Mejorar la extracción de PDF (dígitos separados por espacios, como `1 9 0 7`)
- [ ] Eliminar un libro indexado dos veces (`historia-universal-contemporanea-...-14097 (1).pdf`)
- [ ] Repetir la evaluación varias veces y reportar el rango, no una sola corrida
- [ ] Ampliar el banco de evaluación con preguntas parafraseadas
- [ ] Sesiones de conversación por usuario (actualmente el historial es global, adecuado para demo de un solo usuario)

---

## Autor

Proyecto desarrollado por Carlos Escobar como parte de portafolio para posiciones de AI Engineer.
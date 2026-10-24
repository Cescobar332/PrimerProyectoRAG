import hashlib
import json
import os
import time
from datetime import datetime
from typing import Literal

from google.genai import errors, types
from pydantic import BaseModel

from app.main import client, recuperar, filtrar_por_umbral, generar_respuesta, UMBRAL_RELEVANCIA

N_RESULTADOS = 6
MODELO_JUEZ = "gemini-flash-latest"

CARPETA = os.path.dirname(__file__)
ARCHIVO_PREGUNTAS = os.path.join(CARPETA, "preguntas.jsonl")
ARCHIVO_CACHE = os.path.join(CARPETA, "cache.jsonl")
ARCHIVO_RESULTADOS = os.path.join(CARPETA, "resultados.json")
ARCHIVO_TABLA = os.path.join(CARPETA, "resultados.md")


class Veredicto(BaseModel):
    cobertura: Literal["completa", "parcial", "abstencion"]
    es_fiel: bool
    justificacion: str


PROMPT_JUEZ = """Eres un evaluador estricto de un sistema RAG. Recibes una pregunta, el contexto recuperado de una base de libros y la respuesta que generó el sistema.

Evalúa dos cosas:
1. cobertura: "completa" si la respuesta contesta todo lo que pide la pregunta. "parcial" si contesta solo una parte y declara que el contexto no alcanza para el resto. "abstencion" si no contesta nada de la pregunta y solo declara que no tiene información.
2. es_fiel: true si TODAS las afirmaciones factuales de la respuesta están respaldadas por el contexto. false si contiene al menos una afirmación que no aparece en el contexto o lo contradice, aunque sea verdadera en el mundo real. Si cobertura es "abstencion", es_fiel es true.

En justificacion explica en una o dos frases tu decisión; si es_fiel es false, cita la afirmación no respaldada.

PREGUNTA:
{pregunta}

CONTEXTO:
{contexto}

RESPUESTA DEL SISTEMA:
{respuesta}"""


def con_reintentos(funcion, intentos_maximos=5):
    # 429 = cuota agotada, 5xx = Gemini sobrecargado. Ambos son temporales.
    for intento in range(intentos_maximos):
        try:
            return funcion()
        except (errors.ClientError, errors.ServerError) as e:
            if e.code != 429 and e.code < 500:
                raise
            espera = 15 * (intento + 1)
            print(f"  Error {e.code}, esperando {espera}s...")
            time.sleep(espera)
    raise RuntimeError(f"Se agotaron los {intentos_maximos} intentos")


def juzgar(pregunta, fragmentos, respuesta) -> Veredicto:
    contexto = "\n\n".join(f"[Fuente: {f['libro']}]\n{f['texto']}" for f in fragmentos)
    resultado = client.models.generate_content(
        model=MODELO_JUEZ,
        contents=PROMPT_JUEZ.format(pregunta=pregunta, contexto=contexto, respuesta=respuesta),
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=Veredicto,
        ),
    )
    return resultado.parsed


def clave_cache(item) -> str:
    # Si cambia la pregunta, el umbral, k o el juez, el resultado guardado ya no es válido.
    firma = json.dumps([
        item["pregunta"], item["libros_esperados"], UMBRAL_RELEVANCIA, N_RESULTADOS,
        MODELO_JUEZ, PROMPT_JUEZ, Veredicto.model_json_schema(),
    ])
    return f"{item['id']}:{hashlib.sha256(firma.encode()).hexdigest()[:12]}"


def cargar_cache() -> dict:
    if not os.path.exists(ARCHIVO_CACHE):
        return {}
    with open(ARCHIVO_CACHE, encoding="utf-8") as f:
        registros = [json.loads(linea) for linea in f if linea.strip()]
    return {r["clave"]: r for r in registros}


def guardar_en_cache(registro):
    with open(ARCHIVO_CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def evaluar_pregunta(item) -> dict:
    respondible = item["tipo"] == "respondible"
    fragmentos = con_reintentos(lambda: recuperar(item["pregunta"], n_resultados=N_RESULTADOS))
    filtrados = filtrar_por_umbral(fragmentos)

    libros_crudos = {f["libro"] for f in fragmentos}
    libros_filtrados = {f["libro"] for f in filtrados}
    esperados = set(item["libros_esperados"])

    registro = {
        "id": item["id"],
        "pregunta": item["pregunta"],
        "tipo": item["tipo"],
        "libros_esperados": item["libros_esperados"],
        "recuperados": [{"libro": f["libro"], "distancia": round(f["distancia"], 4)} for f in fragmentos],
        "hit_crudo": bool(libros_crudos & esperados) if respondible else None,
        "hit_filtrado": bool(libros_filtrados & esperados) if respondible else None,
        "abstencion_por_umbral": not filtrados,
    }

    if not filtrados:
        # Mismo comportamiento que /chat: no se llama al LLM.
        registro.update(respuesta=None, cobertura="abstencion", se_abstiene=True, es_fiel=None, justificacion="Ningún fragmento pasó el umbral.")
    else:
        respuesta = con_reintentos(lambda: generar_respuesta(item["pregunta"], filtrados, []))
        veredicto = con_reintentos(lambda: juzgar(item["pregunta"], filtrados, respuesta))
        registro.update(
            respuesta=respuesta,
            cobertura=veredicto.cobertura,
            se_abstiene=veredicto.cobertura == "abstencion",
            # La fidelidad solo se mide en respuestas que realmente afirman algo.
            es_fiel=None if veredicto.cobertura == "abstencion" else veredicto.es_fiel,
            justificacion=veredicto.justificacion,
        )

    registro["abstencion_correcta"] = registro["se_abstiene"] == (not respondible)
    return registro


def proporcion(valores) -> dict:
    valores = [v for v in valores if v is not None]
    aciertos = sum(valores)
    total = len(valores)
    return {"aciertos": aciertos, "total": total, "porcentaje": round(100 * aciertos / total, 1) if total else None}


def calcular_metricas(registros) -> dict:
    respondibles = [r for r in registros if r["tipo"] == "respondible"]
    no_respondibles = [r for r in registros if r["tipo"] == "no_respondible"]
    return {
        "hit_crudo": proporcion(r["hit_crudo"] for r in respondibles),
        "hit_filtrado": proporcion(r["hit_filtrado"] for r in respondibles),
        "responde_respondibles": proporcion(not r["se_abstiene"] for r in respondibles),
        "responde_completo": proporcion(r["cobertura"] == "completa" for r in respondibles),
        "rechaza_no_respondibles": proporcion(r["se_abstiene"] for r in no_respondibles),
        "fidelidad": proporcion(r["es_fiel"] for r in registros),
    }


def formatear(m) -> str:
    if not m["total"]:
        return "n/a"
    return f"{m['aciertos']}/{m['total']} ({m['porcentaje']}%)"


def tabla_markdown(metricas) -> str:
    filas = [
        ("Recuperación: libro correcto en top-k", f"Hit@{N_RESULTADOS} antes del umbral", metricas["hit_crudo"]),
        ("Recuperación tras umbral", f"Hit@{N_RESULTADOS} después del umbral ({UMBRAL_RELEVANCIA})", metricas["hit_filtrado"]),
        ("Responde lo que el corpus cubre", "Preguntas respondibles no rechazadas", metricas["responde_respondibles"]),
        ("Responde por completo", "Preguntas respondibles con respuesta completa (no parcial)", metricas["responde_completo"]),
        ("Rechaza lo que el corpus no cubre", "Preguntas fuera del corpus con abstención", metricas["rechaza_no_respondibles"]),
        ("Fidelidad al contexto", f"Respuestas sin afirmaciones fuera del contexto (juez: {MODELO_JUEZ})", metricas["fidelidad"]),
    ]
    lineas = ["| Métrica | Qué mide | Resultado |", "|---|---|---|"]
    lineas += [f"| {nombre} | {detalle} | {formatear(m)} |" for nombre, detalle, m in filas]
    return "\n".join(lineas)


def listar_fallos(registros) -> str:
    lineas = []
    for r in registros:
        problemas = []
        if r["hit_crudo"] is False:
            problemas.append("libro esperado fuera del top-k")
        elif r["hit_filtrado"] is False:
            problemas.append("el umbral descartó el libro esperado")
        if not r["abstencion_correcta"]:
            problemas.append("respondió sin información" if r["tipo"] == "no_respondible" else "se abstuvo teniendo información")
        if r["tipo"] == "respondible" and r["cobertura"] == "parcial":
            problemas.append("respuesta parcial")
        if r["es_fiel"] is False:
            problemas.append("respuesta no fiel")
        if problemas:
            distancia_top1 = r["recuperados"][0]["distancia"] if r["recuperados"] else None
            lineas.append(f"- `{r['id']}` (distancia top-1: {distancia_top1}): {'; '.join(problemas)}")
    return "\n".join(lineas) if lineas else "- Ninguno"


def main():
    with open(ARCHIVO_PREGUNTAS, encoding="utf-8") as f:
        preguntas = [json.loads(linea) for linea in f if linea.strip()]

    cache = cargar_cache()
    registros = []
    for i, item in enumerate(preguntas, start=1):
        clave = clave_cache(item)
        if clave in cache:
            print(f"[{i}/{len(preguntas)}] {item['id']}: desde caché")
            registros.append(cache[clave])
            continue

        print(f"[{i}/{len(preguntas)}] {item['id']}: evaluando...")
        registro = evaluar_pregunta(item)
        registro["clave"] = clave
        guardar_en_cache(registro)
        registros.append(registro)

    metricas = calcular_metricas(registros)
    tabla = tabla_markdown(metricas)
    fallos = listar_fallos(registros)

    with open(ARCHIVO_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fecha": datetime.now().isoformat(timespec="seconds"),
                "configuracion": {
                    "umbral_relevancia": UMBRAL_RELEVANCIA,
                    "n_resultados": N_RESULTADOS,
                    "modelo_juez": MODELO_JUEZ,
                    "total_preguntas": len(registros),
                },
                "metricas": metricas,
                "preguntas": registros,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(ARCHIVO_TABLA, "w", encoding="utf-8") as f:
        f.write(f"{tabla}\n\n**Casos fallidos:**\n\n{fallos}\n")

    print("\n" + tabla)
    print("\nCasos fallidos:\n" + fallos)
    print(f"\nDetalle completo en {ARCHIVO_RESULTADOS}")


if __name__ == "__main__":
    main()

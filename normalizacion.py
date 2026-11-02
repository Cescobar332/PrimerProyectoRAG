import re
from itertools import groupby

CERO_COMO_LETRA = re.compile(r"(?<=\d )(\d)[oO](?=[\s.,;:)\]]|$)")
CORRIDA = re.compile(r"(?<![^\s(\[])\S(?: \S)+(?![^\s.,;:)\]\-])")
ESPACIOS_MULTIPLES = re.compile(r"(?<=\S) {2,}(?=\S)")

def _unir_corrida(coincidencia):
    texto = coincidencia.string
    inicio, fin = coincidencia.span()
    contexto_espaciado = texto[max(inicio - 2, 0):inicio] == "  " or texto [fin:fin + 2] == "  "

    partes = []
    for es_digito, grupo in groupby(coincidencia.group(0).split(" "), key=str.isdigit):
        caracteres = list(grupo)
        if (es_digito and len(caracteres) >= 3) or (contexto_espaciado and len(caracteres) >= 2):
            partes.append("".join(caracteres))
        else:
            partes.append(" ".join(caracteres))
    return " ".join(partes)

def normalizar_texto(texto: str) -> str:
    texto = CERO_COMO_LETRA.sub(r"\1 0", texto)
    texto = CORRIDA.sub(_unir_corrida, texto)
    return ESPACIOS_MULTIPLES.sub(" ", texto)


import pytest

from normalizacion import normalizar_texto

@pytest.mark.parametrize("entrada, esperado", [
    ("Coltejer de 1 9 0 7 y Fabricato", "Coltejer de 1907 y Fabricato"),
    ("fundada en 1 9 2o)", "fundada en 1920)"),
    ("e s e  g r a d u ó  e n  1 8 3 4.", "ese graduó en 1834."),
    ("H I S T O R I A  U N I V E R S A L", "HISTORIA UNIVERSAL"),
    ("(1 9 2 8- 1 9 4 7;", "(1928- 1947;"),
])
def test_une_texto_espaciado(entrada, esperado):
    assert normalizar_texto(entrada) == esperado


@pytest.mark.parametrize("texto", [
    "entre 3 o 4 grupos",
    "en 1948 y 1949",
    "entre A y B",
    "El 5 y 6 de abril",
    "de 5 años",
    "Texto normal, sin problemas.",
    "",
])
def test_no_modifica_texto_correcto(texto):
    assert normalizar_texto(texto) == texto
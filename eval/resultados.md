| Métrica | Qué mide | Resultado |
|---|---|---|
| Recuperación: libro correcto en top-k | Hit@6 antes del umbral | 18/18 (100.0%) |
| Recuperación tras umbral | Hit@6 después del umbral (0.65) | 18/18 (100.0%) |
| Responde lo que el corpus cubre | Preguntas respondibles no rechazadas | 17/18 (94.4%) |
| Responde por completo | Preguntas respondibles con respuesta completa (no parcial) | 15/18 (83.3%) |
| Rechaza lo que el corpus no cubre | Preguntas fuera del corpus con abstención | 12/12 (100.0%) |
| Fidelidad al contexto | Respuestas sin afirmaciones fuera del contexto (juez: gemini-flash-latest) | 17/17 (100.0%) |

**Casos fallidos:**

- `emp-01` (distancia top-1: 0.5633): se abstuvo teniendo información
- `ind-01` (distancia top-1: 0.4803): respuesta parcial
- `con-02` (distancia top-1: 0.5594): respuesta parcial

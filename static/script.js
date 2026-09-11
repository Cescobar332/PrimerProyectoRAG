const inputPregunta = document.getElementById("input-pregunta");
const botonEnviar = document.getElementById("boton-enviar");
const mensajes = document.getElementById("mensajes");

async function enviarPregunta() {
    const pregunta = inputPregunta.value.trim();
    if (pregunta === "") return;

    agregarMensaje(pregunta, "usuario");
    inputPregunta.value = "";
    botonEnviar.disabled = true;

    try {
        const respuesta = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ pregunta: pregunta }),
        });

        if (!respuesta.ok) {
            throw new Error(`Error del servidor: ${respuesta.status}`);
        }

        const datos = await respuesta.json();
        agregarMensaje(datos.respuesta, "bot", datos.fuentes);
    } catch (error) {
        agregarMensaje(`Ocurrió un error: ${error.message}`, "bot");
    } finally {
        botonEnviar.disabled = false;
    }
}

function agregarMensaje(texto, tipo, fuentes = []) {
    const div = document.createElement("div");
    div.className = tipo === "usuario" ? "mensaje-usuario" : "mensaje-bot";

    if (tipo === "usuario") {
        div.textContent = texto;
    } else {
        const htmlConvertido = marked.parse(texto);
        div.innerHTML = DOMPurify.sanitize(htmlConvertido);
    }

    if (fuentes.length > 0) {
        const divFuentes = document.createElement("div");
        divFuentes.className = "fuentes";
        divFuentes.textContent = "Fuentes: " + fuentes.join(", ");
        div.appendChild(divFuentes);
    }

    mensajes.appendChild(div);
    mensajes.scrollTop = mensajes.scrollHeight;
}
botonEnviar.addEventListener("click", enviarPregunta);
inputPregunta.addEventListener("keypress", (evento) => {
    if (evento.key === "Enter") enviarPregunta();
});

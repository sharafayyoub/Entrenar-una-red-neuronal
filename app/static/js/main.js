function guardarEnFirebase() {
    fetch("/save", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            resultado: "Clase A",
            confianza: 0.95,
            timestamp: new Date().toISOString()
        })
    })
    .then(res => res.json())
    .then(data => {
        console.log("Respuesta del servidor:", data);
        alert("Datos guardados en Firebase ✅");
    })
    .catch(err => console.error("Error al guardar:", err));
}

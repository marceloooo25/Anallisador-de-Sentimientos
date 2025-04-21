async function analyze() {
    const tweetText = document.getElementById("tweet-input").value;
    
    if (!tweetText.trim()) {
        alert("¡Por favor ingresa un tweet!");
        return;
    }

    const startTime = performance.now();

    try {
        const response = await fetch("http://localhost:5000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tweet: tweetText })
        });

        if (!response.ok) {
            throw new Error(`Error HTTP: ${response.status}`);
        }

        const result = await response.json();
        const endTime = performance.now();
        updateUI(result.sentiment, (endTime - startTime).toFixed(2));

    } catch (error) {
        console.error("Error:", error);
        document.getElementById("results").innerHTML = `
            <p class="error">Error: ${error.message}</p>
            <p>¿El servidor está corriendo en <a href="http://localhost:5000" target="_blank">http://localhost:5000</a>?</p>
        `;
    }
}

function updateUI(sentiment, time) {
    const colors = {
        positivo: "#4CAF50",
        negativo: "#F44336",
        neutral: "#FFC107"
    };
    
    document.getElementById("results").innerHTML = `
        <h3>Resultado:</h3>
        <div class="sentiment-bar" style="
            background-color: ${colors[sentiment] || "#9E9E9E"};
            width: 100%;
            height: 30px;
            margin: 10px 0;
            border-radius: 4px;">
        </div>
        <p>Sentimiento: <strong>${sentiment}</strong></p>
        <p>Tiempo: <strong>${time} ms</strong></p>
    `;
}
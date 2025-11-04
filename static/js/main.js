const form = document.getElementById("uploadForm");
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const formData = new FormData(form);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    document.getElementById("result").innerText =
      data.error ? " " + data.error :
      `Prediction: ${data.pred_label} (Confidence: ${(data.probability*100).toFixed(2)}%)`;
  } catch (err) {
    document.getElementById("result").innerText = " Request failed: " + err;
  }
});
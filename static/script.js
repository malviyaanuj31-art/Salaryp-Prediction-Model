/**
 * static/script.js
 * ----------------
 * Client-side logic for the Salary Prediction web application.
 *
 * Sections:
 *  1. DOM references
 *  2. File drag-and-drop & selection
 *  3. Upload handler
 *  4. Train handler
 *  5. Predict handler
 *  6. UI helpers (messages, spinner, format currency)
 */

"use strict";

/* ============================================================
   1. DOM references
============================================================ */
const csvFileInput     = document.getElementById("csv-file");
const uploadArea       = document.getElementById("upload-area");
const fileNameDisplay  = document.getElementById("file-name-display");
const btnUpload        = document.getElementById("btn-upload");
const uploadMessage    = document.getElementById("upload-message");
const datasetInfo      = document.getElementById("dataset-info");
const statsGrid        = document.getElementById("stats-grid");
const previewTbody     = document.getElementById("preview-tbody");
const basicStats       = document.getElementById("basic-stats");

const btnTrain         = document.getElementById("btn-train");
const trainMessage     = document.getElementById("train-message");
const trainResults     = document.getElementById("train-results");
const metricsGrid      = document.getElementById("metrics-grid");
const equationBox      = document.getElementById("equation-box");

const chartsPlaceholder = document.getElementById("charts-placeholder");
const chartsContainer   = document.getElementById("charts-container");

const yearsInput       = document.getElementById("years-input");
const btnPredict       = document.getElementById("btn-predict");
const predictMessage   = document.getElementById("predict-message");
const predictResult    = document.getElementById("predict-result");
const resultValue      = document.getElementById("result-value");
const resultSub        = document.getElementById("result-sub");

/* ============================================================
   2. File drag-and-drop & selection
============================================================ */

csvFileInput.addEventListener("change", () => {
  if (csvFileInput.files.length > 0) {
    const name = csvFileInput.files[0].name;
    fileNameDisplay.textContent = `Selected: ${name}`;
    fileNameDisplay.classList.remove("hidden");
    btnUpload.disabled = false;
    // Reset downstream state
    hideElement(datasetInfo);
    hideElement(trainResults);
    hideElement(chartsContainer);
    showElement(chartsPlaceholder);
    hideElement(predictResult);
    btnTrain.disabled   = true;
    btnPredict.disabled = true;
    yearsInput.disabled = true;
    clearMessage(uploadMessage);
    clearMessage(trainMessage);
    clearMessage(predictMessage);
  }
});

// Drag-and-drop visual feedback
uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", () => {
  uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    // Assign to the file input so the change handler fires
    const dt = new DataTransfer();
    dt.items.add(files[0]);
    csvFileInput.files = dt.files;
    csvFileInput.dispatchEvent(new Event("change"));
  }
});

/* ============================================================
   3. Upload handler
============================================================ */

btnUpload.addEventListener("click", async () => {
  const file = csvFileInput.files[0];
  if (!file) {
    showMessage(uploadMessage, "error", "Please select a CSV file first.");
    return;
  }

  setButtonLoading(btnUpload, true, "Uploading…");
  clearMessage(uploadMessage);
  hideElement(datasetInfo);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res  = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();

    if (!data.success) {
      showMessage(uploadMessage, "error", data.error);
      return;
    }

    // Success – render dataset info
    showMessage(uploadMessage, "success",
      `✓ Dataset loaded successfully: ${data.num_records} rows, ${data.num_features} features.`);
    renderDatasetInfo(data);
    showElement(datasetInfo);
    btnTrain.disabled = false;

  } catch (err) {
    showMessage(uploadMessage, "error", "Network error. Make sure the Flask server is running.");
  } finally {
    setButtonLoading(btnUpload, false, "Upload & Preview Dataset");
  }
});

/* ============================================================
   4. Train handler
============================================================ */

btnTrain.addEventListener("click", async () => {
  setButtonLoading(btnTrain, true, "Training…");
  clearMessage(trainMessage);
  hideElement(trainResults);
  hideElement(chartsContainer);
  showElement(chartsPlaceholder);

  try {
    const res  = await fetch("/train", { method: "POST" });
    const data = await res.json();

    if (!data.success) {
      showMessage(trainMessage, "error", data.error);
      return;
    }

    // Metrics
    renderMetrics(data);
    showElement(trainResults);

    // Charts
    const scatterDiv = document.getElementById("scatter-chart");
    const avpDiv     = document.getElementById("avp-chart");

    Plotly.newPlot(scatterDiv, data.scatter_chart.data, data.scatter_chart.layout, { responsive: true });
    Plotly.newPlot(avpDiv,     data.avp_chart.data,     data.avp_chart.layout,     { responsive: true });

    hideElement(chartsPlaceholder);
    showElement(chartsContainer);

    showMessage(trainMessage, "success",
      `✓ Model trained! Training set: ${data.train_size} samples, Test set: ${data.test_size} samples.`);

    // Enable prediction
    yearsInput.disabled = false;
    btnPredict.disabled = false;

  } catch (err) {
    showMessage(trainMessage, "error", "Network error. Make sure the Flask server is running.");
  } finally {
    setButtonLoading(btnTrain, false, "⚙ Train Model");
  }
});

/* ============================================================
   5. Predict handler
============================================================ */

btnPredict.addEventListener("click", async () => {
  const raw = yearsInput.value.trim();

  if (raw === "") {
    showMessage(predictMessage, "error", "Please enter the years of experience.");
    return;
  }

  const years = parseFloat(raw);
  if (isNaN(years)) {
    showMessage(predictMessage, "error", "Please enter a valid number (e.g. 5.5).");
    return;
  }

  if (years < 0) {
    showMessage(predictMessage, "error", "Years of experience cannot be negative.");
    return;
  }

  setButtonLoading(btnPredict, true, "Predicting…");
  clearMessage(predictMessage);
  hideElement(predictResult);

  try {
    const res  = await fetch("/predict", {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({ years_experience: years }),
    });
    const data = await res.json();

    if (!data.success) {
      showMessage(predictMessage, "error", data.error);
      return;
    }

    // Show result
    resultValue.textContent = formatINR(data.predicted_salary);
    resultSub.textContent   = `For ${data.years_experience} year(s) of experience`;
    showElement(predictResult);
    clearMessage(predictMessage);

  } catch (err) {
    showMessage(predictMessage, "error", "Network error. Make sure the Flask server is running.");
  } finally {
    setButtonLoading(btnPredict, false, "💡 Predict Salary");
  }
});

// Also allow pressing Enter in the input field
yearsInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !btnPredict.disabled) btnPredict.click();
});

/* ============================================================
   6. UI helpers
============================================================ */

/** Show a styled message inside the given element */
function showMessage(el, type, text) {
  el.textContent = text;
  el.className   = `message ${type}`;
  el.classList.remove("hidden");
}

function clearMessage(el) {
  el.textContent = "";
  el.classList.add("hidden");
}

function showElement(el)  { el.classList.remove("hidden"); }
function hideElement(el)  { el.classList.add("hidden"); }

/** Replace button label with a spinner while async work runs */
function setButtonLoading(btn, loading, label) {
  if (loading) {
    btn.disabled   = true;
    btn.innerHTML  = `<span class="spinner"></span> ${label}`;
  } else {
    btn.disabled   = false;
    btn.textContent = label;
  }
}

/** Format number as Indian Rupee currency string */
function formatINR(amount) {
  return new Intl.NumberFormat("en-IN", {
    style   : "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(amount);
}

/** Render the dataset info section */
function renderDatasetInfo(data) {
  // Stats grid
  statsGrid.innerHTML = `
    <div class="stat-box">
      <div class="stat-value">${data.num_records}</div>
      <div class="stat-label">Total Records</div>
    </div>
    <div class="stat-box">
      <div class="stat-value">${data.num_features}</div>
      <div class="stat-label">Features</div>
    </div>
    <div class="stat-box">
      <div class="stat-value">${data.columns.join(", ")}</div>
      <div class="stat-label">Columns</div>
    </div>
  `;

  // Preview table
  previewTbody.innerHTML = data.preview.map((row, i) => `
    <tr>
      <td>${i + 1}</td>
      <td>${row.YearsExperience}</td>
      <td>${formatINR(row.Salary)}</td>
    </tr>
  `).join("");

  // Basic statistics
  const stats = data.stats;
  const statKeys = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"];
  const labelMap = {
    count: "Count", mean: "Mean", std: "Std Dev",
    min: "Min", "25%": "25th %ile", "50%": "Median",
    "75%": "75th %ile", max: "Max",
  };

  const cols = ["YearsExperience", "Salary"];
  basicStats.innerHTML = cols.map(col => `
    <div class="stats-col">
      <h4>${col}</h4>
      ${statKeys.map(k => {
        const val = stats[col] && stats[col][k] !== undefined
          ? (col === "Salary" ? formatINR(stats[col][k]) : parseFloat(stats[col][k]).toFixed(2))
          : "—";
        return `<div class="stats-row"><span>${labelMap[k]}</span><span>${val}</span></div>`;
      }).join("")}
    </div>
  `).join("");
}

/** Render model metrics after training */
function renderMetrics(data) {
  const m = data.metrics;
  metricsGrid.innerHTML = `
    <div class="metric-box green">
      <div class="metric-value">${m.r2}</div>
      <div class="metric-label">R² Score</div>
    </div>
    <div class="metric-box">
      <div class="metric-value">${formatINR(m.coefficient)}</div>
      <div class="metric-label">Coefficient (slope)</div>
    </div>
    <div class="metric-box">
      <div class="metric-value">${formatINR(m.intercept)}</div>
      <div class="metric-label">Intercept</div>
    </div>
    <div class="metric-box orange">
      <div class="metric-value">${formatINR(m.mae)}</div>
      <div class="metric-label">MAE</div>
    </div>
    <div class="metric-box orange">
      <div class="metric-value">${formatINR(m.mse)}</div>
      <div class="metric-label">MSE</div>
    </div>
    <div class="metric-box purple">
      <div class="metric-value">${formatINR(m.rmse)}</div>
      <div class="metric-label">RMSE</div>
    </div>
  `;

  equationBox.textContent = `📐 ${data.equation}`;
}

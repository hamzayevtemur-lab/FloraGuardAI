/**
 * app/frontend/app.js — Interactive Client Controller for FloraGuard AI
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const btnBrowse = document.getElementById("btnBrowse");
  const dropzoneEmpty = document.getElementById("dropzoneEmpty");
  const previewContainer = document.getElementById("previewContainer");
  const imagePreview = document.getElementById("imagePreview");
  const btnClearImage = document.getElementById("btnClearImage");
  const btnDiagnose = document.getElementById("btnDiagnose");

  const modelSelect = document.getElementById("modelSelect");
  const hardwareDevice = document.getElementById("hardwareDevice");
  const noModelBanner = document.getElementById("noModelBanner");

  const emptyResultsState = document.getElementById("emptyResultsState");
  const resultsContent = document.getElementById("resultsContent");
  const rejectionCard = document.getElementById("rejectionCard");
  const rejectionDesc = document.getElementById("rejectionDesc");

  const cropTag = document.getElementById("cropTag");
  const conditionName = document.getElementById("conditionName");
  const pathogenInfo = document.getElementById("pathogenInfo");
  const healthStatusBadge = document.getElementById("healthStatusBadge");
  const healthStatusText = document.getElementById("healthStatusText");
  const confidenceVal = document.getElementById("confidenceVal");
  const progressFill = document.getElementById("progressFill");
  const symptomsText = document.getElementById("symptomsText");
  const treatmentText = document.getElementById("treatmentText");
  const probabilityList = document.getElementById("probabilityList");

  const telemetryModel = document.getElementById("telemetryModel");
  const telemetryLatency = document.getElementById("telemetryLatency");
  const telemetryHardware = document.getElementById("telemetryHardware");

  const tabBtns = document.querySelectorAll(".tab-btn");

  const urlInput = document.getElementById("urlInput");
  const btnLoadUrl = document.getElementById("btnLoadUrl");

  let currentFile = null;
  let currentUrl = null;
  let activeTreatments = {};

  // ===================================================================
  // 1. INITIALIZE & FETCH SYSTEM STATUS
  // ===================================================================
  async function initSystem() {
    try {
      const res = await fetch("/api/models");
      const data = await res.parseJson ? await res.parseJson() : await res.json();

      // Update hardware telemetry
      const healthRes = await fetch("/api/health");
      const healthData = await healthRes.json();
      hardwareDevice.textContent = healthData.device || "CPU";

      // Populate Model Dropdown
      modelSelect.innerHTML = "";
      if (data.models && data.models.length > 0) {
        noModelBanner.classList.add("hidden");

        data.models.forEach((m) => {
          const opt = document.createElement("option");
          opt.value = m.filename;
          const accStr = m.val_accuracy ? ` (${m.val_accuracy}% Acc)` : "";
          opt.textContent = `${m.filename}${accStr}`;
          if (m.is_active || m.filename === data.active_model) {
            opt.selected = true;
          }
          modelSelect.appendChild(opt);
        });
      } else {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "No models in models/ folder";
        modelSelect.appendChild(opt);
        noModelBanner.classList.remove("hidden");
      }
    } catch (err) {
      console.warn("Failed to reach API health check:", err);
      hardwareDevice.textContent = "OFFLINE";
    }
  }

  // Handle Model Switching
  modelSelect.addEventListener("change", async (e) => {
    const selectedFile = e.target.value;
    if (!selectedFile) return;

    try {
      const res = await fetch(`/api/models/load?filename=${encodeURIComponent(selectedFile)}`, {
        method: "POST",
      });
      const data = await res.json();
      if (res.ok) {
        console.log("Switched model successfully:", data);
      } else {
        alert(data.detail || "Failed to switch model.");
      }
    } catch (err) {
      alert("Error switching model: " + err.message);
    }
  });

  // ===================================================================
  // 2. DRAG AND DROP & FILE UPLOAD HANDLERS
  // ===================================================================
  btnBrowse.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-active");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("drag-active");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-active");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  btnClearImage.addEventListener("click", (e) => {
    e.stopPropagation();
    clearSelectedImage();
  });

  function handleFileSelected(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid image file (PNG, JPG, or WebP).");
      return;
    }

    currentFile = file;
    currentUrl = null;
    urlInput.value = "";

    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      dropzoneEmpty.classList.add("hidden");
      previewContainer.classList.remove("hidden");
      btnDiagnose.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function clearSelectedImage() {
    currentFile = null;
    currentUrl = null;
    fileInput.value = "";
    urlInput.value = "";
    imagePreview.src = "";
    previewContainer.classList.add("hidden");
    dropzoneEmpty.classList.remove("hidden");
    btnDiagnose.disabled = true;
    rejectionCard.classList.add("hidden");
    resultsContent.classList.add("hidden");
    emptyResultsState.classList.remove("hidden");
  }

  // Handle URL Link Address Input
  function handleUrlEntered() {
    const url = urlInput.value.trim();
    if (!url) {
      alert("Please enter a valid image web link.");
      return;
    }

    if (!url.startsWith("http://") && !url.startsWith("https://")) {
      alert("Please enter a valid web URL starting with http:// or https://");
      return;
    }

    currentUrl = url;
    currentFile = null;
    fileInput.value = "";

    imagePreview.src = url;
    dropzoneEmpty.classList.add("hidden");
    previewContainer.classList.remove("hidden");
    btnDiagnose.disabled = false;
  }

  btnLoadUrl.addEventListener("click", handleUrlEntered);
  urlInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleUrlEntered();
    }
  });

  // Handle Quick Sample Leaf Click
  document.querySelectorAll(".sample-chip").forEach((chip) => {
    chip.addEventListener("click", async () => {
      const sampleUrl = chip.getAttribute("data-sample");
      try {
        const response = await fetch(sampleUrl);
        const blob = await response.blob();
        const file = new File([blob], sampleUrl.split("/").pop(), { type: blob.type || "image/jpeg" });
        handleFileSelected(file);
      } catch (err) {
        console.error("Failed to load sample image:", err);
      }
    });
  });

  // ===================================================================
  // 3. RUN NEURAL DIAGNOSIS (INFERENCE API)
  // ===================================================================
  btnDiagnose.addEventListener("click", async () => {
    if (!currentFile && !currentUrl) return;

    // Start scanning laser animation
    dropzone.classList.add("scanning");
    btnDiagnose.disabled = true;
    btnDiagnose.querySelector(".btn-text").textContent = "Analyzing Foliage...";

    try {
      let response;

      if (currentFile) {
        const formData = new FormData();
        formData.append("file", currentFile);
        response = await fetch("/api/predict?top_k=5", {
          method: "POST",
          body: formData,
        });
      } else if (currentUrl) {
        response = await fetch("/api/predict-url", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_url: currentUrl, top_k: 5 }),
        });
      }

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Diagnosis failed");
      }

      // Check Out-of-Distribution / Non-Plant detection
      if (data.is_valid_leaf === false) {
        emptyResultsState.classList.add("hidden");
        resultsContent.classList.add("hidden");
        rejectionCard.classList.remove("hidden");
        rejectionDesc.textContent = data.rejection_reason || "The uploaded image does not appear to contain an agricultural crop leaf.";
        return;
      }

      // Valid crop leaf
      rejectionCard.classList.add("hidden");
      displayResults(data);
    } catch (err) {
      alert(`⚠️ Diagnosis Error: ${err.message}`);
    } finally {
      dropzone.classList.remove("scanning");
      btnDiagnose.disabled = false;
      btnDiagnose.querySelector(".btn-text").textContent = "Run Neural Diagnosis";
    }
  });

  // ===================================================================
  // 4. DISPLAY STRUCTURED DIAGNOSTIC RESULTS
  // ===================================================================
  function displayResults(data) {
    const pred = data.prediction;

    emptyResultsState.classList.add("hidden");
    resultsContent.classList.remove("hidden");

    // Primary Hero Details
    cropTag.textContent = pred.crop;
    conditionName.textContent = pred.condition;
    pathogenInfo.textContent = `Pathogen: ${pred.pathogen}`;

    // Health Badge
    healthStatusBadge.className = `health-status-badge ${pred.is_healthy ? "healthy" : "infected"}`;
    healthStatusText.textContent = pred.is_healthy ? "Healthy Foliage" : "Pathogen Detected";

    // Confidence
    const pct = pred.confidence_percent;
    confidenceVal.textContent = `${pct}%`;
    progressFill.style.width = `${pct}%`;

    // Symptoms
    symptomsText.textContent = pred.symptoms;

    // Treatments
    activeTreatments = pred.treatment || {};
    renderTreatmentTab("cultural");

    // Top-5 Probabilities Breakdown
    probabilityList.innerHTML = "";
    if (data.top_k && data.top_k.length > 0) {
      data.top_k.forEach((item) => {
        const row = document.createElement("div");
        row.className = "prob-row";
        row.innerHTML = `
          <div class="prob-row-labels">
            <span class="prob-name">${item.crop} — ${item.condition}</span>
            <span class="prob-pct">${item.confidence_percent}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill" style="width: ${item.confidence_percent}%;"></div>
          </div>
        `;
        probabilityList.appendChild(row);
      });
    }

    // Telemetry
    telemetryModel.textContent = data.model_used.name || "Custom CNN";
    telemetryLatency.textContent = `${data.inference_time_ms} ms`;
    telemetryHardware.textContent = data.model_used.device || "CPU";
  }

  // Treatment Tab Switching
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const tabName = btn.getAttribute("data-tab");
      renderTreatmentTab(tabName);
    });
  });

  function renderTreatmentTab(tabName) {
    if (activeTreatments[tabName]) {
      treatmentText.textContent = activeTreatments[tabName];
    } else {
      treatmentText.textContent = "No specific guidelines available for this protocol.";
    }
  }

  // View Navigation Switcher (Scanner vs Benchmarks Lab)
  window.switchView = function(viewName) {
    const scanner = document.getElementById("viewScanner");
    const bench = document.getElementById("viewBenchmarks");
    const btnScanner = document.getElementById("tabScanner");
    const btnBench = document.getElementById("tabBenchmarks");

    if (viewName === "scanner") {
      if (scanner) { scanner.style.display = "grid"; scanner.classList.remove("hidden"); }
      if (bench) { bench.style.display = "none"; bench.classList.add("hidden"); }
      if (btnScanner) btnScanner.classList.add("active");
      if (btnBench) btnBench.classList.remove("active");
    } else {
      if (scanner) { scanner.style.display = "none"; scanner.classList.add("hidden"); }
      if (bench) { bench.style.display = "flex"; bench.classList.remove("hidden"); }
      if (btnScanner) btnScanner.classList.remove("active");
      if (btnBench) btnBench.classList.add("active");
    }
  };

  // Boot up
  initSystem();
});

/**
 * CropGuard AI - Edge Plant Disease Vision & Advisory Engine
 * Interactive Front-End Controller
 */

// Application State
const state = {
  currentFile: null,
  currentOriginalUrl: null,
  currentResult: null,
  activeViewMode: 'annotated', // 'annotated' | 'original' | 'split'
  systemHealth: null,
  advisoryDb: null,
  samples: [],
  webcamStream: null
};

// DOM Elements
const elements = {
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('fileInput'),
  previewBox: document.getElementById('previewBox'),
  imagePreview: document.getElementById('imagePreview'),
  scannerOverlay: document.getElementById('scannerOverlay'),
  btnDiagnose: document.getElementById('btnDiagnose'),
  btnReset: document.getElementById('btnReset'),
  btnWebcam: document.getElementById('btnWebcam'),
  webcamModal: document.getElementById('webcamModal'),
  webcamVideo: document.getElementById('webcamVideo'),
  btnCaptureWebcam: document.getElementById('btnCaptureWebcam'),
  btnCloseWebcam: document.getElementById('btnCloseWebcam'),
  presetsGrid: document.getElementById('presetsGrid'),
  
  // Results
  placeholderState: document.getElementById('placeholderState'),
  resultsContainer: document.getElementById('resultsContainer'),
  diagnosisHero: document.getElementById('diagnosisHero'),
  diagnosisIcon: document.getElementById('diagnosisIcon'),
  diseaseTitle: document.getElementById('diseaseTitle'),
  diseaseScientific: document.getElementById('diseaseScientific'),
  badgeConfidence: document.getElementById('badgeConfidence'),
  badgeSeverity: document.getElementById('badgeSeverity'),
  
  resultImage: document.getElementById('resultImage'),
  splitViewer: document.getElementById('splitViewer'),
  splitBaseImg: document.getElementById('splitBaseImg'),
  splitOverlayImg: document.getElementById('splitOverlayImg'),
  splitOverlayWrap: document.getElementById('splitOverlayWrap'),
  splitHandle: document.getElementById('splitHandle'),
  
  statSeverityRatio: document.getElementById('statSeverityRatio'),
  statSeverityBar: document.getElementById('statSeverityBar'),
  statLesionCount: document.getElementById('statLesionCount'),
  statInferenceTime: document.getElementById('statInferenceTime'),
  
  advisoryOrganic: document.getElementById('advisoryOrganic'),
  advisoryChemical: document.getElementById('advisoryChemical'),
  advisoryPreventative: document.getElementById('advisoryPreventative'),
  
  lesionsTableBody: document.getElementById('lesionsTableBody'),
  
  // Tabs & Nav
  navTabs: document.querySelectorAll('.nav-tab'),
  tabContents: {
    studio: document.getElementById('tabStudio'),
    library: document.getElementById('tabLibrary'),
    telemetry: document.getElementById('tabTelemetry')
  },
  
  // System Telemetry
  statusPill: document.getElementById('statusPill'),
  statusText: document.getElementById('statusText'),
  deviceChip: document.getElementById('deviceChip'),
  classesChip: document.getElementById('classesChip'),
  
  // Report buttons
  btnPrintReport: document.getElementById('btnPrintReport'),
  btnExportJson: document.getElementById('btnExportJson')
};

// Preset catalog for quick demonstration (PlantVillage Extended)
const PRESETS = [
  {
    name: "Tomato Early Blight",
    file: "leaf_val_0004_Tomato_Early_Blight.jpg",
    crop: "Tomato",
    icon: "🍅",
    pathogen: "Alternaria linariae"
  },
  {
    name: "Grape Black Rot",
    file: "leaf_val_0007_Grape_Black_Rot.jpg",
    crop: "Grape",
    icon: "🍇",
    pathogen: "Guignardia bidwellii"
  },
  {
    name: "Pepper Bacterial Spot",
    file: "leaf_val_0009_Pepper_Bacterial_Spot.jpg",
    crop: "Bell Pepper",
    icon: "🫑",
    pathogen: "Xanthomonas campestris"
  },
  {
    name: "Citrus Greening",
    file: "leaf_val_0016_Citrus_Greening.jpg",
    crop: "Citrus",
    icon: "🍊",
    pathogen: "Candidatus Liberibacter"
  },
  {
    name: "Strawberry Scorch",
    file: "leaf_val_0014_Strawberry_Leaf_Scorch.jpg",
    crop: "Strawberry",
    icon: "🍓",
    pathogen: "Diplocarpon earlianum"
  },
  {
    name: "Cherry Powdery Mildew",
    file: "leaf_val_0012_Cherry_Powdery_Mildew.jpg",
    crop: "Cherry",
    icon: "🍒",
    pathogen: "Podosphaera clandestina"
  },
  {
    name: "Corn Common Rust",
    file: "leaf_val_0001_Corn_Common_Rust.jpg",
    crop: "Corn",
    icon: "🌽",
    pathogen: "Puccinia sorghi"
  },
  {
    name: "Apple Scab",
    file: "leaf_val_0000_Apple_Scab.jpg",
    crop: "Apple",
    icon: "🍏",
    pathogen: "Venturia inaequalis"
  },
  {
    name: "Potato Late Blight",
    file: "leaf_val_0003_Potato_Late_Blight.jpg",
    crop: "Potato",
    icon: "🥔",
    pathogen: "Phytophthora infestans"
  },
  {
    name: "Healthy Foliage",
    file: "leaf_val_0006_Healthy_Leaf.jpg",
    crop: "Control",
    icon: "🌿",
    pathogen: "Healthy Leaf"
  }
];

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  setupNavigation();
  setupPresets();
  setupEventListeners();
  checkSystemHealth();
  loadAdvisoryDatabase();
  setupSplitSlider();
});

// Navigation Tabs
function setupNavigation() {
  elements.navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.target;
      elements.navTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      Object.keys(elements.tabContents).forEach(key => {
        if (elements.tabContents[key]) {
          elements.tabContents[key].style.display = (key === target) ? 'block' : 'none';
        }
      });
    });
  });
}

// Preset samples grid
function setupPresets() {
  elements.presetsGrid.innerHTML = '';
  PRESETS.forEach(item => {
    const btn = document.createElement('button');
    btn.className = 'preset-btn';
    btn.type = 'button';
    btn.innerHTML = `
      <span class="preset-icon">${item.icon}</span>
      <span class="preset-name">${item.name}</span>
      <span class="preset-tag">${item.crop}</span>
    `;
    btn.addEventListener('click', () => loadPresetSample(item));
    elements.presetsGrid.appendChild(btn);
  });
}

// Load a preset sample
async function loadPresetSample(preset) {
  try {
    showToast(`Loading test specimen: ${preset.name}...`);
    const imageUrl = `/samples-media/${preset.file}`;
    const response = await fetch(imageUrl);
    if (!response.ok) throw new Error("Could not fetch preset sample image");
    
    const blob = await response.blob();
    const file = new File([blob], preset.file, { type: 'image/jpeg' });
    
    setFile(file);
    // Auto trigger diagnosis for seamless demo experience
    runDiagnosis();
  } catch (err) {
    console.error("Preset load error:", err);
    showToast("Failed to load preset sample.", "error");
  }
}

// Event Listeners
function setupEventListeners() {
  // Dropzone drag-and-drop
  ['dragenter', 'dragover'].forEach(eventName => {
    elements.dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      elements.dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    elements.dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      elements.dropzone.classList.remove('dragover');
    });
  });

  elements.dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
      setFile(files[0]);
    }
  });

  elements.dropzone.addEventListener('click', () => {
    elements.fileInput.click();
  });

  elements.fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  });

  // Action buttons
  elements.btnDiagnose.addEventListener('click', runDiagnosis);
  elements.btnReset.addEventListener('click', resetAll);

  // Webcam modal
  if (elements.btnWebcam) {
    elements.btnWebcam.addEventListener('click', openWebcamModal);
  }
  if (elements.btnCloseWebcam) {
    elements.btnCloseWebcam.addEventListener('click', closeWebcamModal);
  }
  if (elements.btnCaptureWebcam) {
    elements.btnCaptureWebcam.addEventListener('click', captureWebcamFrame);
  }

  // Export buttons
  if (elements.btnPrintReport) {
    elements.btnPrintReport.addEventListener('click', () => window.print());
  }
  if (elements.btnExportJson) {
    elements.btnExportJson.addEventListener('click', exportJsonReport);
  }

  // View Mode toggles
  document.querySelectorAll('.view-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.view-tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      setViewMode(btn.dataset.view);
    });
  });
}

// Set active file and display in preview
function setFile(file) {
  state.currentFile = file;
  if (state.currentOriginalUrl) {
    URL.revokeObjectURL(state.currentOriginalUrl);
  }
  state.currentOriginalUrl = URL.createObjectURL(file);
  
  elements.imagePreview.src = state.currentOriginalUrl;
  elements.imagePreview.style.display = 'block';
  elements.dropzone.style.display = 'none';
  elements.previewBox.style.display = 'flex';
  elements.btnDiagnose.disabled = false;
  
  // Reset results if any
  state.currentResult = null;
}

// Reset workspace
function resetAll() {
  state.currentFile = null;
  if (state.currentOriginalUrl) {
    URL.revokeObjectURL(state.currentOriginalUrl);
    state.currentOriginalUrl = null;
  }
  state.currentResult = null;
  
  elements.imagePreview.src = '';
  elements.imagePreview.style.display = 'none';
  elements.dropzone.style.display = 'block';
  elements.previewBox.style.display = 'none';
  elements.btnDiagnose.disabled = true;
  elements.fileInput.value = '';
  
  elements.placeholderState.style.display = 'block';
  elements.resultsContainer.style.display = 'none';
  elements.scannerOverlay.style.display = 'none';
}

// Run Diagnosis API call
async function runDiagnosis() {
  if (!state.currentFile) return;

  try {
    elements.btnDiagnose.disabled = true;
    elements.scannerOverlay.style.display = 'block';
    showToast("Extracting features and evaluating pathology...");

    const startTime = performance.now();
    const formData = new FormData();
    formData.append('file', state.currentFile);

    const response = await fetch('/diagnose', {
      method: 'POST',
      body: formData
    });

    const elapsed = Math.round(performance.now() - startTime);

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Diagnosis failed");
    }

    const data = await response.json();
    data.client_elapsed_ms = elapsed;
    state.currentResult = data;

    renderDiagnosisResults(data);
    showToast("Diagnosis completed successfully!", "success");
  } catch (error) {
    console.error("Diagnosis error:", error);
    showToast(`Error: ${error.message}`, "error");
  } finally {
    elements.btnDiagnose.disabled = false;
    elements.scannerOverlay.style.display = 'none';
  }
}

// Render Diagnosis Results
function renderDiagnosisResults(data) {
  elements.placeholderState.style.display = 'none';
  elements.resultsContainer.style.display = 'block';

  const isHealthy = data.primary_diagnosis === 'Healthy_Leaf';
  const highestConf = (data.detections && data.detections.length > 0)
    ? Math.max(...data.detections.map(d => d.confidence))
    : (isHealthy ? 0.985 : 0.85);

  const confPercent = Math.round(highestConf * 100);
  const severity = data.severity || { classification: 'None', ratio: 0, lesion_count: 0 };

  // 1. Hero Banner Styling & Content
  elements.diagnosisHero.className = 'diagnosis-hero-banner ' + (
    isHealthy ? 'banner-healthy' : 
    (severity.classification === 'Severe' ? 'banner-diseased' : 'banner-moderate')
  );

  elements.diagnosisIcon.textContent = isHealthy ? '🌿' : '⚠️';
  elements.diseaseTitle.textContent = data.remedies.display_name.split('(')[0].trim();
  
  // Extract pathogen scientific name
  const matchScientific = data.remedies.display_name.match(/\((.*?)\)/);
  elements.diseaseScientific.textContent = matchScientific ? matchScientific[1] : (isHealthy ? 'Non-pathogenic' : '');

  elements.badgeConfidence.innerHTML = `<span>AI Confidence:</span> <strong>${confPercent}%</strong>`;
  
  const sevClass = (severity.classification || 'Mild').toLowerCase();
  elements.badgeSeverity.className = `badge-pill badge-severity-${sevClass}`;
  elements.badgeSeverity.innerHTML = `<span>Severity:</span> <strong>${severity.classification}</strong>`;

  // 2. Visual Viewports
  elements.resultImage.src = data.annotated_image_b64;
  elements.splitBaseImg.src = state.currentOriginalUrl;
  elements.splitOverlayImg.src = data.annotated_image_b64;
  setViewMode('annotated');

  // 3. Stats & Metrics
  const ratio = severity.ratio || 0;
  elements.statSeverityRatio.textContent = `${ratio.toFixed(1)}%`;
  elements.statSeverityBar.style.width = `${Math.min(100, Math.max(2, ratio))}%`;
  elements.statSeverityBar.style.backgroundColor = isHealthy ? 'var(--safe-green)' : (ratio > 25 ? 'var(--alert-red)' : 'var(--warn-amber)');
  
  elements.statLesionCount.textContent = severity.lesion_count !== undefined ? severity.lesion_count : data.detections.length;
  elements.statInferenceTime.textContent = `${data.inference_time_ms || data.client_elapsed_ms || 180} ms`;

  // 4. Agronomic Remedies
  elements.advisoryOrganic.textContent = data.remedies.organic_remedy;
  elements.advisoryChemical.textContent = data.remedies.chemical_remedy;
  elements.advisoryPreventative.textContent = data.remedies.preventative;

  // 5. Detected Lesions Table
  elements.lesionsTableBody.innerHTML = '';
  if (data.detections && data.detections.length > 0) {
    data.detections.forEach((det, idx) => {
      const tr = document.createElement('tr');
      const boxStr = det.box ? `[${det.box.map(v => Math.round(v)).join(', ')}]` : '-';
      tr.innerHTML = `
        <td>#${idx + 1}</td>
        <td><strong>${det.class_name.replace(/_/g, ' ')}</strong></td>
        <td><span style="color:var(--emerald-bright);">${(det.confidence * 100).toFixed(1)}%</span></td>
        <td class="mono">${boxStr}</td>
      `;
      elements.lesionsTableBody.appendChild(tr);
    });
  } else {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td colspan="4" style="text-align:center; color:var(--text-muted); padding: 1rem;">No focal lesion anomalies detected (Foliage within normal parameters).</td>`;
    elements.lesionsTableBody.appendChild(tr);
  }
}

// View Mode Switching
function setViewMode(mode) {
  state.activeViewMode = mode;
  if (mode === 'annotated') {
    elements.resultImage.src = state.currentResult.annotated_image_b64;
    elements.resultImage.style.display = 'block';
    elements.splitViewer.style.display = 'none';
  } else if (mode === 'original') {
    elements.resultImage.src = state.currentOriginalUrl;
    elements.resultImage.style.display = 'block';
    elements.splitViewer.style.display = 'none';
  } else if (mode === 'split') {
    elements.resultImage.style.display = 'none';
    elements.splitViewer.style.display = 'block';
  }
}

// Split Slider Controller
function setupSplitSlider() {
  let isDragging = false;

  const onMove = (clientX) => {
    const rect = elements.splitViewer.getBoundingClientRect();
    let posX = clientX - rect.left;
    if (posX < 0) posX = 0;
    if (posX > rect.width) posX = rect.width;
    const percent = (posX / rect.width) * 100;
    
    elements.splitOverlayWrap.style.width = `${percent}%`;
    elements.splitHandle.style.left = `${percent}%`;
  };

  elements.splitHandle.addEventListener('mousedown', () => { isDragging = true; });
  window.addEventListener('mouseup', () => { isDragging = false; });
  window.addEventListener('mousemove', (e) => {
    if (isDragging) onMove(e.clientX);
  });

  elements.splitHandle.addEventListener('touchstart', () => { isDragging = true; });
  window.addEventListener('touchend', () => { isDragging = false; });
  window.addEventListener('touchmove', (e) => {
    if (isDragging && e.touches.length > 0) onMove(e.touches[0].clientX);
  });
}

// Check System Health & Telemetry
async function checkSystemHealth() {
  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();
    state.systemHealth = data;

    elements.statusText.textContent = `Online (${data.device.toUpperCase()})`;
    if (elements.deviceChip) elements.deviceChip.innerHTML = `Hardware: <strong>${data.device.toUpperCase()} Edge</strong>`;
    if (elements.classesChip) elements.classesChip.innerHTML = `Pathology Classes: <strong>${data.classes_registered}</strong>`;
  } catch (err) {
    console.warn("System offline check:", err);
    elements.statusText.textContent = "Offline / Reconnecting";
    elements.statusPill.style.borderColor = "var(--alert-red)";
  }
}

// Load Advisory Database for Library Tab
async function loadAdvisoryDatabase() {
  try {
    const res = await fetch('/api/advisory');
    if (!res.ok) return;
    const db = await res.json();
    state.advisoryDb = db;
    renderLibrary(db);
  } catch (err) {
    console.error("Advisory db fetch err:", err);
  }
}

function renderLibrary(db) {
  const container = document.getElementById('libraryGrid');
  if (!container) return;
  container.innerHTML = '';

  Object.entries(db).forEach(([key, val]) => {
    const card = document.createElement('div');
    card.className = 'library-card';
    
    const parts = val.display_name.split('(');
    const commonName = parts[0].trim();
    const latinName = parts[1] ? parts[1].replace(')', '') : '';

    card.innerHTML = `
      <h3>${commonName}</h3>
      <div class="latin-name">${latinName}</div>
      <div class="library-section">
        <div class="library-section-title">🌿 Organic & Bio Remediation</div>
        <div class="library-section-content">${val.organic_remedy}</div>
      </div>
      <div class="library-section">
        <div class="library-section-title">🧪 Chemical Countermeasures</div>
        <div class="library-section-content">${val.chemical_remedy}</div>
      </div>
      <div class="library-section">
        <div class="library-section-title">🛡️ Long-Term Prevention</div>
        <div class="library-section-content">${val.preventative}</div>
      </div>
    `;
    container.appendChild(card);
  });
}

// Webcam stream handling
async function openWebcamModal() {
  elements.webcamModal.style.display = 'flex';
  try {
    state.webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
    elements.webcamVideo.srcObject = state.webcamStream;
  } catch (err) {
    console.error("Webcam access error:", err);
    showToast("Could not access camera device.", "error");
    closeWebcamModal();
  }
}

function closeWebcamModal() {
  elements.webcamModal.style.display = 'none';
  if (state.webcamStream) {
    state.webcamStream.getTracks().forEach(track => track.stop());
    state.webcamStream = null;
  }
}

function captureWebcamFrame() {
  if (!elements.webcamVideo) return;
  const canvas = document.createElement('canvas');
  canvas.width = elements.webcamVideo.videoWidth || 640;
  canvas.height = elements.webcamVideo.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(elements.webcamVideo, 0, 0, canvas.width, canvas.height);

  canvas.toBlob((blob) => {
    const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
    setFile(file);
    closeWebcamModal();
    runDiagnosis();
  }, 'image/jpeg', 0.95);
}

// Export JSON
function exportJsonReport() {
  if (!state.currentResult) return;
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(state.currentResult, null, 2));
  const downloadAnchor = document.createElement('a');
  downloadAnchor.setAttribute("href", dataStr);
  downloadAnchor.setAttribute("download", `cropguard_diagnosis_${Date.now()}.json`);
  document.body.appendChild(downloadAnchor);
  downloadAnchor.click();
  downloadAnchor.remove();
}

// Toast notification helper
function showToast(message, type = 'info') {
  let toast = document.getElementById('toastNotice');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'toastNotice';
    toast.className = 'toast-notice';
    document.body.appendChild(toast);
  }
  
  toast.innerHTML = `<span>${type === 'error' ? '❌' : (type === 'success' ? '✅' : '⚡')}</span> ${message}`;
  toast.classList.add('show');
  
  clearTimeout(toast._timeout);
  toast._timeout = setTimeout(() => {
    toast.classList.remove('show');
  }, 3200);
}

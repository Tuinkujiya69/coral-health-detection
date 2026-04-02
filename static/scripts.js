// ========== BUBBLE GENERATION ==========
(function createBubbles() {
    const container = document.getElementById('bubblesContainer');
    for (let i = 0; i < 20; i++) {
        const bubble = document.createElement('div');
        bubble.classList.add('bubble');
        const size = Math.random() * 8 + 3;
        const opacity = Math.random() * 0.25 + 0.05;
        bubble.style.width = size + 'px';
        bubble.style.height = size + 'px';
        bubble.style.left = Math.random() * 100 + '%';
        bubble.style.animationDuration = (Math.random() * 6 + 6) + 's';
        bubble.style.animationDelay = (Math.random() * 8) + 's';
        bubble.style.setProperty('--bubble-opacity', opacity);
        container.appendChild(bubble);
    }
})();

// ========== DOM REFERENCES ==========
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadUI = document.getElementById('uploadUI');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const clearBtn = document.getElementById('clearBtn');
const emptyState = document.getElementById('emptyState');
const loadingState = document.getElementById('loadingState');
const resultCard = document.getElementById('resultCard');
const predictionLabel = document.getElementById('predictionLabel');
const confidenceFill = document.getElementById('confidenceFill');
const confidenceValue = document.getElementById('confidenceValue');
const resetBtn = document.getElementById('resetBtn');

// ========== MOCK DATA ==========
const MOCK_RESULTS = [
    { label: "Healthy Coral", confidence: 0.947 },
    { label: "Bleached Coral", confidence: 0.823 },
    { label: "Healthy Coral", confidence: 0.891 },
    { label: "Partially Bleached", confidence: 0.756 },
];

// ========== FILE HANDLING ==========
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragging');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragging');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragging');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleUpload(file);
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleUpload(file);
});

clearBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetScanner();
});

resetBtn.addEventListener('click', () => resetScanner());

// ========== UPLOAD & PROCESS ==========
function handleUpload(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadUI.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        processImage(file);
    };
    reader.readAsDataURL(file);
}

async function processImage(file) {
    // Show scanning animation
    previewContainer.classList.add('scanning');
    emptyState.classList.add('hidden');
    loadingState.classList.remove('hidden');
    resultCard.classList.add('hidden');
    clearBtn.classList.add('hidden');

    // Try real Flask API, fallback to mock
    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) throw new Error('API unavailable');
        const data = await response.json();
        setTimeout(() => displayResults(data), 1500);
    } catch (err) {
        // Mock fallback for demo
        const mock = MOCK_RESULTS[Math.floor(Math.random() * MOCK_RESULTS.length)];
        setTimeout(() => displayResults(mock), 2200);
    }
}

// ========== DISPLAY RESULTS ==========
function displayResults(data) {
    previewContainer.classList.remove('scanning');
    loadingState.classList.add('hidden');
    clearBtn.classList.remove('hidden');

    // Set label + color
    predictionLabel.innerText = data.label;
    predictionLabel.className = ''; // reset
    
    if (data.label.includes("Not a Coral")) {
        predictionLabel.style.color = "red";
    } else {
        predictionLabel.style.color = "white";
    }
    
    if (data.label.toLowerCase().includes('healthy')) {
        predictionLabel.classList.add('healthy');
    } else if (data.label.toLowerCase().includes('bleach')) {
        predictionLabel.classList.add('destructive');
    }

    // Animate confidence bar
    const pct = (data.confidence * 100).toFixed(1);
    confidenceValue.innerText = pct + '%';
    requestAnimationFrame(() => {
        confidenceFill.style.width = (data.confidence * 100) + '%';
    });

    resultCard.classList.remove('hidden');
}

// ========== RESET ==========
function resetScanner() {
    uploadUI.classList.remove('hidden');
    previewContainer.classList.add('hidden');
    previewContainer.classList.remove('scanning');
    clearBtn.classList.add('hidden');
    resultCard.classList.add('hidden');
    loadingState.classList.add('hidden');
    emptyState.classList.remove('hidden');
    confidenceFill.style.width = '0%';
    fileInput.value = '';
}

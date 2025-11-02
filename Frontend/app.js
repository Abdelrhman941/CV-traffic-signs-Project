// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Global State Management
const state = {
    originalImage: null,
    preprocessedImage: null,
    segmentedImage: null,
    currentTab: 'preprocessing',
    autoUpdate: true,
};

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    checkAPIHealth();
    setupEventListeners();
    setupFileUpload();
    setupTabs();
    setupSliders();
    setupPreprocessingControls();
    setupSegmentationControls();
    setupClassificationControls();
}

// API Health Check
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        const data = await response.json();

        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');

        if (data.status === 'healthy') {
            statusDot.classList.add('online');
            statusText.textContent = 'API Online';
        } else {
            statusText.textContent = 'API Error';
        }
    } catch (error) {
        console.error('API health check failed:', error);
        document.getElementById('statusText').textContent = 'API Offline';
    }
}

// File Upload
function setupFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    uploadArea.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
}

async function handleFileUpload(file) {
    showLoading();

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/api/upload`, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (data.success) {
            state.originalImage = data.original_image;
            displayImage('originalPreview', data.original_image);
            displayImage('segmentationInput', data.original_image);
            displayImage('featuresInput', data.original_image);
            displayImage('classificationInput', data.original_image);

            // Auto-apply preprocessing if enabled
            if (state.autoUpdate) {
                await applyPreprocessing();
            }
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert('Error uploading image. Please try again.');
    } finally {
        hideLoading();
    }
}

// Tab Management
function setupTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;
            switchTab(tabName);
        });
    });
}

function switchTab(tabName) {
    // Update buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

    // Update content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tabName).classList.add('active');

    state.currentTab = tabName;
}

// Slider Updates
function setupSliders() {
    const sliders = [
        { id: 'resizeWidth', valueId: 'resizeWidthValue' },
        { id: 'resizeHeight', valueId: 'resizeHeightValue' },
        { id: 'noiseReduction', valueId: 'noiseReductionValue' },
        { id: 'brightnessAlpha', valueId: 'brightnessAlphaValue' },
        { id: 'brightnessBeta', valueId: 'brightnessBetaValue' },
        { id: 'contrast', valueId: 'contrastValue' },
        { id: 'blockSize', valueId: 'blockSizeValue' },
        { id: 'cParam', valueId: 'cValue' },
        { id: 'kParam', valueId: 'kValue' },
    ];

    sliders.forEach(({ id, valueId }) => {
        const slider = document.getElementById(id);
        const valueDisplay = document.getElementById(valueId);

        if (slider && valueDisplay) {
            slider.addEventListener('input', (e) => {
                valueDisplay.textContent = e.target.value;

                if (state.autoUpdate && state.originalImage) {
                    if (id.startsWith('block') || id === 'cParam' || id === 'kParam') {
                        // Segmentation parameter
                        if (state.segmentedImage || document.getElementById('segmentationSource').value === 'original' || document.getElementById('segmentationSource').value === 'preprocessed') {
                            debounce(() => applySegmentation(), 500)();
                        }
                    } else {
                        // Preprocessing parameter
                        debounce(() => applyPreprocessing(), 500)();
                    }
                }
            });
        }
    });
}

// Preprocessing Controls
function setupPreprocessingControls() {
    const autoUpdateCheckbox = document.getElementById('autoUpdate');
    const toGrayscaleCheckbox = document.getElementById('toGrayscale');
    const applyButton = document.getElementById('applyPreprocessing');
    const useNextButton = document.getElementById('usePreprocessedNext');

    autoUpdateCheckbox.addEventListener('change', (e) => {
        state.autoUpdate = e.target.checked;
    });

    toGrayscaleCheckbox.addEventListener('change', () => {
        if (state.autoUpdate && state.originalImage) {
            applyPreprocessing();
        }
    });

    applyButton.addEventListener('click', applyPreprocessing);
    useNextButton.addEventListener('click', () => {
        if (state.preprocessedImage) {
            document.getElementById('segmentationSource').value = 'preprocessed';
            displayImage('segmentationInput', state.preprocessedImage);
            switchTab('segmentation');
        } else {
            alert('Please apply preprocessing first.');
        }
    });
}

async function applyPreprocessing() {
    if (!state.originalImage) {
        alert('Please upload an image first.');
        return;
    }

    showLoading();

    try {
        const params = {
            resize_width: parseInt(document.getElementById('resizeWidth').value),
            resize_height: parseInt(document.getElementById('resizeHeight').value),
            noise_reduction: parseInt(document.getElementById('noiseReduction').value),
            brightness_alpha: parseFloat(document.getElementById('brightnessAlpha').value),
            brightness_beta: parseInt(document.getElementById('brightnessBeta').value),
            contrast: parseInt(document.getElementById('contrast').value),
            to_grayscale: document.getElementById('toGrayscale').checked,
        };

        const response = await fetch(`${API_BASE_URL}/api/preprocess`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: state.originalImage,
                params: params,
            }),
        });

        const data = await response.json();

        if (data.success) {
            state.preprocessedImage = data.processed_image;
            displayImage('preprocessedPreview', data.processed_image);
        }
    } catch (error) {
        console.error('Preprocessing error:', error);
        alert('Error applying preprocessing. Please try again.');
    } finally {
        hideLoading();
    }
}

// Segmentation Controls
function setupSegmentationControls() {
    const sourceSelect = document.getElementById('segmentationSource');
    const methodSelect = document.getElementById('segmentationMethod');
    const applyButton = document.getElementById('applySegmentation');
    const useNextButton = document.getElementById('useSegmentedNext');

    sourceSelect.addEventListener('change', () => {
        updateSegmentationInput();
    });

    methodSelect.addEventListener('change', () => {
        updateSegmentationMethodUI();
        if (state.autoUpdate) {
            applySegmentation();
        }
    });

    applyButton.addEventListener('click', applySegmentation);
    useNextButton.addEventListener('click', () => {
        if (state.segmentedImage) {
            document.getElementById('classificationSource').value = 'segmented';
            displayImage('classificationInput', state.segmentedImage);
            switchTab('classification');
        } else {
            alert('Please apply segmentation first.');
        }
    });
}

function updateSegmentationInput() {
    const source = document.getElementById('segmentationSource').value;
    let imageData;

    if (source === 'original') {
        imageData = state.originalImage;
    } else if (source === 'preprocessed') {
        imageData = state.preprocessedImage;
    }

    if (imageData) {
        displayImage('segmentationInput', imageData);
    }
}

function updateSegmentationMethodUI() {
    const method = document.getElementById('segmentationMethod').value;
    const blockSizeGroup = document.getElementById('blockSizeGroup');
    const cValueGroup = document.getElementById('cValueGroup');
    const kValueGroup = document.getElementById('kValueGroup');

    // Show/hide controls based on method
    if (method === 'otsu') {
        blockSizeGroup.style.display = 'none';
        cValueGroup.style.display = 'none';
        kValueGroup.style.display = 'none';
    } else if (method === 'adaptive_mean') {
        blockSizeGroup.style.display = 'block';
        cValueGroup.style.display = 'block';
        kValueGroup.style.display = 'none';
    } else if (method === 'chow_kaneko') {
        blockSizeGroup.style.display = 'block';
        cValueGroup.style.display = 'none';
        kValueGroup.style.display = 'none';
    } else if (method === 'cheng_jin_kuo') {
        blockSizeGroup.style.display = 'block';
        cValueGroup.style.display = 'none';
        kValueGroup.style.display = 'block';
    }
}

async function applySegmentation() {
    const source = document.getElementById('segmentationSource').value;
    let imageData;

    if (source === 'original') {
        imageData = state.originalImage;
    } else if (source === 'preprocessed') {
        imageData = state.preprocessedImage;
    }

    if (!imageData) {
        alert('Please select an image source first.');
        return;
    }

    showLoading();

    try {
        const params = {
            method: document.getElementById('segmentationMethod').value,
            block_size: parseInt(document.getElementById('blockSize').value),
            c: parseInt(document.getElementById('cParam').value),
            k: parseFloat(document.getElementById('kParam').value),
        };

        const response = await fetch(`${API_BASE_URL}/api/segment`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData,
                params: params,
            }),
        });

        const data = await response.json();

        if (data.success) {
            state.segmentedImage = data.segmented_image;
            displayImage('segmentedPreview', data.segmented_image);
        }
    } catch (error) {
        console.error('Segmentation error:', error);
        alert('Error applying segmentation. Please try again.');
    } finally {
        hideLoading();
    }
}

// Classification Controls
function setupClassificationControls() {
    const sourceSelect = document.getElementById('classificationSource');
    const classifyButton = document.getElementById('classifyImage');

    sourceSelect.addEventListener('change', () => {
        updateClassificationInput();
    });

    classifyButton.addEventListener('click', classifyImage);
}

function updateClassificationInput() {
    const source = document.getElementById('classificationSource').value;
    let imageData;

    if (source === 'original') {
        imageData = state.originalImage;
    } else if (source === 'preprocessed') {
        imageData = state.preprocessedImage;
    } else if (source === 'segmented') {
        imageData = state.segmentedImage;
    }

    if (imageData) {
        displayImage('classificationInput', imageData);
    }
}

async function classifyImage() {
    const source = document.getElementById('classificationSource').value;
    let imageData;

    if (source === 'original') {
        imageData = state.originalImage;
    } else if (source === 'preprocessed') {
        imageData = state.preprocessedImage;
    } else if (source === 'segmented') {
        imageData = state.segmentedImage;
    }

    if (!imageData) {
        alert('Please select an image source first.');
        return;
    }

    showLoading();

    try {
        const response = await fetch(`${API_BASE_URL}/api/classify`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData,
            }),
        });

        const data = await response.json();

        if (data.success && data.prediction) {
            // Display main prediction
            const predictionMain = document.getElementById('predictionMain');
            const className = predictionMain.querySelector('.prediction-class');
            const confidenceFill = predictionMain.querySelector('.confidence-fill');
            const confidenceText = predictionMain.querySelector('.confidence-text');

            className.textContent = data.prediction.class_name;
            const confidencePercent = (data.prediction.confidence * 100).toFixed(2);
            confidenceFill.style.width = `${confidencePercent}%`;
            confidenceText.textContent = `${confidencePercent}%`;

            // Display top 5 predictions
            const topPredictions = document.getElementById('topPredictions');
            topPredictions.innerHTML = '';

            data.top5_predictions.forEach(pred => {
                const item = document.createElement('div');
                item.className = 'top-prediction-item';

                const name = document.createElement('span');
                name.className = 'prediction-name';
                name.textContent = pred.class_name;

                const prob = document.createElement('span');
                prob.className = 'prediction-prob';
                prob.textContent = `${(pred.confidence * 100).toFixed(2)}%`;

                item.appendChild(name);
                item.appendChild(prob);
                topPredictions.appendChild(item);
            });
        } else if (data.error) {
            alert(data.error);
        }
    } catch (error) {
        console.error('Classification error:', error);
        alert('Error classifying image. Please try again.');
    } finally {
        hideLoading();
    }
}

// Utility Functions
function displayImage(elementId, imageData) {
    const element = document.getElementById(elementId);
    if (!element) return;

    element.innerHTML = '';

    const img = document.createElement('img');
    img.src = imageData;
    img.alt = 'Processed image';

    element.appendChild(img);
}

function showLoading() {
    document.getElementById('loadingOverlay').classList.add('active');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('active');
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Event Listeners
function setupEventListeners() {
    // Smooth scrolling for navigation
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Update nav active state on scroll
    window.addEventListener('scroll', () => {
        const sections = document.querySelectorAll('section[id]');
        const scrollY = window.pageYOffset;

        sections.forEach(section => {
            const sectionHeight = section.offsetHeight;
            const sectionTop = section.offsetTop - 100;
            const sectionId = section.getAttribute('id');
            const navLink = document.querySelector(`.nav-link[href="#${sectionId}"]`);

            if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
                navLink?.classList.add('active');
            } else {
                navLink?.classList.remove('active');
            }
        });
    });
}

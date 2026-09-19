document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const uploadForm = document.getElementById('uploadForm');
    const fileUpload = document.getElementById('fileUpload');
    const dropZone = document.getElementById('dropZone');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const fileHelp = document.getElementById('fileHelp');
    const confidenceSlider = document.getElementById('confidence');
    const confValue = document.getElementById('confValue');
    const mediaTypeRadios = document.getElementsByName('mediaType');
    const submitBtn = document.getElementById('submitBtn');
    
    const loadingIndicator = document.getElementById('loadingIndicator');
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');
    const resultsContainer = document.getElementById('resultsContainer');
    const initialPlaceholder = document.getElementById('initialPlaceholder');
    
    const resultImage = document.getElementById('resultImage');
    const resultVideo = document.getElementById('resultVideo');
    const statsGrid = document.getElementById('statsGrid');

    let currentFile = null;

    // Update confidence display
    confidenceSlider.addEventListener('input', (e) => {
        confValue.textContent = e.target.value;
    });

    // Handle Media Type Change
    Array.from(mediaTypeRadios).forEach(radio => {
        radio.addEventListener('change', (e) => {
            const isVideo = e.target.value === 'video';
            fileUpload.accept = isVideo ? 'video/*' : 'image/*';
            fileHelp.textContent = isVideo ? 'MP4, AVI, MOV up to 50MB' : 'PNG, JPG, JPEG up to 10MB';
            
            // Clear current file when switching types
            currentFile = null;
            fileUpload.value = '';
            fileNameDisplay.textContent = '';
            fileNameDisplay.classList.add('hidden');
        });
    });

    // File Selection
    fileUpload.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    // Drag and Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-blue-500', 'bg-blue-50');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-blue-500', 'bg-blue-50');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-blue-500', 'bg-blue-50');
        
        if (e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    function handleFileSelection(file) {
        const mediaType = document.querySelector('input[name="mediaType"]:checked').value;
        
        if (mediaType === 'image' && !file.type.startsWith('image/')) {
            showError('Please select an image file.');
            return;
        }
        if (mediaType === 'video' && !file.type.startsWith('video/')) {
            showError('Please select a video file.');
            return;
        }

        currentFile = file;
        fileNameDisplay.textContent = file.name;
        fileNameDisplay.classList.remove('hidden');
        hideError();
    }

    // Form Submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!currentFile) {
            showError('Please upload a file first.');
            return;
        }

        const mediaType = document.querySelector('input[name="mediaType"]:checked').value;
        const confidence = confidenceSlider.value;
        
        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('confidence', confidence);

        // UI State: Loading
        showLoading();

        try {
            const endpoint = mediaType === 'image' ? '/api/detect/image' : '/api/detect/video';
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'An error occurred during inference.');
            }

            const data = await response.json();
            displayResults(data, mediaType);
            
        } catch (error) {
            showError(error.message);
            hideLoading();
        }
    });

    function displayResults(data, mediaType) {
        hideLoading();
        initialPlaceholder.classList.add('hidden');
        resultsContainer.classList.remove('hidden');
        
        // Setup Media
        if (mediaType === 'image') {
            resultVideo.classList.add('hidden');
            resultVideo.pause();
            
            resultImage.src = data.image;
            resultImage.classList.remove('hidden');
        } else {
            resultImage.classList.add('hidden');
            
            // Add a timestamp to bypass cache
            resultVideo.src = data.video_url + '?t=' + new Date().getTime();
            resultVideo.classList.remove('hidden');
            resultVideo.play();
        }

        // Setup Stats
        statsGrid.innerHTML = '';
        
        if (mediaType === 'image') {
            addStatCard('Birds Detected', data.stats.bird, 'fa-crow', 'text-blue-500');
            addStatCard('Drones Detected', data.stats.drone, 'fa-helicopter', 'text-green-500');
            addStatCard('Other Objects', data.stats.other, 'fa-shapes', 'text-gray-500');
            addStatCard('Inference Time', `${data.stats.inference_time_ms.toFixed(1)}ms`, 'fa-stopwatch', 'text-orange-500');
        } else {
            addStatCard('Frames with Birds', data.stats.bird_frames, 'fa-crow', 'text-blue-500');
            addStatCard('Frames with Drones', data.stats.drone_frames, 'fa-helicopter', 'text-green-500');
            addStatCard('Frames Processed', data.stats.total_frames_processed, 'fa-film', 'text-gray-500');
        }
    }

    function addStatCard(label, value, icon, iconColor) {
        const card = document.createElement('div');
        card.className = 'bg-gray-50 border rounded-lg p-4 flex flex-col items-center justify-center text-center';
        
        card.innerHTML = `
            <i class="fa-solid ${icon} text-2xl ${iconColor} mb-2"></i>
            <div class="text-2xl font-bold text-gray-800">${value}</div>
            <div class="text-xs text-gray-500 uppercase tracking-wide mt-1">${label}</div>
        `;
        
        statsGrid.appendChild(card);
    }

    function showLoading() {
        submitBtn.disabled = true;
        submitBtn.classList.add('opacity-75', 'cursor-not-allowed');
        loadingIndicator.classList.remove('hidden');
        errorMessage.classList.add('hidden');
        resultsContainer.classList.add('hidden');
        initialPlaceholder.classList.add('hidden');
    }

    function hideLoading() {
        submitBtn.disabled = false;
        submitBtn.classList.remove('opacity-75', 'cursor-not-allowed');
        loadingIndicator.classList.add('hidden');
    }

    function showError(msg) {
        errorText.textContent = msg;
        errorMessage.classList.remove('hidden');
        hideLoading();
    }

    function hideError() {
        errorMessage.classList.add('hidden');
    }
});

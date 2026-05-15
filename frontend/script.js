document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const uploadSection = document.getElementById('uploadSection');
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const fileInfoSection = document.getElementById('fileInfoSection');
    const fileName = document.getElementById('fileName');
    const fileMeta = document.getElementById('fileMeta');
    const btnPlay = document.getElementById('btnPlay');
    const playIcon = document.getElementById('playIcon');
    const btnRemove = document.getElementById('btnRemove');
    const btnAnalyze = document.getElementById('btnAnalyze');
    const loadingSection = document.getElementById('loadingSection');
    const resultsSection = document.getElementById('resultsSection');
    const resultLabel = document.getElementById('resultLabel');
    const resultEmoji = document.getElementById('resultEmoji');
    const confidenceValue = document.getElementById('confidenceValue');
    const ringFill = document.getElementById('ringFill');
    const top5Bars = document.getElementById('top5Bars');
    const btnNewAnalysis = document.getElementById('btnNewAnalysis');
    const audioPlayer = document.getElementById('audioPlayer');
    const waveformCanvas = document.getElementById('waveformCanvas');

    // Recording Elements
    const btnRecord = document.getElementById('btnRecord');
    const recordText = document.getElementById('recordText');
    const recordingIndicator = document.getElementById('recordingIndicator');
    const recordingTime = document.getElementById('recordingTime');

    // State
    let currentFile = null;
    let audioContext = null;
    let isPlaying = false;
    
    // Recording State
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;
    let recordInterval = null;
    let recordSeconds = 0;

    // Emoji Map for Categories (partial mapping for effect)
    const emojiMap = {
        'dog': '🐶', 'cat': '🐱', 'cow': '🐮', 'pig': '🐷', 'frog': '🐸',
        'rooster': '🐓', 'hen': '🐔', 'insects': '🦗', 'sheep': '🐑', 'crow': '🐦‍⬛',
        'rain': '🌧️', 'sea_waves': '🌊', 'crackling_fire': '🔥', 'crickets': '🦗',
        'chirping_birds': '🐦', 'water_drops': '💧', 'wind': '💨', 'pouring_water': '🚰',
        'thunderstorm': '⛈️', 'crying_baby': '👶', 'sneezing': '🤧', 'clapping': '👏',
        'breathing': '😮‍💨', 'coughing': '😷', 'footsteps': '👞', 'laughing': '😆',
        'brushing_teeth': '🪥', 'snoring': '😴', 'drinking_sipping': '🥤', 'door_knock': '🚪',
        'keyboard_typing': '⌨️', 'door_slam': '🚪💥', 'clock_alarm': '⏰', 'church_bells': '🔔',
        'airplane': '✈️', 'helicopter': '🚁', 'fireworks': '🎆', 'train': '🚆',
        'car_horn': '🚗', 'engine': '⚙️', 'siren': '🚨'
    };

    // Initialize Background Particles
    initParticles();

    // Event Listeners for Drag & Drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Recording Logic
    if (btnRecord) {
        btnRecord.addEventListener('click', async () => {
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        });
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                
                // Stop all tracks to release mic
                stream.getTracks().forEach(track => track.stop());
                
                try {
                    // Kaydedilen WebM verisini WAV formatına çevir (Backend soundfile hatasını önlemek için)
                    const wavBlob = await convertBlobToWav(audioBlob);
                    const audioFile = new File([wavBlob], "mikrofon_kaydi.wav", { type: 'audio/wav' });
                    handleFile(audioFile);
                } catch (error) {
                    console.error("Dönüştürme hatası:", error);
                    showToast("Ses formatı dönüştürülürken hata oluştu.");
                }
            };

            mediaRecorder.start();
            isRecording = true;
            
            // UI Updates
            btnRecord.classList.add('recording');
            recordText.textContent = "Kaydı Bitir";
            document.getElementById('micIcon').innerHTML = '<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"/>';
            recordingIndicator.classList.remove('hidden');
            
            // Timer
            recordSeconds = 0;
            recordingTime.textContent = "00:00";
            recordInterval = setInterval(() => {
                recordSeconds++;
                const mins = Math.floor(recordSeconds / 60).toString().padStart(2, '0');
                const secs = (recordSeconds % 60).toString().padStart(2, '0');
                recordingTime.textContent = `${mins}:${secs}`;
            }, 1000);

        } catch (err) {
            console.error("Mikrofon hatası:", err);
            showToast("Mikrofona erişilemedi. İzin verdiğinizden emin olun.");
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        isRecording = false;
        clearInterval(recordInterval);
        
        // Reset UI
        btnRecord.classList.remove('recording');
        recordText.textContent = "Mikrofondan Kaydet";
        document.getElementById('micIcon').innerHTML = '<path d="M12 2C10.3431 2 9 3.34315 9 5V11C9 12.6569 10.3431 14 12 14C13.6569 14 15 12.6569 15 11V5C15 3.34315 13.6569 2 12 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M19 10V11C19 14.866 15.866 18 12 18C8.13401 18 5 14.866 5 11V10" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 18V22" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M8 22H16" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>';
        recordingIndicator.classList.add('hidden');
    }

    // WebM to WAV Converter (Client Side)
    async function convertBlobToWav(blob) {
        const arrayBuffer = await blob.arrayBuffer();
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        const wavBuffer = audioBufferToWav(audioBuffer);
        return new Blob([wavBuffer], { type: 'audio/wav' });
    }

    function audioBufferToWav(buffer) {
        let numOfChan = buffer.numberOfChannels,
            length = buffer.length * numOfChan * 2 + 44,
            bufferArray = new ArrayBuffer(length),
            view = new DataView(bufferArray),
            channels = [], i, sample,
            offset = 0,
            pos = 0;

        function setUint16(data) {
            view.setUint16(pos, data, true);
            pos += 2;
        }

        function setUint32(data) {
            view.setUint32(pos, data, true);
            pos += 4;
        }

        setUint32(0x46464952); // "RIFF"
        setUint32(length - 8); // file length - 8
        setUint32(0x45564157); // "WAVE"

        setUint32(0x20746d66); // "fmt " chunk
        setUint32(16); // length = 16
        setUint16(1); // PCM (uncompressed)
        setUint16(numOfChan);
        setUint32(buffer.sampleRate);
        setUint32(buffer.sampleRate * 2 * numOfChan); // avg. bytes/sec
        setUint16(numOfChan * 2); // block-align
        setUint16(16); // 16-bit

        setUint32(0x61746164); // "data" - chunk
        setUint32(length - pos - 4); // chunk length

        for (i = 0; i < buffer.numberOfChannels; i++) {
            channels.push(buffer.getChannelData(i));
        }

        while (pos < length) {
            for (i = 0; i < numOfChan; i++) {
                sample = Math.max(-1, Math.min(1, channels[i][offset])); // clamp
                sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0; // scale to 16-bit signed int
                view.setInt16(pos, sample, true); // write 16-bit sample
                pos += 2;
            }
            offset++; // next source sample
        }
        return bufferArray;
    }

    // File Handling
    function handleFile(file) {
        if (!file.type.startsWith('audio/')) {
            showToast('Lütfen geçerli bir ses dosyası yükleyin.');
            return;
        }

        currentFile = file;
        
        // Update UI
        fileName.textContent = file.name;
        fileMeta.textContent = formatBytes(file.size);
        
        // Setup Audio Player
        const fileURL = URL.createObjectURL(file);
        audioPlayer.src = fileURL;
        
        // Draw initial fake waveform
        drawFakeWaveform(waveformCanvas);

        // Switch Sections
        uploadSection.classList.add('hidden');
        fileInfoSection.classList.remove('hidden');
        btnAnalyze.disabled = false;
    }

    // Controls
    btnRemove.addEventListener('click', () => {
        resetApp();
    });

    btnPlay.addEventListener('click', () => {
        if (isPlaying) {
            audioPlayer.pause();
            playIcon.innerHTML = '<polygon points="4,2 16,9 4,16" fill="currentColor"/>';
            isPlaying = false;
        } else {
            audioPlayer.play();
            playIcon.innerHTML = '<rect x="4" y="4" width="4" height="10" fill="currentColor"/><rect x="10" y="4" width="4" height="10" fill="currentColor"/>';
            isPlaying = true;
            animateWaveform(waveformCanvas);
        }
    });

    audioPlayer.addEventListener('ended', () => {
        playIcon.innerHTML = '<polygon points="4,2 16,9 4,16" fill="currentColor"/>';
        if (replayIcon) replayIcon.innerHTML = '<polygon points="4,2 16,9 4,16" fill="currentColor"/>';
        isPlaying = false;
    });

    // Replay on Results Section
    const btnReplayResult = document.getElementById('btnReplayResult');
    const replayIcon = document.getElementById('replayIcon');
    if (btnReplayResult) {
        btnReplayResult.addEventListener('click', () => {
            if (isPlaying) {
                audioPlayer.pause();
                replayIcon.innerHTML = '<polygon points="4,2 16,9 4,16" fill="currentColor"/>';
                isPlaying = false;
            } else {
                audioPlayer.play();
                replayIcon.innerHTML = '<rect x="4" y="4" width="4" height="10" fill="currentColor"/><rect x="10" y="4" width="4" height="10" fill="currentColor"/>';
                isPlaying = true;
            }
        });
    }

    // Analysis
    btnAnalyze.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI Transition
        fileInfoSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');
        
        if (isPlaying) {
            audioPlayer.pause();
            isPlaying = false;
        }

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Sunucu Hatası: ${response.status}`);
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            console.error('Analiz hatası:', error);
            showToast(error.message || 'Analiz sırasında bir hata oluştu.');
            resetApp();
        }
    });

    btnNewAnalysis.addEventListener('click', resetApp);

    // Display Results
    function displayResults(data) {
        loadingSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        // Primary Result
        const confPct = Math.round(data.confidence * 100);
        resultLabel.textContent = data.prediction.replace(/_/g, ' ');
        confidenceValue.textContent = `${confPct}%`;
        resultEmoji.textContent = emojiMap[data.prediction] || '🔊';

        // Animate Ring (Circumference is ~326)
        setTimeout(() => {
            const offset = 326.73 - (326.73 * confPct) / 100;
            ringFill.style.strokeDashoffset = offset;
        }, 100);

        // Top 5 Bars
        top5Bars.innerHTML = '';
        data.top5.forEach((item, index) => {
            const itemConf = Math.round(item.confidence * 100);
            const row = document.createElement('div');
            row.className = 'bar-row';
            row.innerHTML = `
                <div class="bar-labels">
                    <span class="bar-name">${item.label.replace(/_/g, ' ')}</span>
                    <span class="bar-val">${itemConf}%</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" id="bar-${index}"></div>
                </div>
            `;
            top5Bars.appendChild(row);

            // Animate bar
            setTimeout(() => {
                document.getElementById(`bar-${index}`).style.width = `${itemConf}%`;
            }, 200 + (index * 100));
        });
    }

    // Utilities
    function resetApp() {
        currentFile = null;
        fileInput.value = '';
        if (audioPlayer.src) {
            URL.revokeObjectURL(audioPlayer.src);
            audioPlayer.src = '';
        }
        isPlaying = false;
        playIcon.innerHTML = '<polygon points="4,2 16,9 4,16" fill="currentColor"/>';
        ringFill.style.strokeDashoffset = '326.73';
        
        resultsSection.classList.add('hidden');
        loadingSection.classList.add('hidden');
        fileInfoSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
    }

    function formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    function showToast(message) {
        let toast = document.getElementById('toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'toast';
            toast.className = 'toast';
            toast.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg><span>${message}</span>`;
            document.body.appendChild(toast);
        } else {
            toast.querySelector('span').textContent = message;
        }
        
        toast.classList.add('show');
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    }

    // Background Particles
    function initParticles() {
        const container = document.getElementById('bgParticles');
        const count = 30;
        for (let i = 0; i < count; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = `${Math.random() * 100}vw`;
            particle.style.animationDuration = `${15 + Math.random() * 20}s`;
            particle.style.animationDelay = `${Math.random() * 10}s`;
            particle.style.opacity = Math.random() * 0.5;
            const size = Math.random() * 4 + 2;
            particle.style.width = `${size}px`;
            particle.style.height = `${size}px`;
            container.appendChild(particle);
        }
    }

    // Simple visual waveform (fake for aesthetics)
    function drawFakeWaveform(canvas) {
        const ctx = canvas.getContext('2d');
        const width = canvas.offsetWidth;
        const height = canvas.offsetHeight;
        canvas.width = width * window.devicePixelRatio;
        canvas.height = height * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = 'rgba(139, 92, 246, 0.4)';
        
        const bars = 60;
        const barWidth = width / bars - 2;
        
        for (let i = 0; i < bars; i++) {
            const h = Math.random() * (height * 0.8);
            const x = i * (width / bars);
            const y = (height - h) / 2;
            
            ctx.beginPath();
            ctx.roundRect(x, y, barWidth, h, 2);
            ctx.fill();
        }
    }

    function animateWaveform(canvas) {
        if (!isPlaying) return;
        drawFakeWaveform(canvas);
        requestAnimationFrame(() => animateWaveform(canvas));
    }
});

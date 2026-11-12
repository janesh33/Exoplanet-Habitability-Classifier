/**
 * app.js
 * ======
 * Scientific dashboard controller for the Exoplanet Habitability Classifier.
 * 
 * Features:
 * 1. Minimal Sticky Centered Navigation with active section indicator.
 * 2. Full-screen Cinematic Hero with central orbit animation.
 * 3. Search & Select Planet (dataset-driven, clean searchable dropdown).
 * 4. Astronomy-Style Loading State during analysis.
 * 5. Clean Scientific Prediction Card:
 *    - Strictly HABITABLE / NON-HABITABLE verdict
 *    - 'XX% HABITABLE SCORE' badge ONLY displayed for HABITABLE planets
 *    - Absolutely no ML model names or algorithm details
 * 6. Planet Profile (clean stat cards from actual dataset measurements).
 * 7. Habitability Breakdown (smooth animated horizontal meters).
 * 8. Simplified Key Factors (physical astrophysics parameters only).
 * 9. Earth Reference Comparison (selected planet vs Earth baseline).
 * 10. Document Analysis with strict planet identity matching and simplified 4-column evidence table.
 * 11. Clean Document Analysis Result Card with score badge for habitable only.
 * 12. Global Scroll & Reveal animations.
 */

document.addEventListener('DOMContentLoaded', () => {

    // =========================================================================
    // API Request Helper
    // =========================================================================
    async function apiRequest(endpoint, options = {}) {
        const urls = [
            endpoint,
            `http://127.0.0.1:8000${endpoint}`,
            `http://localhost:8000${endpoint}`
        ];

        let lastErr = null;
        for (const url of urls) {
            try {
                const res = await fetch(url, options);
                if (res.ok) {
                    return await res.json();
                } else {
                    const errBody = await res.json().catch(() => ({}));
                    lastErr = new Error(errBody.detail || `Server responded with ${res.status}`);
                }
            } catch (err) {
                lastErr = err;
            }
        }
        throw lastErr || new Error(`Failed to request ${endpoint}`);
    }

    // =========================================================================
    // Global State
    // =========================================================================
    let catalogPlanets = [];
    let selectedPlanet = null;
    let docAnalysisSelectedPlanet = null;
    let currentUploadedFile = null;
    let activeDropdownIndex = -1;

    // =========================================================================
    // DOM Element References
    // =========================================================================
    // Astronomy Loader
    const astronomyLoader = document.getElementById('astronomyLoader');
    const loaderMessage = document.getElementById('loaderMessage');

    // Search / Selection
    const searchInput = document.getElementById('planetSearchInput');
    const dropdown = document.getElementById('planetDropdown');
    const predictBtn = document.getElementById('predictBtn');
    const resetBtn = document.getElementById('resetBtn');
    const quickPickChips = document.getElementById('quickPickChips');

    // Main Prediction Result Card
    const resultSection = document.getElementById('result-section');
    const resultPlanetName = document.getElementById('resultPlanetName');
    const predictionVerdictBox = document.getElementById('predictionVerdictBox');
    const predictionVerdictText = document.getElementById('predictionVerdictText');
    const habitableScoreBadge = document.getElementById('habitableScoreBadge');

    // Planet Profile Stat Cards
    const profRadius = document.getElementById('profRadius');
    const profMass = document.getElementById('profMass');
    const profDensity = document.getElementById('profDensity');
    const profPeriod = document.getElementById('profPeriod');
    const profSemiMajor = document.getElementById('profSemiMajor');
    const profEccentricity = document.getElementById('profEccentricity');
    const profStellarTeff = document.getElementById('profStellarTeff');
    const profStellarRadius = document.getElementById('profStellarRadius');
    const profStellarMass = document.getElementById('profStellarMass');

    // Habitability Breakdown Meters
    const barValSize = document.getElementById('barValSize');
    const barFillSize = document.getElementById('barFillSize');
    const barValStellar = document.getElementById('barValStellar');
    const barFillStellar = document.getElementById('barFillStellar');
    const barValOrbital = document.getElementById('barValOrbital');
    const barFillOrbital = document.getElementById('barFillOrbital');
    const barValTemp = document.getElementById('barValTemp');
    const barFillTemp = document.getElementById('barFillTemp');

    // Earth Reference Comparison
    const thPlanetName = document.getElementById('thPlanetName');
    const earthComparisonBody = document.getElementById('earthComparisonBody');
    const visValPlanetRadius = document.getElementById('visValPlanetRadius');
    const visFillPlanetRadius = document.getElementById('visFillPlanetRadius');
    const visValPlanetAxis = document.getElementById('visValPlanetAxis');
    const visFillPlanetAxis = document.getElementById('visFillPlanetAxis');
    const visValPlanetTeff = document.getElementById('visValPlanetTeff');
    const visFillPlanetTeff = document.getElementById('visFillPlanetTeff');

    // Document Analysis Upload
    const uploadDropzone = document.getElementById('uploadDropzone');
    const docFileInput = document.getElementById('docFileInput');
    const selectedFileName = document.getElementById('selectedFileName');
    const selectedFileSize = document.getElementById('selectedFileSize');
    const uploadBtn = document.getElementById('uploadBtn');
    const testPdfBtn = document.getElementById('testPdfBtn');
    const testImgBtn = document.getElementById('testImgBtn');

    // Document Analysis Results
    const docResultContainer = document.getElementById('docResultContainer');
    const docMatchStatus = document.getElementById('docMatchStatus');
    const evidencePlanetName = document.getElementById('evidencePlanetName');
    const evidenceCompleteness = document.getElementById('evidenceCompleteness');
    const evidenceDatasetMatch = document.getElementById('evidenceDatasetMatch');
    const evidenceTableBody = document.getElementById('evidenceTableBody');
    const multiplePlanetsNotice = document.getElementById('multiplePlanetsNotice');
    const multiplePlanetsButtons = document.getElementById('multiplePlanetsButtons');

    // Document Clean Scientific Result Card
    const docPredictionCard = document.getElementById('docPredictionCard');
    const docResultPlanetName = document.getElementById('docResultPlanetName');
    const docVerdictBox = document.getElementById('docVerdictBox');
    const docResultPrediction = document.getElementById('docResultPrediction');
    const docHabitableScoreBadge = document.getElementById('docHabitableScoreBadge');
    const docBarValSize = document.getElementById('docBarValSize');
    const docBarFillSize = document.getElementById('docBarFillSize');
    const docBarValStellar = document.getElementById('docBarValStellar');
    const docBarFillStellar = document.getElementById('docBarFillStellar');
    const docBarValOrbital = document.getElementById('docBarValOrbital');
    const docBarFillOrbital = document.getElementById('docBarFillOrbital');
    const docBarValTemp = document.getElementById('docBarValTemp');
    const docBarFillTemp = document.getElementById('docBarFillTemp');

    // =========================================================================
    // Astronomy Loading State Helper
    // =========================================================================
    function showLoader(msg = 'ANALYZING EXOPLANET TELEMETRY') {
        if (loaderMessage) loaderMessage.textContent = msg;
        if (astronomyLoader) astronomyLoader.classList.remove('hidden');
    }

    function hideLoader() {
        if (astronomyLoader) astronomyLoader.classList.add('hidden');
    }

    // =========================================================================
    // 1. Initial Catalog Telemetry Loading
    // =========================================================================
    async function initDashboard() {
        try {
            const data = await apiRequest('/api/planets');
            if (data && data.planets) {
                catalogPlanets = data.planets;
            }
        } catch (err) {
            console.warn('Initial planet list warning:', err.message);
        }

        // Initialize scroll reveal animations
        initScrollReveal();
    }

    // =========================================================================
    // 2. Searchable Planet Dropdown UI
    // =========================================================================
    function renderDropdown(items) {
        dropdown.innerHTML = '';
        activeDropdownIndex = -1;

        if (!items || items.length === 0) {
            const emptyEl = document.createElement('div');
            emptyEl.className = 'dropdown-item text-muted';
            emptyEl.textContent = 'No matching exoplanets found';
            dropdown.appendChild(emptyEl);
            dropdown.classList.remove('hidden');
            return;
        }

        items.slice(0, 25).forEach(name => {
            const item = document.createElement('div');
            item.className = 'dropdown-item';
            item.textContent = name;
            item.addEventListener('click', () => {
                selectPlanet(name);
            });
            dropdown.appendChild(item);
        });

        dropdown.classList.remove('hidden');
    }

    function selectPlanet(planetName) {
        selectedPlanet = planetName;
        searchInput.value = planetName;
        dropdown.classList.add('hidden');
        predictBtn.disabled = false;

        // Highlight matching chip if available
        document.querySelectorAll('.preset-chip').forEach(chip => {
            if (chip.dataset.planet.toLowerCase() === planetName.toLowerCase()) {
                chip.classList.add('active');
            } else {
                chip.classList.remove('active');
            }
        });
    }

    // Open dropdown on focus or click
    searchInput.addEventListener('focus', () => {
        const query = searchInput.value.trim().toLowerCase();
        if (query.length === 0) {
            renderDropdown(catalogPlanets.slice(0, 20));
        } else {
            const matches = catalogPlanets.filter(name => name.toLowerCase().includes(query));
            renderDropdown(matches);
        }
    });

    searchInput.addEventListener('input', () => {
        const query = searchInput.value.trim().toLowerCase();
        selectedPlanet = null;
        predictBtn.disabled = true;

        if (query.length === 0) {
            renderDropdown(catalogPlanets.slice(0, 20));
            return;
        }

        const matches = catalogPlanets.filter(name => name.toLowerCase().includes(query));
        renderDropdown(matches);
    });

    searchInput.addEventListener('keydown', (e) => {
        const items = dropdown.querySelectorAll('.dropdown-item');
        if (items.length === 0 || dropdown.classList.contains('hidden')) return;

        if (e.key === 'ArrowDown') {
            e.preventDefault();
            activeDropdownIndex = Math.min(activeDropdownIndex + 1, items.length - 1);
            updateDropdownHighlight(items);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            activeDropdownIndex = Math.max(activeDropdownIndex - 1, 0);
            updateDropdownHighlight(items);
        } else if (e.key === 'Enter') {
            e.preventDefault();
            if (activeDropdownIndex >= 0 && activeDropdownIndex < items.length) {
                selectPlanet(items[activeDropdownIndex].textContent);
            } else if (items.length > 0) {
                selectPlanet(items[0].textContent);
            }
        } else if (e.key === 'Escape') {
            dropdown.classList.add('hidden');
        }
    });

    function updateDropdownHighlight(items) {
        items.forEach((item, idx) => {
            if (idx === activeDropdownIndex) {
                item.classList.add('active');
                item.scrollIntoView({ block: 'nearest' });
            } else {
                item.classList.remove('active');
            }
        });
    }

    // Close dropdown on outside click
    document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target) && !dropdown.contains(e.target)) {
            dropdown.classList.add('hidden');
        }
    });

    // Preset selection chips
    if (quickPickChips) {
        quickPickChips.querySelectorAll('.preset-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const planet = chip.dataset.planet;
                selectPlanet(planet);
            });
        });
    }

    // Reset button
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            searchInput.value = '';
            selectedPlanet = null;
            predictBtn.disabled = true;
            dropdown.classList.add('hidden');
            document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
            resultSection.classList.add('hidden');
        });
    }

    // =========================================================================
    // 3. Planet Analysis Action ([ANALYZE PLANET])
    // =========================================================================
    predictBtn.addEventListener('click', async () => {
        const planetToAnalyze = selectedPlanet || searchInput.value.trim();
        if (!planetToAnalyze) return;

        predictBtn.disabled = true;
        showLoader('ANALYZING EXOPLANET TELEMETRY');

        try {
            await executePlanetAnalysis(planetToAnalyze);
        } catch (err) {
            alert(`Analysis Error: ${err.message}`);
        } finally {
            predictBtn.disabled = false;
            hideLoader();
        }
    });

    async function executePlanetAnalysis(planetName) {
        const [predData, planetData] = await Promise.all([
            apiRequest('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ planet_name: planetName })
            }),
            apiRequest(`/api/planet/${encodeURIComponent(planetName)}`)
        ]);

        renderAnalysisResults(predData, planetData);

        // Smooth scroll to prediction result
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function renderAnalysisResults(predData, planetData) {
        const { prediction, model_confidence, habitability_breakdown } = predData;
        const profile = (planetData && planetData.profile) ? planetData.profile : {};
        const earthComp = (planetData && planetData.earth_comparison) ? planetData.earth_comparison : [];

        // Reveal view container
        resultSection.classList.remove('hidden');

        // Target Planet Name
        resultPlanetName.textContent = predData.planet || profile.name || selectedPlanet;

        // Strictly Binary Verdict (HABITABLE / NON-HABITABLE)
        const isHabitable = prediction === 'HABITABLE';
        predictionVerdictText.textContent = prediction;
        predictionVerdictBox.className = `verdict-banner-box ${isHabitable ? 'verdict-habitable' : 'verdict-non-habitable'}`;

        // Habitable Score Badge: Displayed ONLY when result is HABITABLE
        if (habitableScoreBadge) {
            if (isHabitable && model_confidence !== undefined && model_confidence !== null) {
                habitableScoreBadge.textContent = `${Math.round(model_confidence)}% HABITABLE SCORE`;
                habitableScoreBadge.classList.remove('hidden');
            } else {
                habitableScoreBadge.classList.add('hidden');
            }
        }

        // Planet Profile Stat Cards
        profRadius.textContent = formatMetric(profile.radius_eu, ' R⊕');
        profMass.textContent = formatMetric(profile.mass_eu, ' M⊕');
        profDensity.textContent = formatMetric(profile.density_eu, ' ρ⊕');
        profPeriod.textContent = formatMetric(profile.period_days, ' days');
        profSemiMajor.textContent = formatMetric(profile.sem_major_axis_au, ' AU');
        profEccentricity.textContent = formatMetric(profile.eccentricity, '');
        profStellarTeff.textContent = formatMetric(profile.stellar_teff_k, ' K');
        profStellarRadius.textContent = formatMetric(profile.stellar_radius_su, ' R☉');
        profStellarMass.textContent = formatMetric(profile.stellar_mass_su, ' M☉');

        // Habitability Breakdown with Smooth Animated Progress Bars
        if (habitability_breakdown) {
            setMeter(barValSize, barFillSize, habitability_breakdown.planetary_size);
            setMeter(barValStellar, barFillStellar, habitability_breakdown.stellar_conditions);
            setMeter(barValOrbital, barFillOrbital, habitability_breakdown.orbital_conditions);
            setMeter(barValTemp, barFillTemp, habitability_breakdown.temperature);
        }

        // Visual Analysis (Selected Planet vs Earth Reference)
        if (thPlanetName) {
            thPlanetName.textContent = predData.planet || 'Selected Planet';
        }
        renderComparisonTable(earthComp);
        renderVisualCharts(profile);
    }

    function setMeter(textEl, fillEl, val) {
        if (!textEl || !fillEl) return;
        const clamped = Math.min(Math.max(val || 0, 0), 100);
        textEl.textContent = `${clamped}%`;
        fillEl.style.width = '0%';
        setTimeout(() => {
            fillEl.style.width = `${clamped}%`;
        }, 80);
    }

    function formatMetric(val, unit = '') {
        if (val === null || val === undefined || isNaN(val)) return '—';
        const num = parseFloat(val);
        if (isNaN(num)) return String(val);
        if (num >= 1000) {
            return num.toLocaleString(undefined, { maximumFractionDigits: 1 }) + unit;
        }
        return num.toFixed(2) + unit;
    }

    function renderComparisonTable(comparisons) {
        earthComparisonBody.innerHTML = '';
        if (!comparisons || comparisons.length === 0) {
            earthComparisonBody.innerHTML = `<tr><td colspan="5" class="text-muted">No physical comparison metrics available.</td></tr>`;
            return;
        }

        comparisons.forEach(item => {
            const tr = document.createElement('tr');
            const planetValStr = item.planet_val !== null ? item.planet_val : '—';
            const earthValStr = item.earth_val !== null ? item.earth_val : '—';
            let ratioStr = '—';
            if (item.planet_val !== null && item.earth_val !== null && item.earth_val !== 0) {
                ratioStr = `${(item.planet_val / item.earth_val).toFixed(2)}×`;
            }

            tr.innerHTML = `
                <td><strong>${item.feature}</strong></td>
                <td><strong>${planetValStr}</strong></td>
                <td>${earthValStr}</td>
                <td>${item.unit}</td>
                <td><span class="badge-info">${ratioStr}</span></td>
            `;
            earthComparisonBody.appendChild(tr);
        });
    }

    function renderVisualCharts(profile) {
        // Planetary Radius (1.0 R⊕ baseline)
        if (profile.radius_eu !== null && !isNaN(profile.radius_eu)) {
            const rad = parseFloat(profile.radius_eu);
            visValPlanetRadius.textContent = `${rad.toFixed(2)} R⊕`;
            const fillWidth = Math.min(Math.max((rad / 5.0) * 100, 2), 100);
            visFillPlanetRadius.style.width = `${fillWidth}%`;
        } else {
            visValPlanetRadius.textContent = '—';
            visFillPlanetRadius.style.width = '0%';
        }

        // Semi-Major Axis (1.0 AU baseline)
        if (profile.sem_major_axis_au !== null && !isNaN(profile.sem_major_axis_au)) {
            const axis = parseFloat(profile.sem_major_axis_au);
            visValPlanetAxis.textContent = `${axis.toFixed(2)} AU`;
            const fillWidth = Math.min(Math.max((axis / 3.0) * 100, 2), 100);
            visFillPlanetAxis.style.width = `${fillWidth}%`;
        } else {
            visValPlanetAxis.textContent = '—';
            visFillPlanetAxis.style.width = '0%';
        }

        // Host Star Temperature (Sun = 5778 K baseline)
        if (profile.stellar_teff_k !== null && !isNaN(profile.stellar_teff_k)) {
            const teff = parseFloat(profile.stellar_teff_k);
            visValPlanetTeff.textContent = `${Math.round(teff).toLocaleString()} K`;
            const fillWidth = Math.min(Math.max((teff / 10000.0) * 100, 2), 100);
            visFillPlanetTeff.style.width = `${fillWidth}%`;
        } else {
            visValPlanetTeff.textContent = '—';
            visFillPlanetTeff.style.width = '0%';
        }
    }

    // =========================================================================
    // 4. Document Analysis (Upload, Multi-planet Choice, Clean Result Card)
    // =========================================================================
    if (uploadDropzone && docFileInput) {
        uploadDropzone.addEventListener('click', () => docFileInput.click());

        uploadDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadDropzone.classList.add('dragover');
        });

        uploadDropzone.addEventListener('dragleave', () => {
            uploadDropzone.classList.remove('dragover');
        });

        uploadDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadDropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleFileSelection(e.dataTransfer.files[0]);
            }
        });

        docFileInput.addEventListener('change', () => {
            if (docFileInput.files.length > 0) {
                handleFileSelection(docFileInput.files[0]);
            }
        });
    }

    function handleFileSelection(file) {
        currentUploadedFile = file;
        docAnalysisSelectedPlanet = null;
        if (multiplePlanetsNotice) {
            multiplePlanetsNotice.classList.add('hidden');
        }
        if (multiplePlanetsButtons) {
            multiplePlanetsButtons.innerHTML = '';
        }
        uploadBtn.disabled = false;
        if (selectedFileName) {
            selectedFileName.textContent = file.name;
        }
        if (selectedFileSize) {
            selectedFileSize.textContent = `(${(file.size / 1024).toFixed(1)} KB)`;
        }
    }

    async function triggerDocAnalysisForPlanet(planetName) {
        if (!currentUploadedFile) return;
        docAnalysisSelectedPlanet = planetName;
        uploadBtn.disabled = true;
        showLoader(`ANALYZING ${planetName.toUpperCase()} DATA`);

        try {
            const formData = new FormData();
            formData.append('file', currentUploadedFile);
            formData.append('selected_planet', planetName);

            const res = await apiRequest('/api/analyze-document', {
                method: 'POST',
                body: formData
            });

            displayDocAnalysis(res);
        } catch (err) {
            alert(`Document Analysis Error: ${err.message}`);
        } finally {
            uploadBtn.disabled = false;
            hideLoader();
        }
    }

    if (uploadBtn) {
        uploadBtn.addEventListener('click', async () => {
            if (!currentUploadedFile) return;
            uploadBtn.disabled = true;
            showLoader('PARSING SCIENTIFIC DOCUMENT');

            try {
                const formData = new FormData();
                formData.append('file', currentUploadedFile);
                if (docAnalysisSelectedPlanet) {
                    formData.append('selected_planet', docAnalysisSelectedPlanet);
                }

                const res = await apiRequest('/api/analyze-document', {
                    method: 'POST',
                    body: formData
                });

                displayDocAnalysis(res);
            } catch (err) {
                alert(`Document Analysis Error: ${err.message}`);
            } finally {
                uploadBtn.disabled = false;
                hideLoader();
            }
        });
    }

    // Sample Test Buttons
    if (testPdfBtn) {
        testPdfBtn.addEventListener('click', async () => {
            testPdfBtn.disabled = true;
            showLoader('ANALYZING SAMPLE KEPLER-452 b PAPER');
            try {
                const res = await fetch('/static/assets/test_samples/test_kepler452b_paper.pdf');
                if (!res.ok) throw new Error('Sample PDF not found on server.');
                const blob = await res.blob();
                const file = new File([blob], 'test_kepler452b_paper.pdf', { type: 'application/pdf' });
                
                handleFileSelection(file);
                const formData = new FormData();
                formData.append('file', file);
                const data = await apiRequest('/api/analyze-document', {
                    method: 'POST',
                    body: formData
                });
                displayDocAnalysis(data);
            } catch (err) {
                alert(`Sample PDF Error: ${err.message}`);
            } finally {
                testPdfBtn.disabled = false;
                hideLoader();
            }
        });
    }

    if (testImgBtn) {
        testImgBtn.addEventListener('click', async () => {
            testImgBtn.disabled = true;
            showLoader('ANALYZING OBSERVATION SHEET IMAGE');
            try {
                const res = await fetch('/static/assets/test_samples/test_kepler452b_sheet.png');
                if (!res.ok) throw new Error('Sample image not found on server.');
                const blob = await res.blob();
                const file = new File([blob], 'test_kepler452b_sheet.png', { type: 'image/png' });
                
                handleFileSelection(file);
                const formData = new FormData();
                formData.append('file', file);
                const data = await apiRequest('/api/analyze-document', {
                    method: 'POST',
                    body: formData
                });
                displayDocAnalysis(data);
            } catch (err) {
                alert(`Sample Image Error: ${err.message}`);
            } finally {
                testImgBtn.disabled = false;
                hideLoader();
            }
        });
    }

    function displayDocAnalysis(data) {
        if (!data || data.status === 'error') {
            alert(data ? data.message : 'Analysis failed.');
            return;
        }

        docResultContainer.classList.remove('hidden');

        // Detected Planet Name
        const detectedName = data.selected_planet || (data.detected_planets && data.detected_planets[0]) || '—';
        if (evidencePlanetName) evidencePlanetName.textContent = detectedName;
        if (docResultPlanetName) docResultPlanetName.textContent = detectedName;

        // Multiple Planets Choice Bar (RULE 11)
        if (data.multiple_planets && data.detected_planets && data.detected_planets.length > 1) {
            if (multiplePlanetsNotice) multiplePlanetsNotice.classList.remove('hidden');
            if (multiplePlanetsButtons) {
                multiplePlanetsButtons.innerHTML = '';
                data.detected_planets.forEach(pName => {
                    const btn = document.createElement('button');
                    btn.type = 'button';
                    btn.className = `mp-btn ${pName === detectedName ? 'active' : ''}`;
                    btn.textContent = pName;
                    btn.addEventListener('click', () => {
                        triggerDocAnalysisForPlanet(pName);
                    });
                    multiplePlanetsButtons.appendChild(btn);
                });
            }
        } else {
            if (multiplePlanetsNotice) multiplePlanetsNotice.classList.add('hidden');
        }

        // Handle Catalogue Match / Error Cases (RULES 6 & 7)
        if (data.status === 'not_found' || data.status === 'mismatch' || !data.dataset_match) {
            if (evidenceDatasetMatch) {
                evidenceDatasetMatch.innerHTML = `<span class="badge-danger">✕ Planet not found in current dataset</span>`;
            }
            if (docMatchStatus) {
                docMatchStatus.innerHTML = `<span class="badge-danger">Unmatched</span>`;
            }
            if (evidenceCompleteness) {
                evidenceCompleteness.textContent = 'No verified catalogue record';
            }
            if (docPredictionCard) {
                docPredictionCard.classList.add('hidden');
            }
            if (evidenceTableBody) {
                evidenceTableBody.innerHTML = `<tr><td colspan="4" class="text-danger" style="text-align: center; padding: 24px;"><strong>${data.message || 'No verified catalogue match found. Analysis halted.'}</strong></td></tr>`;
            }
            docResultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            return;
        }

        // Verified Match Confirmed
        if (evidenceDatasetMatch) {
            evidenceDatasetMatch.innerHTML = `<span class="badge-success">✓ Planet found in current dataset</span>`;
        }
        if (docMatchStatus) {
            docMatchStatus.innerHTML = `<span class="badge-success">✓ Catalog Match</span>`;
        }

        // Completeness
        if (data.completeness && evidenceCompleteness) {
            evidenceCompleteness.textContent = `${data.completeness.found} / ${data.completeness.total} required inputs available`;
        }

        // Simplified Evidence Table (Section 10: PARAMETER | EXTRACTED VALUE | SCIENTIFIC UNIT | CONTEXT)
        if (evidenceTableBody) {
            evidenceTableBody.innerHTML = '';
            const features = data.extracted_features || {};
            const entries = Object.entries(features);

            if (entries.length === 0) {
                evidenceTableBody.innerHTML = `<tr><td colspan="4" class="text-muted">No explicit parameters extracted.</td></tr>`;
            } else {
                entries.forEach(([paramName, info]) => {
                    const tr = document.createElement('tr');
                    const isVerified = info.verified !== false;
                    const valDisplay = isVerified ? (info.value !== null ? info.value : '—') : 'Not available / not verified';
                    tr.innerHTML = `
                        <td><strong>${paramName}</strong></td>
                        <td><strong class="${isVerified ? '' : 'text-muted'}">${valDisplay}</strong></td>
                        <td>${info.unit || '—'}</td>
                        <td class="context-snippet"><em>"${info.snippet || '—'}"</em></td>
                    `;
                    evidenceTableBody.appendChild(tr);
                });
            }
        }

        // Clean Scientific Result Card (Section 11)
        if (data.prediction && docPredictionCard) {
            docPredictionCard.classList.remove('hidden');

            if (docResultPlanetName) {
                docResultPlanetName.textContent = data.selected_planet;
            }

            if (docResultPrediction) {
                docResultPrediction.textContent = data.prediction;
            }

            const isHabitable = data.prediction === 'HABITABLE';
            if (docVerdictBox) {
                docVerdictBox.className = `doc-verdict-box ${isHabitable ? 'verdict-habitable' : 'verdict-non-habitable'}`;
            }

            // Habitable Score Badge (Section 11: Displayed ONLY when result is HABITABLE)
            if (docHabitableScoreBadge) {
                if (isHabitable && data.confidence) {
                    docHabitableScoreBadge.textContent = `${Math.round(data.confidence)}% HABITABLE SCORE`;
                    docHabitableScoreBadge.classList.remove('hidden');
                } else {
                    docHabitableScoreBadge.classList.add('hidden');
                }
            }

            // Habitability Parameter actual values & clean animated progress bars
            const breakdown = data.habitability_breakdown || {};
            const actuals = data.param_actual_values || {};

            if (docBarValSize) docBarValSize.textContent = actuals.planetary_size || 'Not available / not verified';
            if (docBarFillSize) docBarFillSize.style.width = `${breakdown.planetary_size || 0}%`;

            if (docBarValStellar) docBarValStellar.textContent = actuals.stellar_conditions || 'Not available / not verified';
            if (docBarFillStellar) docBarFillStellar.style.width = `${breakdown.stellar_conditions || 0}%`;

            if (docBarValOrbital) docBarValOrbital.textContent = actuals.orbital_conditions || 'Not available / not verified';
            if (docBarFillOrbital) docBarFillOrbital.style.width = `${breakdown.orbital_conditions || 0}%`;

            if (docBarValTemp) docBarValTemp.textContent = actuals.temperature || 'Not available / not verified';
            if (docBarFillTemp) docBarFillTemp.style.width = `${breakdown.temperature || 0}%`;
        }

        docResultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // =========================================================================
    // 5. Smooth Scrolling Navigation & Scrollspy
    // =========================================================================
    document.querySelectorAll('.nav-item, #heroBtnAnalyze, #heroBtnUpload').forEach(link => {
        link.addEventListener('click', (e) => {
            const href = link.getAttribute('href');
            if (href && href.startsWith('#')) {
                e.preventDefault();
                const targetId = href.substring(1);
                const targetEl = document.getElementById(targetId);
                if (targetEl) {
                    targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }

                document.querySelectorAll('.nav-item').forEach(l => l.classList.remove('active'));
                const navMatch = document.querySelector(`.nav-item[href="${href}"]`);
                if (navMatch) navMatch.classList.add('active');
            }
        });
    });

    // Scrollspy to automatically highlight active nav item on scroll
    window.addEventListener('scroll', () => {
        const sections = [
            document.getElementById('hero-section'),
            document.getElementById('analysis-section'),
            document.getElementById('document-section')
        ];

        let currentSectionId = 'hero-section';
        const scrollPosition = window.scrollY + 200;

        sections.forEach(sec => {
            if (sec && sec.offsetTop <= scrollPosition) {
                currentSectionId = sec.id;
            }
        });

        document.querySelectorAll('.nav-item').forEach(item => {
            const href = item.getAttribute('href');
            if (href === `#${currentSectionId}`) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
    }, { passive: true });

    // =========================================================================
    // 6. Global Scroll & Reveal Animations
    // =========================================================================
    function initScrollReveal() {
        const revealElements = document.querySelectorAll('.reveal-on-scroll');
        if (!('IntersectionObserver' in window)) {
            revealElements.forEach(el => el.classList.add('is-revealed'));
            return;
        }

        const observer = new IntersectionObserver((entries, obs) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('is-revealed');
                    obs.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.12,
            rootMargin: '0px 0px -40px 0px'
        });

        revealElements.forEach(el => observer.observe(el));
    }

    // Start initialization
    initDashboard();
});

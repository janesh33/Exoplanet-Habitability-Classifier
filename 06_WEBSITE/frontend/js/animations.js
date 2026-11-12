/**
 * animations.js
 * =============
 * Premium cinematic animation system for the Exoplanet Habitability Classifier.
 * Handles:
 *   - Hero scroll parallax
 *   - Desktop pointer parallax on planetary system
 *   - Extended IntersectionObserver scroll-reveal (up/left/right/scale)
 *   - Section heading animation trigger
 *   - Preset chip staggered entrance
 *   - Planet analysis result sequential reveal with animated counters & bars
 *   - Earth comparison table row stagger & chart bar animation
 *   - Document analysis flow animations
 *   - Multiple-planet panel slide-in/stagger
 *   - Catalogue verification sweep
 *   - Document evidence table row stagger
 *   - Smooth loader hide with fade-out
 *   - Habitable score animated counter
 * 
 * DOES NOT touch API calls, prediction logic, or backend.
 */

(function () {
    'use strict';

    /* ------------------------------------------------------------------
       Helpers
    ------------------------------------------------------------------ */
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function delay(ms) {
        return new Promise(r => setTimeout(r, ms));
    }

    function rAF() {
        return new Promise(r => requestAnimationFrame(r));
    }

    /* ------------------------------------------------------------------
       1. HERO SCROLL PARALLAX
    ------------------------------------------------------------------ */
    (function initParallax() {
        if (reducedMotion) return;

        const starsCanvas = document.querySelector('.stars-bg-canvas');
        const hero = document.getElementById('hero-section');
        if (!starsCanvas || !hero) return;

        let ticking = false;

        window.addEventListener('scroll', () => {
            if (ticking) return;
            ticking = true;
            requestAnimationFrame(() => {
                const scrollY = window.scrollY;
                const heroBottom = hero.getBoundingClientRect().bottom + scrollY;
                if (scrollY > heroBottom) { ticking = false; return; }

                // Stars move slowest (depth = far)
                starsCanvas.style.transform = `translateY(${scrollY * 0.12}px)`;

                ticking = false;
            });
        }, { passive: true });
    })();

    /* ------------------------------------------------------------------
       2. DESKTOP POINTER PARALLAX ON PLANETARY SYSTEM
    ------------------------------------------------------------------ */
    (function initPointerParallax() {
        if (reducedMotion) return;

        const systemWrapper = document.querySelector('.planetary-system-wrapper');
        const hero = document.getElementById('hero-section');
        if (!systemWrapper || !hero) return;

        // Only on non-touch desktops
        if (window.matchMedia('(hover: none)').matches) return;

        let targetX = 0, targetY = 0;
        let currentX = 0, currentY = 0;
        let rafId = null;

        const MAX_SHIFT = 8; // px max

        function lerp(a, b, t) { return a + (b - a) * t; }

        function tick() {
            currentX = lerp(currentX, targetX, 0.06);
            currentY = lerp(currentY, targetY, 0.06);
            systemWrapper.style.transform = `translate(${currentX}px, ${currentY}px)`;
            rafId = requestAnimationFrame(tick);
        }

        hero.addEventListener('mousemove', (e) => {
            const rect = hero.getBoundingClientRect();
            const cx = rect.left + rect.width / 2;
            const cy = rect.top + rect.height / 2;
            const dx = (e.clientX - cx) / (rect.width / 2);
            const dy = (e.clientY - cy) / (rect.height / 2);
            targetX = dx * MAX_SHIFT;
            targetY = dy * MAX_SHIFT;
        }, { passive: true });

        hero.addEventListener('mouseleave', () => {
            targetX = 0;
            targetY = 0;
        });

        // Start animation loop only while hero is visible
        const heroObserver = new IntersectionObserver(entries => {
            if (entries[0].isIntersecting) {
                if (!rafId) rafId = requestAnimationFrame(tick);
            } else {
                if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
                currentX = 0; currentY = 0;
                systemWrapper.style.transform = '';
            }
        });
        heroObserver.observe(hero);
    })();

    /* ------------------------------------------------------------------
       3. EXTENDED SCROLL-REVEAL OBSERVER
    ------------------------------------------------------------------ */
    function initScrollReveal() {
        const selectors = [
            '.reveal-on-scroll',
            '.reveal-from-left',
            '.reveal-from-right',
            '.reveal-scale',
            '.section-title-header'
        ].join(', ');

        const elements = document.querySelectorAll(selectors);

        if (!('IntersectionObserver' in window) || reducedMotion) {
            elements.forEach(el => {
                el.classList.add('is-revealed');
                el.querySelectorAll('h2').forEach(h => h.style.opacity = '1');
            });
            return;
        }

        const observer = new IntersectionObserver((entries, obs) => {
            entries.forEach(entry => {
                if (!entry.isIntersecting) return;
                entry.target.classList.add('is-revealed');
                obs.unobserve(entry.target);
            });
        }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

        elements.forEach(el => observer.observe(el));
    }

    /* ------------------------------------------------------------------
       4. PRESET CHIP STAGGERED ENTRANCE
    ------------------------------------------------------------------ */
    function initChipEntrance() {
        const chips = document.querySelectorAll('.preset-chip');
        if (!chips.length) return;

        if (reducedMotion) {
            chips.forEach(c => c.classList.add('chip-visible'));
            return;
        }

        chips.forEach((chip, i) => {
            setTimeout(() => {
                chip.classList.add('chip-visible');
            }, 600 + i * 70);
        });
    }

    /* ------------------------------------------------------------------
       5. ENHANCED LOADER (fade-out instead of snap)
    ------------------------------------------------------------------ */
    const loaderEl = document.getElementById('astronomyLoader');

    function showLoaderAnimated(msg) {
        if (!loaderEl) return;
        const msgEl = loaderEl.querySelector('#loaderMessage');
        if (msgEl) msgEl.textContent = msg || 'ANALYZING EXOPLANET TELEMETRY';
        loaderEl.classList.remove('loader-fade-out', 'hidden');
    }

    function hideLoaderAnimated() {
        if (!loaderEl) return;
        if (reducedMotion) {
            loaderEl.classList.add('hidden');
            return;
        }
        loaderEl.classList.add('loader-fade-out');
        setTimeout(() => {
            loaderEl.classList.add('hidden');
            loaderEl.classList.remove('loader-fade-out');
        }, 380);
    }

    // Monkey-patch the existing showLoader/hideLoader if app.js defines them
    // We wait a tick so app.js's DOMContentLoaded fires first
    setTimeout(() => {
        // These are already global in app.js's closure — we hook via DOM events
    }, 0);

    /* ------------------------------------------------------------------
       6. ANIMATED COUNTER (number roll-up)
    ------------------------------------------------------------------ */
    function animateCounter(el, from, to, duration, suffix) {
        if (reducedMotion) { el.textContent = `${to}${suffix}`; return; }
        const start = performance.now();
        function step(now) {
            const progress = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3); // ease-out-cubic
            const value = Math.round(from + (to - from) * eased);
            el.textContent = `${value}${suffix}`;
            if (progress < 1) requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
    }

    /* ------------------------------------------------------------------
       7. ANIMATED METER BAR
    ------------------------------------------------------------------ */
    function animateMeter(fillEl, valEl, targetPct, staggerMs) {
        if (!fillEl) return;
        if (reducedMotion) {
            fillEl.style.width = `${targetPct}%`;
            if (valEl) valEl.textContent = `${targetPct}%`;
            return;
        }

        // Reset
        fillEl.style.width = '0%';

        setTimeout(() => {
            requestAnimationFrame(() => {
                fillEl.style.width = `${targetPct}%`;
                // Fade in the numeric label slightly after bar starts
                if (valEl) {
                    setTimeout(() => valEl.classList.add('val-visible'), 200);
                }
            });
        }, staggerMs);
    }

    /* ------------------------------------------------------------------
       8. REVEAL RESULT SECTION SEQUENTIALLY
    ------------------------------------------------------------------ */
    window.animateResultReveal = function (predData, planetData) {
        const resultSection = document.getElementById('result-section');
        if (!resultSection) return;

        if (reducedMotion) {
            // Just reveal everything immediately
            resultSection.querySelectorAll('.stat-box').forEach(b => b.classList.add('stat-visible'));
            resultSection.querySelectorAll('.factor-item').forEach(f => f.classList.add('factor-visible'));
            resultSection.querySelectorAll('.meter-header strong').forEach(s => s.classList.add('val-visible'));
            return;
        }

        const mainCard = document.getElementById('mainResultCard');
        const profileSection = document.getElementById('profileSection');

        // Step 1: Reveal main prediction card
        requestAnimationFrame(() => {
            if (mainCard) {
                mainCard.classList.remove('card-revealed');
                void mainCard.offsetWidth; // force reflow
                setTimeout(() => mainCard.classList.add('card-revealed'), 80);
            }
        });

        // Step 2: Animate verdict text in sequence
        setTimeout(() => {
            const verdictTag = resultSection.querySelector('.verdict-tag');
            const verdictTitle = document.getElementById('predictionVerdictText');
            const scoreBadge = document.getElementById('habitableScoreBadge');

            if (verdictTag) verdictTag.classList.add('verdict-in');
            setTimeout(() => {
                if (verdictTitle) verdictTitle.classList.add('verdict-in');
            }, 150);
            setTimeout(() => {
                if (scoreBadge && !scoreBadge.classList.contains('hidden')) {
                    scoreBadge.classList.add('verdict-in');

                    // Animated counter for habitable score
                    const scoreText = scoreBadge.textContent;
                    const numMatch = scoreText.match(/(\d+)/);
                    if (numMatch) {
                        const target = parseInt(numMatch[1]);
                        animateCounter(scoreBadge, 0, target, 900, '% HABITABLE SCORE');
                    }
                }
            }, 350);
        }, 300);

        // Step 3: Profile card
        setTimeout(() => {
            if (profileSection) {
                profileSection.classList.remove('card-revealed');
                void profileSection.offsetWidth;
                profileSection.classList.add('card-revealed');
            }

            // Stagger stat boxes
            const statBoxes = resultSection.querySelectorAll('.stat-box');
            statBoxes.forEach((box, i) => {
                setTimeout(() => box.classList.add('stat-visible'), i * 60);
            });
        }, 500);

        // Step 4: Animate meter bars (staggered)
        const barPairs = [
            ['barFillSize', 'barValSize'],
            ['barFillStellar', 'barValStellar'],
            ['barFillOrbital', 'barValOrbital'],
            ['barFillTemp', 'barValTemp']
        ];

        barPairs.forEach(([fillId, valId], i) => {
            const fill = document.getElementById(fillId);
            const val = document.getElementById(valId);
            if (!fill) return;
            const targetW = parseFloat(fill.getAttribute('data-target') || fill.style.width || '0');
            // Store target before we reset
            animateMeter(fill, val, targetW, 700 + i * 130);
        });

        // Step 5: Key factors alternating slide
        setTimeout(() => {
            const factors = resultSection.querySelectorAll('.factor-item');
            factors.forEach((f, i) => {
                setTimeout(() => f.classList.add('factor-visible'), i * 100);
            });
        }, 800);

        // Step 6: Comparison table row stagger
        setTimeout(() => {
            const table = resultSection.querySelector('.comparison-table');
            if (table) {
                table.classList.add('table-visible');
                const rows = table.querySelectorAll('tbody tr');
                rows.forEach((row, i) => {
                    setTimeout(() => row.classList.add('row-visible'), 100 + i * 60);
                });
            }

            // Animate importance bars
            const impDivs = resultSection.querySelectorAll('.imp-track div');
            impDivs.forEach((div, i) => {
                const target = div.style.width;
                div.style.width = '0';
                setTimeout(() => {
                    div.style.width = target;
                }, 200 + i * 80);
            });
        }, 1000);
    };

    /* ------------------------------------------------------------------
       9. DOCUMENT ANALYSIS ANIMATIONS
    ------------------------------------------------------------------ */

    // File selected icon bounce
    window.animateFileSelected = function () {
        const icon = document.querySelector('.doc-icon');
        if (!icon || reducedMotion) return;
        icon.classList.add('file-selected');
        setTimeout(() => icon.classList.remove('file-selected'), 500);

        // Filename reveal
        const nameEl = document.getElementById('selectedFileName');
        if (nameEl) {
            nameEl.classList.add('selected-file-reveal');
            setTimeout(() => nameEl.classList.remove('selected-file-reveal'), 500);
        }
    };

    // Multiple planets panel slide-down
    window.animateMultiplePlanets = function (show) {
        const notice = document.getElementById('multiplePlanetsNotice');
        if (!notice) return;

        if (show) {
            notice.classList.add('mp-open');
            // Stagger buttons
            const btns = notice.querySelectorAll('.mp-btn');
            btns.forEach((btn, i) => {
                btn.classList.remove('mp-btn-visible');
                setTimeout(() => btn.classList.add('mp-btn-visible'), 100 + i * 70);
            });
        } else {
            notice.classList.remove('mp-open');
        }
    };

    // Catalogue verification sweep
    window.animateCatalogueVerify = function (el) {
        if (!el || reducedMotion) return;
        el.classList.add('catalogue-verifying');
        setTimeout(() => el.classList.remove('catalogue-verifying'), 1100);
    };

    // Document result reveal
    window.animateDocResult = function () {
        const container = document.getElementById('docResultContainer');
        if (!container || reducedMotion) return;

        container.classList.remove('doc-revealed');
        void container.offsetWidth;

        setTimeout(() => {
            container.classList.add('doc-revealed');

            // Evidence stat cards
            const statCards = container.querySelectorAll('.evidence-stat-card');
            statCards.forEach((card, i) => {
                card.classList.remove('stat-visible');
                setTimeout(() => card.classList.add('stat-visible'), 150 + i * 80);
            });

            // Evidence table rows
            setTimeout(() => {
                const table = container.querySelector('.evidence-table');
                if (table) {
                    table.classList.add('table-visible');
                    const rows = table.querySelectorAll('tbody tr');
                    rows.forEach((row, i) => {
                        row.classList.remove('row-visible');
                        setTimeout(() => row.classList.add('row-visible'), 100 + i * 55);
                    });
                }
            }, 300);

            // Document prediction card
            setTimeout(() => {
                const docCard = document.getElementById('docPredictionCard');
                if (docCard && !docCard.classList.contains('hidden')) {
                    docCard.classList.remove('doc-card-revealed');
                    void docCard.offsetWidth;
                    docCard.classList.add('doc-card-revealed');

                    // Verdict in
                    setTimeout(() => {
                        docCard.querySelectorAll('.verdict-label, .doc-verdict-title').forEach(el => el.classList.add('verdict-in'));
                    }, 250);

                    // Doc habitable score badge counter
                    const docScoreBadge = document.getElementById('docHabitableScoreBadge');
                    if (docScoreBadge && !docScoreBadge.classList.contains('hidden')) {
                        const scoreText = docScoreBadge.textContent;
                        const numMatch = scoreText.match(/(\d+)/);
                        if (numMatch) {
                            const target = parseInt(numMatch[1]);
                            setTimeout(() => {
                                animateCounter(docScoreBadge, 0, target, 900, '% HABITABLE SCORE');
                            }, 400);
                        }
                    }

                    // Doc meter bars
                    const docBarPairs = [
                        ['docBarFillSize', 'docBarValSize'],
                        ['docBarFillStellar', 'docBarValStellar'],
                        ['docBarFillOrbital', 'docBarValOrbital'],
                        ['docBarFillTemp', 'docBarValTemp']
                    ];

                    docBarPairs.forEach(([fillId, valId], i) => {
                        const fill = document.getElementById(fillId);
                        const val = document.getElementById(valId);
                        if (!fill) return;
                        const targetW = parseFloat(fill.getAttribute('data-target') || fill.style.width || '0');
                        animateMeter(fill, val, targetW, 350 + i * 120);
                    });
                }
            }, 500);

        }, 60);
    };

    /* ------------------------------------------------------------------
       10. INTERCEPT setMeter CALLS — store target in data-target
          so our animator can read it back
    ------------------------------------------------------------------ */
    // We wrap the existing setMeter pattern by observing style mutations
    // on meter fill elements to capture their target widths before reset
    (function patchMeterFills() {
        const fillIds = [
            'barFillSize', 'barFillStellar', 'barFillOrbital', 'barFillTemp',
            'docBarFillSize', 'docBarFillStellar', 'docBarFillOrbital', 'docBarFillTemp'
        ];

        fillIds.forEach(id => {
            const el = document.getElementById(id);
            if (!el) return;

            // Use a MutationObserver to catch when app.js sets style.width
            const obs = new MutationObserver(() => {
                const w = el.style.width;
                if (w && w !== '0%') {
                    el.setAttribute('data-target', parseFloat(w));
                }
            });
            obs.observe(el, { attributes: true, attributeFilter: ['style'] });
        });
    })();

    /* ------------------------------------------------------------------
       11. HOOK INTO app.js EVENTS VIA DOM MUTATION / CUSTOM EVENTS
    ------------------------------------------------------------------ */

    // Watch result-section becoming visible → trigger sequential reveal
    (function watchResultSection() {
        const resultSection = document.getElementById('result-section');
        if (!resultSection) return;

        const obs = new MutationObserver((mutations) => {
            mutations.forEach(m => {
                if (m.type === 'attributes' && m.attributeName === 'class') {
                    const wasHidden = m.oldValue && m.oldValue.includes('hidden');
                    const isNowVisible = !resultSection.classList.contains('hidden');
                    if (wasHidden && isNowVisible) {
                        // Small timeout to let app.js populate DOM first
                        setTimeout(() => window.animateResultReveal(), 50);
                    }
                }
            });
        });
        obs.observe(resultSection, { attributes: true, attributeFilter: ['class'], attributeOldValue: true });
    })();

    // Watch docResultContainer becoming visible → trigger doc animations
    (function watchDocResult() {
        const docContainer = document.getElementById('docResultContainer');
        if (!docContainer) return;

        const obs = new MutationObserver((mutations) => {
            mutations.forEach(m => {
                if (m.type === 'attributes' && m.attributeName === 'class') {
                    const wasHidden = m.oldValue && m.oldValue.includes('hidden');
                    const isNowVisible = !docContainer.classList.contains('hidden');
                    if (wasHidden && isNowVisible) {
                        setTimeout(() => window.animateDocResult(), 60);
                    }
                }
            });
        });
        obs.observe(docContainer, { attributes: true, attributeFilter: ['class'], attributeOldValue: true });
    })();

    // Watch multiplePlanetsNotice becoming visible
    (function watchMultiPlanet() {
        const notice = document.getElementById('multiplePlanetsNotice');
        if (!notice) return;

        const obs = new MutationObserver((mutations) => {
            mutations.forEach(m => {
                if (m.type === 'attributes' && m.attributeName === 'class') {
                    const wasHidden = m.oldValue && m.oldValue.includes('hidden');
                    const isNowVisible = !notice.classList.contains('hidden');
                    if (wasHidden && isNowVisible) {
                        window.animateMultiplePlanets(true);
                    } else if (!wasHidden && !isNowVisible) {
                        window.animateMultiplePlanets(false);
                    }
                }
            });
        });
        obs.observe(notice, { attributes: true, attributeFilter: ['class'], attributeOldValue: true });
    })();

    // Watch file input changes → trigger file-selected animation
    (function watchFileInput() {
        const docFileInput = document.getElementById('docFileInput');
        if (!docFileInput) return;

        docFileInput.addEventListener('change', () => {
            if (docFileInput.files.length > 0) {
                window.animateFileSelected();
            }
        });

        // Also hook drop event on dropzone
        const dropzone = document.getElementById('uploadDropzone');
        if (dropzone) {
            dropzone.addEventListener('drop', () => {
                setTimeout(() => window.animateFileSelected(), 50);
            });
        }
    })();

    // Watch docMatchStatus for catalogue match reveal
    (function watchCatalogueMatch() {
        const matchStatus = document.getElementById('docMatchStatus');
        if (!matchStatus) return;

        const obs = new MutationObserver(() => {
            window.animateCatalogueVerify(matchStatus);
        });
        obs.observe(matchStatus, { childList: true, subtree: true });
    })();

    /* ------------------------------------------------------------------
       12. INITIALISE ON DOM READY
    ------------------------------------------------------------------ */
    function init() {
        initScrollReveal();
        initChipEntrance();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();

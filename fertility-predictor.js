// Feve Fertility Predictor
// Self-calibrating Bayesian fertility predictor for the Feve app
// Uses period starts, BBT, and mucus observations

const FEVE_FERTILITY = {
    priorOvRange: { min: 8, max: 38 },
    defaultFollicular: 16.9,
    defaultLuteal: 12.4,
    minFollicularStd: 2.0,
    defaultFollicularStd: 5.3,
    bbtSigma: 0.15,
    defaultPreBaseline: 97.3,
    defaultPostBaseline: 97.6,
    fertileSpermDays: 5,
    fertileEggDays: 0,
    highFertilityThresh: 0.30,
    peakFertilityThresh: 0.60,
    tiny: 1e-10,
    recentWeightAlpha: 0.70,
    baselineAlpha: 0.22,
    mucusPriorCount: 1.0,
    mucusNearWindow: { min: -5, max: 2 },
    inferOvProbThresh: 0.35,
    maxHistory: 12,
    minCycleLength: 21,
    maxCycleLength: 42,
    minTemp: 95.0,
    maxTemp: 100.0,
    maxCycleDay: 50,
    // Period-length prediction defaults (Normal-Inverse-Gamma prior)
    defaultPeriodLength: 5,
    defaultPeriodLengthStd: 1.5,
    priorPeriodLengthKappa: 1,   // prior sample size for the mean
    priorPeriodLengthNu: 3,      // prior degrees of freedom for variance
    minPeriodLength: 2,
    maxPeriodLength: 10,
    debug: false // Set to true for development logging
};

// Debug logging helper
function feveDebug(...args) {
    if (FEVE_FERTILITY.debug) {
        console.log('[Feve Fertility]', ...args);
    }
}

// Validation
// Returns true when `day` (cycle day 1..maxCycleDay=50) is a usable integer.
// Guards every loop/lookup keyed by cycle day — callers can trust the value.
function feveIsValidDay(day) {
    const d = Number(day);
    return !isNaN(d) && d >= 1 && d <= FEVE_FERTILITY.maxCycleDay;
}

// Returns true when `temp` (°F) is within a physiologically plausible range
// (95.0–100.0 °F). Filters out typos like 9.7 or 973 before they poison
// the BBT baselines/likelihoods.
function feveIsValidTemperature(temp) {
    const t = Number(temp);
    return !isNaN(t) && t >= FEVE_FERTILITY.minTemp && t <= FEVE_FERTILITY.maxTemp;
}

// Returns true when `len` (whole-cycle days) is in the plausible band
// 21–42 days (WHO "normal" range). Used to vet recorded cycles and to
// reject nonsense intervals between consecutive period starts.
function feveIsValidCycleLength(len) {
    const l = Number(len);
    return !isNaN(l) && l >= FEVE_FERTILITY.minCycleLength && l <= FEVE_FERTILITY.maxCycleLength;
}

// Returns true when `len` (bleeding days) is plausible: 2–10 days.
// Guards the period-length history that feeds the Bayesian length model.
function feveIsValidPeriodLength(len) {
    const l = Number(len);
    return !isNaN(l) && l >= FEVE_FERTILITY.minPeriodLength && l <= FEVE_FERTILITY.maxPeriodLength;
}

// Math helpers
// Arithmetic mean of an array; null on empty input. Used for baselines
// and residual (MAD) estimation.
function feveMean(arr) {
    if (!arr?.length) return null;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}

// Weighted arithmetic mean: Σ(v·w)/Σw. The recency-weighted workhorse —
// recent cycles get geometrically larger weights (recentWeightAlpha),
// so the estimate follows the user's evolving rhythm.
function feveWeightedMean(values, weights) {
    if (!values?.length || !weights?.length || values.length !== weights.length) return null;
    const wsum = weights.reduce((s, w) => s + w, 0);
    if (wsum <= FEVE_FERTILITY.tiny) return null;
    const sum = values.reduce((s, v, i) => s + v * weights[i], 0);
    return sum / wsum;
}

// Unweighted sample standard deviation (n−1 denominator). Falls back to
// defaultFollicularStd when there are <2 values. Used to gauge how
// regular (or irregular) the user's cycles are.
function feveStdSample(arr) {
    if (!arr || arr.length <= 1) return FEVE_FERTILITY.defaultFollicularStd;
    const m = feveMean(arr);
    if (m === null) return FEVE_FERTILITY.defaultFollicularStd;
    const v = arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1);
    return Math.sqrt(v);
}

// Weighted sample std-dev with a Bessel-style correction for the
// weights themselves (denominator adjusts by Σw²/Σw²). Used by the
// period-length Bayesian model where recent periods count more.
function feveWeightedStdSample(arr, weights) {
    if (!arr || arr.length <= 1 || !weights || weights.length !== arr.length) {
        return FEVE_FERTILITY.defaultPeriodLengthStd;
    }
    const mean = feveWeightedMean(arr, weights);
    if (mean === null) return FEVE_FERTILITY.defaultPeriodLengthStd;
    const wsum = weights.reduce((s, w) => s + w, 0);
    if (wsum <= FEVE_FERTILITY.tiny) return FEVE_FERTILITY.defaultPeriodLengthStd;
    // Weighted sample variance (Bessel-corrected approximation)
    const num = arr.reduce((s, x, i) => s + weights[i] * (x - mean) ** 2, 0);
    const denom = wsum * (1 - weights.reduce((s, w) => s + w * w, 0) / (wsum * wsum));
    if (denom <= FEVE_FERTILITY.tiny) return FEVE_FERTILITY.defaultPeriodLengthStd;
    return Math.sqrt(num / denom);
}

// Normalizes a {key: weight} map so values sum to 1 — turning raw
// scores into a probability distribution. On a zero/near-zero sum it
// falls back to a UNIFORM distribution (all keys equal).
function feveNormalizeDict(dict) {
    const keys = Object.keys(dict);
    if (keys.length === 0) return {};
    const sum = Object.values(dict).reduce((s, v) => s + v, 0);
    if (sum < FEVE_FERTILITY.tiny) {
        const u = 1 / keys.length;
        return Object.fromEntries(keys.map(k => [k, u]));
    }
    return Object.fromEntries(keys.map(k => [k, dict[k] / sum]));
}

// Storage with validation
// Persists the learning state (cycleHistory, inferredFollicularHistory,
// periodLengthHistory, baselines, bbtSigma, mucusLRs) to the encrypted
// store under 'fema_fertility_state'. Failures are logged, never thrown —
// a storage hiccup must not break predictions.
function feveSaveFertilityState(state) {
    try {
        secureSetItem('fema_fertility_state', state);
        feveDebug('State saved:', state);
    } catch (e) {
        console.warn("feveSaveFertilityState failed", e);
    }
}

// Loads + sanitizes the learning state (see note above).
function feveLoadFertilityState() {
    try {
        const raw = secureGetItem('fema_fertility_state');
        if (!raw) return null;

        // secureGetItem may already return a parsed object (femia.js decrypts
        // and JSON.parse's encrypted values). Handle both string and object.
        const state = (typeof raw === 'string') ? JSON.parse(raw) : raw;
        
        // Validate and clean state structure
        if (state && typeof state === 'object') {
            // Ensure arrays exist and are valid
            state.cycleHistory = Array.isArray(state.cycleHistory) 
                ? state.cycleHistory.filter(len => feveIsValidCycleLength(len))
                : [];
            
            state.inferredFollicularHistory = Array.isArray(state.inferredFollicularHistory) 
                ? state.inferredFollicularHistory.filter(day => feveIsValidDay(day))
                : [];

            state.periodLengthHistory = Array.isArray(state.periodLengthHistory)
                ? state.periodLengthHistory.filter(len => feveIsValidPeriodLength(len))
                : [];
            
            // Ensure baselines are valid numbers
            if (state.preBaseline !== undefined && !feveIsValidTemperature(state.preBaseline)) {
                delete state.preBaseline;
            }
            if (state.postBaseline !== undefined && !feveIsValidTemperature(state.postBaseline)) {
                delete state.postBaseline;
            }
            
            feveDebug('State loaded:', state);
        }
        
        return state;
    } catch (e) {
        console.warn("feveLoadFertilityState failed, resetting state", e);
        // Return empty state instead of corrupt one
        return {
            cycleHistory: [],
            inferredFollicularHistory: []
        };
    }
}

// Get BBT data for a cycle
// Collects this cycle's BBT readings keyed by CYCLE DAY (1-based, derived
// from each log's date minus cycleStartDate). Source: encrypted
// 'fema_items_data' day entries ({dateKey: {bbt, mucus, ...}}).
function feveGetCycleBbtData(cycleStartDate, cycleLength) {
    const bbtData = {};
    const raw = secureGetItem('fema_items_data') || {};
    const items = (typeof raw === 'string') ? JSON.parse(raw) : raw;
    
    for (const [dateKey, dayItems] of Object.entries(items)) {
        const date = new Date(dateKey);
        const startDate = new Date(cycleStartDate);
        date.setHours(0, 0, 0, 0);
        startDate.setHours(0, 0, 0, 0);
        const diffDays = Math.floor((date - startDate) / (1000 * 60 * 60 * 24)) + 1;
        
        if (diffDays >= 1 && diffDays <= cycleLength) {
            if (dayItems.bbt) {
                bbtData[diffDays] = parseFloat(dayItems.bbt);
            }
        }
    }
    return bbtData;
}

// Get Mucus data for a cycle
// Same shape as BBT collection but for mucus category strings
// (e.g. "eggwhite", "creamy", "dry") keyed by cycle day.
function feveGetCycleMucusData(cycleStartDate, cycleLength) {
    const mucusData = {};
    const raw = secureGetItem('fema_items_data') || {};
    const items = (typeof raw === 'string') ? JSON.parse(raw) : raw;
    
    for (const [dateKey, dayItems] of Object.entries(items)) {
        const date = new Date(dateKey);
        const startDate = new Date(cycleStartDate);
        date.setHours(0, 0, 0, 0);
        startDate.setHours(0, 0, 0, 0);
        const diffDays = Math.floor((date - startDate) / (1000 * 60 * 60 * 24)) + 1;
        
        if (diffDays >= 1 && diffDays <= cycleLength) {
            if (dayItems.mucus) {
                mucusData[diffDays] = dayItems.mucus;
            }
        }
    }
    return mucusData;
}

// Build Gaussian prior on ovulation day
// Builds the PRIOR probability distribution over "which cycle day was
// ovulation" BEFORE looking at this cycle's signs. Strategy:
//   1. ≥3 known follicular lengths (inferred from past BBT shifts) → use them
//   2. else derive follicular ≈ cycleLength − 12.4 (default luteal) from
//      past cycle lengths
//   3. else fall back to the textbook default (16.9 ± 5.3)
// Recent values weigh more (recentWeightAlpha=0.7 geometric decay). The
// spread (std) widens for irregular users and slightly tightens for
// consistent ones — an honest "how sure are we" statement in distribution form.
function feveBuildOvulationPrior(state) {
    const follicHist = state?.inferredFollicularHistory || [];
    const cycleHist = state?.cycleHistory || [];
    let follicValues;

    if (follicHist.length >= 3) {
        follicValues = follicHist;
        feveDebug('Using follicular history for prior:', follicValues);
    } else {
        follicValues = cycleHist
            .map(len => len - FEVE_FERTILITY.defaultLuteal)
            .filter(v => v >= FEVE_FERTILITY.priorOvRange.min && v <= FEVE_FERTILITY.priorOvRange.max);
        feveDebug('Using cycle history for prior:', follicValues);
    }

    follicValues = follicValues.map(v =>
        Math.max(FEVE_FERTILITY.priorOvRange.min, Math.min(FEVE_FERTILITY.priorOvRange.max, v))
    );

    if (follicValues.length === 0) {
        feveDebug('No history, using default prior');
        return feveBuildGaussianPrior(FEVE_FERTILITY.defaultFollicular, FEVE_FERTILITY.defaultFollicularStd);
    }

    const n = follicValues.length;
    const weights = follicValues.map((_, i) => Math.pow(FEVE_FERTILITY.recentWeightAlpha, n - 1 - i));
    const expOv = feveWeightedMean(follicValues, weights);

    if (expOv === null) {
        return feveBuildGaussianPrior(FEVE_FERTILITY.defaultFollicular, FEVE_FERTILITY.defaultFollicularStd);
    }

    let std = Math.max(FEVE_FERTILITY.minFollicularStd, feveStdSample(follicValues) * 1.15);

    if (n >= 6 && std > 3.5) std = Math.max(FEVE_FERTILITY.minFollicularStd, std * 0.85);

    feveDebug('Built prior with center:', Math.round(expOv), 'std:', std);
    return feveBuildGaussianPrior(Math.round(expOv), std);
}

// Turns (center, std) into a discretized Gaussian prior over cycle days
// 8..38 (outside the range → ~0 probability), then normalizes to sum 1.
function feveBuildGaussianPrior(center, std) {
    const prior = {};
    for (let d = 1; d <= FEVE_FERTILITY.maxCycleDay; d++) {
        if (d < FEVE_FERTILITY.priorOvRange.min || d > FEVE_FERTILITY.priorOvRange.max) {
            prior[d] = FEVE_FERTILITY.tiny;
        } else {
            const dev = (d - center) / std;
            prior[d] = Math.exp(-0.5 * dev * dev);
        }
    }
    return feveNormalizeDict(prior);
}

// Mucus likelihood ratios
// Evidence weight ("likelihood ratio") for a mucus observation given a
// hypothesized ovulation day. relDay = observation day − hypothesized
// ovulation day. LR > 1 supports the hypothesis; LR < 1 argues against it.
// Encodes clinical patterning: egg-white/stretchy mucus is strong evidence
// NEAR ovulation (LR 4.2) and mild evidence against far away; dry is
// the reverse. Supports user-learned LRs override later in the caller.
function feveDefaultMucusLR(category, relDay) {
    const c = (category || '').toLowerCase().trim();

    if (c.includes('eggwhite') || c.includes('ew') || c.includes('stretchy') || c.includes('clear')) {
        return (relDay >= -4 && relDay <= 1) ? 4.2 : 0.22;
    }
    if (c.includes('watery') || c.includes('wet')) {
        return (relDay >= -3 && relDay <= 2) ? 2.3 : 0.38;
    }
    if (c.includes('creamy') || c.includes('milky') || c.includes('white')) {
        return (relDay >= -5 && relDay <= 1) ? 1.5 : 0.60;
    }
    if (c.includes('sticky') || c.includes('tacky')) {
        return (Math.abs(relDay) <= 3) ? 0.75 : 1.15;
    }
    if (c.includes('dry') || c === '' || c === 'none' || c === 'nothing') {
        return (relDay <= -6 || relDay >= 3) ? 1.4 : 0.30;
    }
    return 1.0;
}

// Estimate BBT baselines
// Estimates the user's pre- and post-ovulation temperature baselines from
// the BBT curve. Days ≤ referenceDay−8 feed the pre-ovulation average;
// days ≥ referenceDay+1 feed the post-ovulation average (the ±buffer
// skips the fertile/shift zone where temps are in transition). Falls
// back to population defaults when a side has no data.
function feveEstimateBaselines(bbtData, referenceDay = 999) {
    const pre = [], post = [];

    for (const [ds, ts] of Object.entries(bbtData || {})) {
        const day = parseInt(ds);
        const t = parseFloat(ts);
        if (isNaN(day) || isNaN(t) || !feveIsValidDay(day) || !feveIsValidTemperature(t)) continue;

        if (day <= Math.max(1, referenceDay - 8)) pre.push(t);
        if (day >= Math.max(1, referenceDay + 1)) post.push(t);
    }

    const preM = feveMean(pre) ?? FEVE_FERTILITY.defaultPreBaseline;
    const postM = feveMean(post) ?? (preM + 0.3);

    return { preBaseline: preM, postBaseline: postM };
}

// BBT log likelihood
// Average LOG-LIKELIHOOD of the observed BBT curve IF ovulation happened
// on `ovDay`. Model: each temperature is Normally distributed around the
// appropriate baseline (pre-ov before ovDay, post-ov after) with std
// `sigma`. The two-phase step-function of baselines is what lets the
// shift in the curve "pull" the posterior toward the right ovulation day.
function feveBbtLogLikelihood(ovDay, bbtData, baselines, sigma) {
    if (!bbtData || Object.keys(bbtData).length === 0) return 0;

    const s2 = sigma * sigma;
    const logNorm = -0.5 * Math.log(2 * Math.PI * s2);
    let ll = 0;
    let validPoints = 0;

    for (const [ds, ts] of Object.entries(bbtData)) {
        const day = parseInt(ds);
        const t = parseFloat(ts);
        if (isNaN(day) || isNaN(t) || !feveIsValidDay(day) || !feveIsValidTemperature(t)) continue;

        const mu = (day <= ovDay) ? baselines.preBaseline : baselines.postBaseline;
        ll += logNorm - 0.5 * (t - mu) ** 2 / s2;
        validPoints++;
    }

    return validPoints > 0 ? ll / validPoints : 0;
}

// Mucus log likelihood
// Average log-likelihood of the mucus observations under the hypothesis
// "ovulation was on `ovDay`". Each observation contributes log(LR) where
// LR comes from the user's learned ratios (userLRs) or the clinical
// defaults; object-shaped user LRs pick near/far variants by the
// −5..+2 proximity window. More mucus agreement ⇒ higher likelihood.
function feveMucusLogLikelihood(ovDay, mucusData, userLRs) {
    if (!mucusData || Object.keys(mucusData).length === 0) return 0;

    let ll = 0;
    let validPoints = 0;

    for (const [ds, catRaw] of Object.entries(mucusData)) {
        const day = parseInt(ds);
        if (isNaN(day) || !feveIsValidDay(day)) continue;

        const rel = day - ovDay;
        const cat = (catRaw || '').toLowerCase().trim();
        let lr = userLRs?.[cat] ?? feveDefaultMucusLR(cat, rel);

        if (typeof lr === 'object') {
            const inWindow = (rel >= FEVE_FERTILITY.mucusNearWindow.min && rel <= FEVE_FERTILITY.mucusNearWindow.max);
            lr = inWindow ? (lr.nearLR ?? 1) : (lr.farLR ?? 1);
        }

        ll += Math.log(Math.max(lr, FEVE_FERTILITY.tiny));
        validPoints++;
    }

    return validPoints > 0 ? ll / validPoints : 0;
}

// Calculate fertility probability
// Convert the posterior over ovulation days into "how fertile is TODAY".
// Sums the probability that ovulation lands within the next 5 days
// (sperm lifespan) through today (egg viability) — the classic fertile
// window. Then:
//   • scales by a confidence factor (0.7 initial → 1.0 high) so honest
//     uncertainty tempers early estimates,
//   • guarantees the posterior's most-likely ovulation day reads ≥60%
//     (day before: 50%, +2: 35%) so the UI communicates the peak,
//   • applies biological caps near cycle extremes (day < 6 ⇒ ≤10%,
//     day > 30 ⇒ ≤10%...),
//   • floors the result at ≤95% (never "certain").
function feveCalculateFertilityProbability(posterior, cycleDay, confidenceLevel = "initial") {
    // If no posterior or invalid day, return 0
    if (!posterior || Object.keys(posterior).length === 0 || !feveIsValidDay(cycleDay)) {
        return 0;
    }
    
    let fertileProb = 0;
    
    // Fertile window: 5 days before ovulation through ovulation day
    for (let ovDay in posterior) {
        ovDay = Number(ovDay);
        const probOv = posterior[ovDay] || 0;
        
        // If today is within 5 days before ovulation, today is fertile
        if (cycleDay <= ovDay && cycleDay >= ovDay - 5) {
            fertileProb += probOv;
        }
    }
    
    // Find the day with maximum posterior probability (most likely ovulation day)
    let maxProbDay = null;
    let maxProb = 0;
    for (let day in posterior) {
        if (posterior[day] > maxProb) {
            maxProb = posterior[day];
            maxProbDay = Number(day);
        }
    }
    
    // Apply confidence-based adjustment
    let adjustmentFactor = 1.0;
    switch(confidenceLevel) {
        case "initial":
            adjustmentFactor = 0.7; // No personal data
            break;
        case "low":
            adjustmentFactor = 0.85; // Some data
            break;
        case "medium":
            adjustmentFactor = 0.95; // Good data
            break;
        case "high":
            adjustmentFactor = 1.0; // Trust the data
            break;
        default:
            adjustmentFactor = 0.8;
    }
    
    fertileProb = fertileProb * adjustmentFactor;
    
    // Ensure the peak day gets highest probability
    if (maxProbDay !== null) {
        const daysFromOv = maxProbDay - cycleDay;
        
        // If this is the ovulation day or days near it, ensure it's not capped too low
        if (daysFromOv === 0 && fertileProb < 0.60) {
            fertileProb = Math.max(fertileProb, 0.60);
        } else if (daysFromOv === 1 && fertileProb < 0.50) {
            fertileProb = Math.max(fertileProb, 0.50);
        } else if (daysFromOv === 2 && fertileProb < 0.35) {
            fertileProb = Math.max(fertileProb, 0.35);
        }
    }
    
    // Apply biological constraints based on cycle day
    if (cycleDay < 6) {
        fertileProb = Math.min(fertileProb, 0.10);
    } else if (cycleDay < 8) {
        fertileProb = Math.min(fertileProb, 0.20);
    } else if (cycleDay > 30) {
        fertileProb = Math.min(fertileProb, 0.10);
    } else if (cycleDay > 25) {
        fertileProb = Math.min(fertileProb, 0.20);
    }
    
    // Ensure probability doesn't exceed 95%
    return Math.min(fertileProb, 0.95);
}

// Get confidence level based on data and posterior sharpness
// Maps posterior sharpness (max probability) + data presence to a
// coarse confidence tier consumed by both the fertility-probability
// scaling and the UI: "initial" (no data), "low", "medium", "high".
function feveGetConfidenceLevel(confidence, hasData) {
    if (!hasData) return "initial";
    if (confidence < 0.1) return "low";
    if (confidence < 0.25) return "medium";
    return "high";
}

// Core computation
// ═══ THE CORE BAYESIAN MACHINE ═══
// Inputs: cycleDay (today, 1-based), plus this cycle's bbtData/mucusData
// (keyed by cycle day). Pipeline:
//   1. PRIOR — feveBuildOvulationPrior(userState): P(ovulation day) from
//      history or textbook defaults.
//   2. BASELINES — pre/post-ovulation temperatures estimated from this
//      cycle's curve (or the user's stored long-run baselines).
//   3. LIKELIHOOD — for EVERY candidate ovulation day d in 8..38:
//         log P(d|prior) + log L(BBT|d) + log L(mucus|d)
//      computed in log-space (numerically stable).
//   4. POSTERIOR — exponentiate the shifted logs (max-subtraction trick),
//      normalize to a probability distribution over ovulation days.
//   5. OUTPUT — most likely ovulation day, its confidence, today's
//      fertility probability, and human-readable flags/messages.
// Every failure path returns a well-formed result object ("error") so the
// app UI always has something coherent to render.
function feveComputePosterior(inputs, userState = {}) {
    const cycleDay = parseInt(inputs.cycleDay) || 1;
    if (!feveIsValidDay(cycleDay)) {
        feveDebug('Invalid cycle day:', cycleDay);
        return feveMakeResult([], null, 0, 0, "Invalid cycle day", {}, FEVE_FERTILITY.bbtSigma, false, false, "error");
    }

    const bbtData = inputs.bbtData || {};
    const mucusData = inputs.mucusData || {};
    
    feveDebug('Computing posterior for cycle day:', cycleDay);
    feveDebug('BBT data points:', Object.keys(bbtData).length);
    feveDebug('Mucus data points:', Object.keys(mucusData).length);

    const prior = feveBuildOvulationPrior(userState);

    const roughOv = feveWeightedMean(
        Object.keys(prior).map(Number),
        Object.values(prior)
    ) || FEVE_FERTILITY.defaultFollicular;

    const baselines = {
        preBaseline: userState.preBaseline ?? FEVE_FERTILITY.defaultPreBaseline,
        postBaseline: userState.postBaseline ?? (FEVE_FERTILITY.defaultPreBaseline + 0.3)
    };

    const estBases = feveEstimateBaselines(bbtData, roughOv + 5);
    baselines.preBaseline = estBases.preBaseline;
    baselines.postBaseline = estBases.postBaseline;

    const sigma = userState.bbtSigma ?? FEVE_FERTILITY.bbtSigma;
    const userLRs = userState.mucusLRs ?? null;

    const days = Object.keys(prior)
        .map(Number)
        .filter(d => d >= FEVE_FERTILITY.priorOvRange.min && d <= FEVE_FERTILITY.priorOvRange.max)
        .sort((a, b) => a - b);

    const logUnnorm = {};
    let maxL = -Infinity;

    for (const d of days) {
        const logP = Math.log(Math.max(prior[d], FEVE_FERTILITY.tiny));
        const logB = feveBbtLogLikelihood(d, bbtData, baselines, sigma);
        const logM = feveMucusLogLikelihood(d, mucusData, userLRs);
        const total = logP + logB + logM;
        logUnnorm[d] = total;
        if (total > maxL) maxL = total;
    }

    if (!isFinite(maxL)) {
        const postArr = Object.entries(prior)
            .map(([k, v]) => ({ day: Number(k), prob: v }))
            .sort((a, b) => a.day - b.day);
        return feveMakeResult(postArr, null, 0, 0, "Limited data — using prior", baselines, sigma, false, false, "initial");
    }

    const unnorm = {};
    for (const d of days) {
        unnorm[d] = Math.exp(logUnnorm[d] - maxL);
    }

    const total = Object.values(unnorm).reduce((s, v) => s + v, 0);
    const posterior = total > FEVE_FERTILITY.tiny ? feveNormalizeDict(unnorm) : feveNormalizeDict(prior);

    const postArr = Object.entries(posterior)
        .map(([k, v]) => ({ day: Number(k), prob: v }))
        .sort((a, b) => a.day - b.day);

    const best = postArr.reduce((a, c) => c.prob > a.prob ? c : a, { day: null, prob: 0 });
    const predOv = best.day;
    const conf = best.prob;

    // Determine confidence level
    const hasData = Object.keys(bbtData).length > 0 || Object.keys(mucusData).length > 0;
    const confidenceLevel = feveGetConfidenceLevel(conf, hasData);

    // Calculate fertility probability using corrected method
    const fertileProb = feveCalculateFertilityProbability(posterior, cycleDay, confidenceLevel);

    const high = fertileProb >= FEVE_FERTILITY.highFertilityThresh;
    const peak = fertileProb >= FEVE_FERTILITY.peakFertilityThresh;

    let message = "Low fertility today";
    if (peak) {
        message = "Peak fertility - highest chance of conception";
    } else if (high) {
        message = "High fertility - fertile window";
    } else if (fertileProb > 0.15) {
        message = "Possible fertile window";
    } else if (fertileProb > 0.05) {
        message = "Low fertility";
    } else {
        message = "Very low fertility";
    }

    return feveMakeResult(postArr, predOv, conf, fertileProb, message, baselines, sigma, high, peak, confidenceLevel);
}

// Assembles the standardized prediction envelope consumed across the
// app (calendar shading, insights charts, banners). Percentages are
// precomputed; hasPersonalData distinguishes "generic" from
// "personalized" predictions (also drives UI copy).
function feveMakeResult(postArr, predOv, conf, fertileProb, message, baselines, sigma, isHigh = false, isPeak = false, confidenceLevel = "initial") {
    return {
        posterior: postArr,
        predictedOvulationDay: predOv,
        confidence: conf,
        confidenceLevel: confidenceLevel,
        fertileProbabilityToday: fertileProb,
        fertilePercent: Math.round(fertileProb * 100),
        isHighFertilityToday: isHigh,
        isPeakFertilityToday: isPeak,
        message,
        baselinesUsed: baselines,
        sigmaUsed: sigma,
        hasPersonalData: confidenceLevel !== "initial"
    };
}

// Load period data for prediction
// If excludeAfterDate is provided (e.g. pregnancy birth date), periods on or
// after that date are excluded from both the start list and the average.
// Reads the period log ({dateKey: {isStart, isPeriod, length...}}) and
// summarizes it for the predictor: the latest start, ALL starts (sorted),
// and the average cycle length (mean of intervals between consecutive
// starts; invalid intervals are skipped). excludeAfterDate (e.g. a
// pregnancy) truncates history so pre-pregnancy cycles don't pollute
// post-partum predictions. Defaults: avgCycle 28, empty lists.
function feveLoadPeriodDataForPrediction(excludeAfterDate = null) {
    const periodData = secureGetItem('fema_period_data');
    if (!periodData) return { lastPeriodStart: null, avgCycle: 28, allPeriodStarts: [] };

    const cutoff = excludeAfterDate ? new Date(excludeAfterDate) : null;

    try {
        let starts = Object.keys(periodData)
            .filter(key => periodData[key] && periodData[key].isStart)
            .sort();

        if (cutoff) {
            starts = starts.filter(start => new Date(start) < cutoff);
        }

        if (starts.length === 0) return { lastPeriodStart: null, avgCycle: 28, allPeriodStarts: [] };

        // Calculate average cycle length if we have multiple starts
        let avgCycle = 28;
        if (starts.length >= 2) {
            const lengths = [];
            for (let i = 1; i < starts.length; i++) {
                const prev = new Date(starts[i-1]);
                const curr = new Date(starts[i]);
                const days = Math.round((curr - prev) / (1000 * 60 * 60 * 24));
                if (feveIsValidCycleLength(days)) {
                    lengths.push(days);
                }
            }
            if (lengths.length > 0) {
                avgCycle = Math.round(lengths.reduce((a, b) => a + b, 0) / lengths.length);
            }
        }

        return {
            lastPeriodStart: starts[starts.length - 1],
            allPeriodStarts: starts,
            avgCycle: avgCycle
        };
    } catch (e) {
        console.warn("Error loading period data:", e);
        return { lastPeriodStart: null, avgCycle: 28, allPeriodStarts: [] };
    }
}

// Update state when new period is recorded
// LEARNING step — called when a cycle completes. Updates the encrypted
// learning state:
//   • cycleHistory += previousCycleLength (capped at maxHistory=12)
//   • if ≥6 BBT points exist, infers follicular length
//     (cycleLength − defaultLuteal) and appends it
//   • blends baselines toward this cycle's estimates (EMA, alpha=0.22)
//   • re-estimates the BBT measurement noise (sigma) from residuals via
//     a MAD-based estimator, blended the same way
// This is what makes the predictor self-calibrating over months of use.
function feveRecordNewPeriod(periodData) {
    let state = feveLoadFertilityState() || {
        cycleHistory: [],
        inferredFollicularHistory: []
    };

    // Check if previousCycleLength exists
    const len = periodData.previousCycleLength ? parseInt(periodData.previousCycleLength) : null;
    if (len !== null && !isNaN(len) && feveIsValidCycleLength(len)) {
        state.cycleHistory = (state.cycleHistory || []).concat(len).slice(-FEVE_FERTILITY.maxHistory);
        feveDebug('Added cycle length to history:', len);
    }

    // Learn from completed cycle BBT data
    if (periodData.bbtData && Object.keys(periodData.bbtData).length >= 6) {
        // Estimate ovulation from BBT shift
        const roughOv = Math.round(periodData.cycleLength - FEVE_FERTILITY.defaultLuteal);
        
        if (roughOv >= 9 && roughOv <= periodData.cycleLength - 5) {
            state.inferredFollicularHistory = (state.inferredFollicularHistory || [])
                .concat(roughOv)
                .slice(-FEVE_FERTILITY.maxHistory);
            feveDebug('Added inferred follicular length:', roughOv);

            // Update baselines
            const bases = feveEstimateBaselines(periodData.bbtData, roughOv);
            const ba = state.baselineAlpha ?? FEVE_FERTILITY.baselineAlpha;

            state.preBaseline = state.preBaseline !== undefined
                ? (1 - ba) * state.preBaseline + ba * bases.preBaseline
                : bases.preBaseline;

            state.postBaseline = state.postBaseline !== undefined
                ? (1 - ba) * state.postBaseline + ba * bases.postBaseline
                : bases.postBaseline;

            // Compute BBT sigma from residuals
            const residuals = [];
            for (const [ds, ts] of Object.entries(periodData.bbtData)) {
                const day = parseInt(ds);
                const t = parseFloat(ts);
                if (isNaN(day) || isNaN(t)) continue;
                const mu = (day <= roughOv) ? state.preBaseline : state.postBaseline;
                residuals.push(Math.abs(t - mu));
            }

            if (residuals.length >= 6) {
                const mad = feveMean(residuals);
                if (mad !== null) {
                    const estSigma = Math.max(0.09, mad / 0.8);
                    state.bbtSigma = state.bbtSigma !== undefined
                        ? (1 - ba) * state.bbtSigma + ba * estSigma
                        : estSigma;
                    feveDebug('Updated BBT sigma:', state.bbtSigma);
                }
            }
        }
    }

    feveSaveFertilityState(state);
    return state;
}

// Format date key with validation
// Date → 'YYYY-MM-DD' (local timezone, no UTC shifting). Returns null and
// logs on invalid input — all day-key arithmetic depends on this format.
function feveFormatDateKey(date) {
    if (!date || isNaN(new Date(date).getTime())) {
        console.error("Invalid date provided to feveFormatDateKey");
        return null;
    }
    const d = new Date(date);
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

// Simple prediction cache
const predictionCache = new Map();

// 5-second memoization of predictions keyed by
// "{cycleDay}_{bbtKeys}_{mucusKeys}". The calendar re-queries the same
// days on every render; the cache keeps that cheap. Expired/absent →
// null (caller recomputes). predictionCache.clear() invalidates (testing).
function feveGetCachedPrediction(cacheKey, inputs, state) {
    if (predictionCache.has(cacheKey)) {
        const cached = predictionCache.get(cacheKey);
        if (Date.now() - cached.timestamp < 5000) { // 5 second cache
            feveDebug('Using cached prediction');
            return cached.result;
        }
    }
    return null;
}

// Public API
const FeveFertilityPredictor = {
    getPrediction(cycleDay, bbtData = {}, mucusData = {}) {
        try {
            const state = feveLoadFertilityState() || {};
            return feveComputePosterior({ cycleDay, bbtData, mucusData }, state);
        } catch (error) {
            console.error("Feve Fertility Prediction Error:", error);
            return {
                posterior: [],
                predictedOvulationDay: null,
                confidence: 0,
                confidenceLevel: "error",
                fertileProbabilityToday: 0,
                fertilePercent: 0,
                isHighFertilityToday: false,
                isPeakFertilityToday: false,
                message: "Prediction unavailable",
                baselinesUsed: {},
                sigmaUsed: FEVE_FERTILITY.bbtSigma,
                hasPersonalData: false
            };
        }
    },

    getPredictionForDate(date, referencePeriodStart = null) {
        try {
            const state = feveLoadFertilityState() || {};
            const periodData = feveLoadPeriodDataForPrediction();
            
            // Check if we have any period start dates
            if (!periodData.lastPeriodStart) {
                feveDebug('No period start date found');
                return {
                    ...this.getPrediction(1),
                    error: "need_period_start",
                    message: "Please log when your last period started",
                    needsPeriodStart: true
                };
            }

            const currentDate = new Date(date);
            currentDate.setHours(0, 0, 0, 0);
            
            // Find the correct period start for this date
            let periodStart = referencePeriodStart;
            if (!periodStart) {
                for (const start of periodData.allPeriodStarts) {
                    const startDate = new Date(start + 'T00:00:00');
                    if (startDate <= currentDate) {
                        periodStart = start;
                    } else {
                        break;
                    }
                }
            }
            
            // If no period start found for this date, check if date is before first period
            if (!periodStart) {
                const firstPeriod = new Date(periodData.allPeriodStarts[0] + 'T00:00:00');
                if (currentDate < firstPeriod) {
                    feveDebug('Date is before first period');
                    return {
                        ...this.getPrediction(1),
                        error: "date_before_period",
                        message: "This date is before your first recorded period"
                    };
                }
            }
            
            if (!periodStart) {
                periodStart = periodData.lastPeriodStart;
            }
            
            const periodDate = new Date(periodStart + 'T00:00:00');
            
            const daysSincePeriod = Math.floor((currentDate - periodDate) / (1000 * 60 * 60 * 24)) + 1;
            
            // Check if cycle is too long
            if (daysSincePeriod > FEVE_FERTILITY.maxCycleDay) {
                feveDebug('Cycle day exceeds maximum:', daysSincePeriod);
                return {
                    ...this.getPrediction(FEVE_FERTILITY.maxCycleDay),
                    error: "cycle_too_long",
                    message: "This seems longer than a typical cycle - please verify your period start date",
                    cycleDay: daysSincePeriod
                };
            }

            const bbtData = {};
            const mucusData = {};
            const items = secureGetItem('fema_items_data') || {};
            
            // Get BBT and mucus for this cycle
            for (const [dateKey, dayItems] of Object.entries(items)) {
                const itemDate = new Date(dateKey);
                if (itemDate >= periodDate && itemDate <= currentDate) {
                    const dayNum = Math.floor((itemDate - periodDate) / (1000 * 60 * 60 * 24)) + 1;
                    if (dayItems.bbt) {
                        const temp = parseFloat(dayItems.bbt);
                        if (feveIsValidTemperature(temp)) {
                            bbtData[dayNum] = temp;
                        }
                    }
                    if (dayItems.mucus) {
                        mucusData[dayNum] = dayItems.mucus;
                    }
                }
            }

            feveDebug('Calculating prediction for cycle day:', daysSincePeriod);
            
            // Generate cache key
            const cacheKey = `${daysSincePeriod}_${Object.keys(bbtData).join('_')}_${Object.keys(mucusData).join('_')}`;
            const cached = feveGetCachedPrediction(cacheKey, { cycleDay: daysSincePeriod, bbtData, mucusData }, state);
            if (cached) return cached;
            
            const result = feveComputePosterior({ 
                cycleDay: daysSincePeriod, 
                bbtData, 
                mucusData 
            }, state);
            
            // Cache result
            predictionCache.set(cacheKey, {
                result,
                timestamp: Date.now()
            });
            
            return result;
        } catch (error) {
            console.error("Error in getPredictionForDate:", error);
            return {
                ...this.getPrediction(1),
                error: "calculation_error",
                message: "Unable to calculate prediction"
            };
        }
    },

    recordCompletedCycle(cycleLength, bbtData = {}) {
        return feveRecordNewPeriod({ 
            previousCycleLength: cycleLength, 
            bbtData, 
            cycleLength 
        });
    },

    saveBbtReading(date, temperature) {
        const temp = parseFloat(temperature);
        if (!feveIsValidTemperature(temp)) {
            console.warn('Invalid temperature:', temperature);
            return false;
        }
        
        const key = feveFormatDateKey(date);
        if (!key) return false;
        
        let items = secureGetItem('fema_items_data') || {};
        if (!items[key]) items[key] = {};
        items[key].bbt = temp.toString();
        secureSetItem('fema_items_data', items);
        feveDebug('Saved BBT for', key, temp);
        return true;
    },

    saveMucusObservation(date, observation) {
        const key = feveFormatDateKey(date);
        if (!key) return false;
        
        let items = secureGetItem('fema_items_data') || {};
        if (!items[key]) items[key] = {};
        
        if (observation === null || observation === undefined || observation === '') {
            delete items[key].mucus;
        } else {
            items[key].mucus = observation;
        }
        
        secureSetItem('fema_items_data', items);
        feveDebug('Saved mucus for', key, observation);
        return true;
    },

    saveTemperature(date, temperature, unit = 'fahrenheit') {
        const key = feveFormatDateKey(date);
        if (!key) return false;
        
        let items = secureGetItem('fema_items_data') || {};
        if (!items[key]) items[key] = {};
        
        if (temperature === null || temperature === undefined || temperature === '') {
            delete items[key].bbt;
        } else {
            items[key].bbt = temperature;
        }
        
        secureSetItem('fema_items_data', items);
        feveDebug('Saved temperature for', key, temperature);
        return true;
    },

    getFertilityLevel(probability) {
        if (probability >= FEVE_FERTILITY.peakFertilityThresh) return 'peak';
        if (probability >= FEVE_FERTILITY.highFertilityThresh) return 'high';
        if (probability > 0.15) return 'medium';
        if (probability > 0.05) return 'low';
        return 'very low';
    },

    getCycleDayForDate(date, referencePeriodStart = null) {
        const periodData = feveLoadPeriodDataForPrediction();
        if (!periodData.lastPeriodStart) return null;

        const currentDate = new Date(date);
        currentDate.setHours(0, 0, 0, 0);

        let periodStart = referencePeriodStart;
        if (!periodStart) {
            // Find the most recent period start that is not after the target date
            for (const start of periodData.allPeriodStarts) {
                const startDate = new Date(start + 'T00:00:00');
                if (startDate <= currentDate) {
                    periodStart = start;
                } else {
                    break;
                }
            }
        }
        if (!periodStart) return null;

        const startDate = new Date(periodStart + 'T00:00:00');
        const daysSincePeriod = Math.floor((currentDate - startDate) / (1000 * 60 * 60 * 24)) + 1;
        return daysSincePeriod;
    },

    getPredictedCycleLength(excludeAfterDate = null) {
        const periodData = feveLoadPeriodDataForPrediction(excludeAfterDate);
        return periodData.avgCycle;
    },

    // Bayesian period-length prediction using a Normal-Inverse-Gamma prior.
    // Observed period lengths are stored in state.periodLengthHistory.
    // Returns the posterior predictive mean (point estimate) and, optionally,
    // a credible interval for the next period length.
    _getPeriodLengthPosterior() {
        const state = feveLoadFertilityState() || {};
        const history = (state.periodLengthHistory || []).slice(-FEVE_FERTILITY.maxHistory);

        const prior = {
            mu0: FEVE_FERTILITY.defaultPeriodLength,
            kappa0: FEVE_FERTILITY.priorPeriodLengthKappa,
            nu0: FEVE_FERTILITY.priorPeriodLengthNu,
            sigmaSq0: FEVE_FERTILITY.defaultPeriodLengthStd ** 2
        };

        if (history.length === 0) {
            return { muN: prior.mu0, kappaN: prior.kappa0, nuN: prior.nu0, sigmaSqN: prior.sigmaSq0, history: [] };
        }

        // Recency weighting: recent periods influence the estimate more.
        const weights = history.map((_, i) =>
            Math.pow(FEVE_FERTILITY.recentWeightAlpha, history.length - 1 - i)
        );
        const mean = feveWeightedMean(history, weights);
        const variance = history.length > 1
            ? feveWeightedStdSample(history, weights) ** 2
            : prior.sigmaSq0;

        const n = history.length;
        const kappaN = prior.kappa0 + n;
        const nuN = prior.nu0 + n;
        const muN = (prior.kappa0 * prior.mu0 + n * mean) / kappaN;
        const sigmaSqN = (
            prior.nu0 * prior.sigmaSq0 +
            (n - 1) * variance +
            (prior.kappa0 * n / kappaN) * (mean - prior.mu0) ** 2
        ) / nuN;

        return { muN, kappaN, nuN, sigmaSqN, history };
    },

    getPredictedPeriodLength() {
        const posterior = this._getPeriodLengthPosterior();
        let predicted = posterior.muN;
        if (Number.isNaN(predicted) || predicted === null) {
            predicted = FEVE_FERTILITY.defaultPeriodLength;
        }
        // Clamp to biologically plausible range and round to whole days.
        predicted = Math.max(FEVE_FERTILITY.minPeriodLength,
            Math.min(FEVE_FERTILITY.maxPeriodLength, predicted));
        return Math.round(predicted);
    },

    getPredictedPeriodLengthInterval(confidence = 0.95) {
        const posterior = this._getPeriodLengthPosterior();
        const { muN, kappaN, nuN, sigmaSqN } = posterior;
        if (!muN || kappaN <= 0 || nuN <= 0 || sigmaSqN == null) {
            return {
                mean: FEVE_FERTILITY.defaultPeriodLength,
                min: FEVE_FERTILITY.minPeriodLength,
                max: FEVE_FERTILITY.maxPeriodLength
            };
        }

        // Posterior predictive scale for a new observation is sigma * sqrt(1 + 1/kappa).
        const scale = Math.sqrt(sigmaSqN * (1 + 1 / kappaN));

        // Approximate t-distribution quantile with a simple expansion.
        // For small nu (<30), use a correction factor; for larger nu, use normal 1.96.
        let z;
        if (nuN >= 30) {
            z = 1.96;
        } else if (nuN >= 10) {
            z = 2.0 + 0.2 * (30 - nuN) / 20;
        } else if (nuN >= 5) {
            z = 2.2 + 0.3 * (10 - nuN) / 5;
        } else {
            z = 2.8; // conservative for very few observations
        }

        let min = muN - z * scale;
        let max = muN + z * scale;

        // Clamp and round.
        min = Math.max(FEVE_FERTILITY.minPeriodLength, Math.round(min));
        max = Math.min(FEVE_FERTILITY.maxPeriodLength, Math.round(max));
        if (max < min) max = min;

        return { mean: Math.round(muN), min, max };
    },

    recordPeriodLength(length) {
        const l = parseInt(length);
        if (!feveIsValidPeriodLength(l)) {
            console.warn('Invalid period length, not recording:', length);
            return false;
        }
        let state = feveLoadFertilityState() || {
            cycleHistory: [],
            inferredFollicularHistory: [],
            periodLengthHistory: []
        };
        state.periodLengthHistory = (state.periodLengthHistory || [])
            .concat(l)
            .filter(len => feveIsValidPeriodLength(len))
            .slice(-FEVE_FERTILITY.maxHistory);
        feveSaveFertilityState(state);
        feveDebug('Recorded period length:', l, 'history:', state.periodLengthHistory);
        return true;
    },

    // Rebuild the entire period-length history from stored period data.
    // Useful after imports or when the user edits historical periods.
    rebuildPeriodLengthHistory(periodData = null) {
        if (!periodData) {
            periodData = secureGetItem('fema_period_data') || {};
        }
        const starts = Object.keys(periodData)
            .filter(key => periodData[key] && periodData[key].isStart)
            .sort();
        const lengths = [];
        for (const start of starts) {
            const len = periodData[start]?.length;
            if (feveIsValidPeriodLength(len)) {
                lengths.push(len);
            }
        }
        let state = feveLoadFertilityState() || {
            cycleHistory: [],
            inferredFollicularHistory: [],
            periodLengthHistory: []
        };
        state.periodLengthHistory = lengths.slice(-FEVE_FERTILITY.maxHistory);
        feveSaveFertilityState(state);
        return state.periodLengthHistory;
    },

    getPredictedNextPeriodDate(referencePeriodStart = null) {
        const periodData = feveLoadPeriodDataForPrediction();
        if (!periodData.lastPeriodStart) return null;
        const start = referencePeriodStart
            ? new Date(referencePeriodStart + 'T00:00:00')
            : new Date(periodData.lastPeriodStart + 'T00:00:00');
        if (isNaN(start.getTime())) return null;
        const next = new Date(start);
        next.setDate(next.getDate() + periodData.avgCycle);
        return next;
    },

    getPredictedOvulationDate(referencePeriodStart = null) {
        const periodData = feveLoadPeriodDataForPrediction();
        if (!periodData.lastPeriodStart) return null;
        const startKey = referencePeriodStart || periodData.lastPeriodStart;
        const start = new Date(startKey + 'T00:00:00');
        if (isNaN(start.getTime())) return null;
        const prediction = this.getPredictionForDate(start, startKey);
        const ovDay = prediction && prediction.predictedOvulationDay;
        if (!ovDay) return null;
        const ovDate = new Date(start);
        ovDate.setDate(ovDate.getDate() + ovDay - 1);
        return ovDate;
    },

    findPeriodStartForDate(date) {
        const periodData = feveLoadPeriodDataForPrediction();
        if (!periodData.allPeriodStarts.length) return null;
        const target = new Date(date);
        target.setHours(0, 0, 0, 0);
        let best = null;
        for (const start of periodData.allPeriodStarts) {
            const startDate = new Date(start + 'T00:00:00');
            if (startDate <= target) {
                best = start;
            } else {
                break;
            }
        }
        return best;
    },

    getFertilityPercent(date) {
        const prediction = this.getPredictionForDate(date);
        return prediction ? prediction.fertilePercent : 0;
    },

    isFertilityDay(date) {
        return this.getFertilityPercent(date) >= 10;
    },

    isHighFertilityDay(date) {
        const prediction = this.getPredictionForDate(date);
        if (!prediction) return false;
        if (prediction.isHighFertilityToday || prediction.isPeakFertilityToday) return true;
        if (prediction.predictedOvulationDay !== null) {
            const cycleDay = this.getCycleDayForDate(date);
            if (cycleDay > 0 && Math.abs(cycleDay - prediction.predictedOvulationDay) <= 2) return true;
        }
        return false;
    },

    isPeakFertilityDay(date) {
        const prediction = this.getPredictionForDate(date);
        return prediction ? prediction.isPeakFertilityToday : false;
    },

    isInFertileWindow(date) {
        const prediction = this.getPredictionForDate(date);
        if (!prediction || prediction.predictedOvulationDay === null) return false;
        const cycleDay = this.getCycleDayForDate(date);
        if (cycleDay === null || cycleDay < 0) return false;
        const daysFromOv = cycleDay - prediction.predictedOvulationDay;
        return daysFromOv >= -5 && daysFromOv <= 1;
    },

    getCycleMaxFertility(maxDay = null) {
        const periodData = feveLoadPeriodDataForPrediction();
        if (!periodData.lastPeriodStart) return 0;
        const cycleLength = maxDay || FEVE_FERTILITY.maxCycleDay;
        let maxFertility = 0;
        for (let day = 1; day <= cycleLength; day++) {
            const prediction = this.getPrediction(day);
            if (prediction && prediction.fertilePercent > maxFertility) {
                maxFertility = prediction.fertilePercent;
            }
        }
        return maxFertility;
    },

    getPredictedPeriodDays(leadIn = 3, length = 5) {
        const next = this.getPredictedNextPeriodDate();
        if (!next) return [];
        const days = [];
        const start = new Date(next);
        start.setDate(start.getDate() - leadIn);
        for (let i = 0; i < leadIn + length; i++) {
            const d = new Date(start);
            d.setDate(d.getDate() + i);
            days.push(d);
        }
        return days;
    },

    isPredictedPeriodDay(date, windowDays = 7) {
        const next = this.getPredictedNextPeriodDate();
        if (!next) return false;
        const target = new Date(date);
        target.setHours(0, 0, 0, 0);
        const dayDiff = Math.round((target - next) / (1000 * 60 * 60 * 24));
        return Math.abs(dayDiff) <= windowDays;
    },

    loadState: feveLoadFertilityState,
    saveState: feveSaveFertilityState,
    defaults: FEVE_FERTILITY,
    
    // Clear cache (useful for testing)
    clearCache: () => predictionCache.clear()
};

// Export for use in browser
if (typeof window !== 'undefined') {
    window.FeveFertilityPredictor = FeveFertilityPredictor;
    window.FEVE_FERTILITY = FEVE_FERTILITY;
    // Backward-compat aliases — femia.js/index.html still call the old names.
    // Remove once no consumer references them.
    window.FEMAFertilityPredictor = FeveFertilityPredictor;
    window.FEMAFertility = FEVE_FERTILITY;
}
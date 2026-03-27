import { useState, useEffect, useRef } from 'react';
import StrikeZone from './StrikeZone';
import './index.css';

const API_BASE = 'http://localhost:5001/api';

const PITCH_TYPES = [
  'None', '4-Seam Fastball', 'Changeup', 'Curveball', 'Cutter',
  'Knuckle Curve', 'Sinker', 'Slider', 'Slurve', 'Split-Finger', 'Sweeper'
];

/* ── Live Game outcome logic ─────────────────────────────────────── */
function applyOutcome(form, outcome, actualPitchType, actualZone) {
  const f = { ...form };
  
  f.prev2_pitch = f.prev_pitch;
  f.prev2_zone = f.prev_zone;
  f.prev_pitch = actualPitchType || 'None';
  f.prev_zone = actualZone || '';

  const resetCount = () => { f.balls = 0; f.strikes = 0; };
  const resetBatting = () => {
    f.prev_pitch = 'None'; f.prev_zone = '';
    f.prev2_pitch = 'None'; f.prev2_zone = '';
  };
  
  const advanceOut = (n = 1) => {
    f.outs = Math.min(f.outs + n, 3);
    if (f.outs >= 3) {
      f.outs = 0;
      resetCount();
      resetBatting();
      f.on_1b = false; f.on_2b = false; f.on_3b = false;
      f.inning_topbot = f.inning_topbot === 'Top' ? 'Bot' : 'Top';
      if (f.inning_topbot === 'Top') f.inning = Math.min(f.inning + 1, 18);
    } else {
      resetBatting();
    }
  };

  const scoreBattingRun = (runs) => {
    if (f.inning_topbot === 'Top') f.away_score += runs;
    else f.home_score += runs;
  };

  switch (outcome) {
    case 'ball': {
      f.balls += 1;
      if (f.balls >= 4) {
        // Walk
        let runs = 0;
        if (f.on_1b && f.on_2b && f.on_3b) runs = 1;
        if (f.on_3b && f.on_1b && f.on_2b) { f.on_3b = true; }
        if (f.on_2b) f.on_3b = true;
        if (f.on_1b) f.on_2b = true;
        f.on_1b = true;
        scoreBattingRun(runs);
        resetCount();
        resetBatting();
      }
      break;
    }
    case 'called_strike':
    case 'swinging_strike': {
      f.strikes += 1;
      if (f.strikes >= 3) {
        resetCount();
        advanceOut();
      }
      break;
    }
    case 'foul': {
      if (f.strikes < 2) f.strikes += 1;
      break;
    }
    case 'single': {
      let runs = 0;
      if (f.on_3b) { runs++; f.on_3b = false; }
      if (f.on_2b) { f.on_3b = true; f.on_2b = false; }
      if (f.on_1b) { f.on_2b = true; }
      f.on_1b = true;
      scoreBattingRun(runs);
      resetCount();
      resetBatting();
      break;
    }
    case 'double': {
      let runs = 0;
      if (f.on_3b) { runs++; f.on_3b = false; }
      if (f.on_2b) { runs++; f.on_2b = false; }
      if (f.on_1b) { f.on_3b = true; f.on_1b = false; }
      f.on_2b = true;
      scoreBattingRun(runs);
      resetCount();
      resetBatting();
      break;
    }
    case 'triple': {
      let runs = 0;
      if (f.on_3b) runs++;
      if (f.on_2b) runs++;
      if (f.on_1b) runs++;
      f.on_1b = false; f.on_2b = false; f.on_3b = true;
      scoreBattingRun(runs);
      resetCount();
      resetBatting();
      break;
    }
    case 'home_run': {
      let runs = 1;
      if (f.on_1b) runs++;
      if (f.on_2b) runs++;
      if (f.on_3b) runs++;
      f.on_1b = false; f.on_2b = false; f.on_3b = false;
      scoreBattingRun(runs);
      resetCount();
      resetBatting();
      break;
    }
    case 'field_out': {
      resetCount();
      advanceOut();
      break;
    }
    case 'double_play': {
      resetCount();
      if (f.on_1b) f.on_1b = false;
      advanceOut(2);
      break;
    }
    case 'sac_fly': {
      let runs = 0;
      if (f.on_3b) { runs++; f.on_3b = false; }
      scoreBattingRun(runs);
      resetCount();
      advanceOut();
      break;
    }
    case 'hbp': {
      let runs = 0;
      if (f.on_1b && f.on_2b && f.on_3b) runs = 1;
      if (f.on_2b) f.on_3b = true;
      if (f.on_1b) f.on_2b = true;
      f.on_1b = true;
      scoreBattingRun(runs);
      resetCount();
      resetBatting();
      break;
    }
    default:
      break;
  }
  return f;
}

/* ── Searchable Dropdown Component ───────────────────────────────── */
function SearchableDropdown({ label, items, value, onChange, placeholder }) {
  const [search, setSearch] = useState(value || '');
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => { setSearch(value || ''); }, [value]);

  useEffect(() => {
    const handler = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const filtered = items.filter(i => i.toLowerCase().includes(search.toLowerCase()));

  return (
    <div className="form-group full-width" ref={ref}>
      <label className="form-label">{label}</label>
      <div className="pitcher-search-wrapper">
        <input
          className="form-input"
          type="text"
          placeholder={placeholder}
          value={search}
          onChange={e => { setSearch(e.target.value); setOpen(true); }}
          onFocus={() => setOpen(true)}
        />
        {open && filtered.length > 0 && (
          <div className="pitcher-dropdown">
            {filtered.slice(0, 50).map(p => (
              <div
                key={p}
                className={`pitcher-dropdown-item ${value === p ? 'active' : ''}`}
                onClick={() => { onChange(p); setSearch(p); setOpen(false); }}
              >
                {p}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Main App ────────────────────────────────────────────────────── */
function App() {
  const [pitchers, setPitchers] = useState([]);
  const [batters, setBatters] = useState([]);
  const [teams, setTeams] = useState([]);

  const [form, setForm] = useState({
    pitcher_name: '',
    batter_name: '',
    balls: 0,
    strikes: 0,
    outs: 0,
    inning: 1,
    inning_topbot: 'Top',
    on_1b: false,
    on_2b: false,
    on_3b: false,
    home_team: '',
    away_team: '',
    home_score: 0,
    away_score: 0,
    stand: 'R',
    p_throws: 'R',
    prev_pitch: 'None',
    prev_zone: '',
    prev2_pitch: 'None',
    prev2_zone: '',
  });

  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [liveMode, setLiveMode] = useState(false);
  
  const [actualPitchType, setActualPitchType] = useState('');
  const [actualZone, setActualZone] = useState('');

  useEffect(() => {
    fetch(`${API_BASE}/pitchers`).then(r => r.json()).then(setPitchers).catch(console.error);
    fetch(`${API_BASE}/batters`).then(r => r.json()).then(data => {
      setBatters(data.map(b => b.name));
    }).catch(console.error);
    fetch(`${API_BASE}/teams`).then(r => r.json()).then(setTeams).catch(console.error);
  }, []);

  const handleChange = (field, value) => setForm(f => ({ ...f, [field]: value }));
  const toggleBase = (base) => setForm(f => ({ ...f, [base]: !f[base] }));

  const handlePredict = async () => {
    if (!form.pitcher_name) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      setPredictions(data.predictions);
      if (data.predictions && data.predictions.length > 0) {
        setActualPitchType(data.predictions[0].pitch_type);
        setActualZone(data.predictions[0].zone);
      } else {
        setActualPitchType('');
        setActualZone('');
      }
    } catch (err) { console.error(err); }
    finally { setLoading(false); }
  };

  const handleOutcome = (outcome) => {
    const pitchToSave = actualPitchType || predictions?.[0]?.pitch_type || form.prev_pitch;
    const zoneToSave = actualZone || predictions?.[0]?.zone || form.prev_zone;
    const updated = applyOutcome(form, outcome, pitchToSave, zoneToSave);
    setForm(updated);
    setPredictions(null);
  };

  const zoneHighlights = predictions
    ? predictions.map(p => ({ zone: p.zone, probability: p.probability }))
    : [];

  return (
    <div className="app">
      <header className="app-header">
        <h1>⚾ MLB Pitch Predictor</h1>
        <p>Predict the next pitch type and location based on game state</p>
        <div style={{ marginTop: '0.75rem' }}>
          <button
            className={`toggle-btn ${liveMode ? 'active' : ''}`}
            onClick={() => setLiveMode(!liveMode)}
            style={{ fontSize: '0.85rem' }}
          >
            {liveMode ? '🟢 Live Game Mode ON' : '⚪ Live Game Mode OFF'}
          </button>
        </div>
      </header>

      {/* ── Scoreboard ──────────────────────────────────────────── */}
      <div className="scoreboard">
        <div className="scoreboard-team">
          <span className="scoreboard-label">AWAY</span>
          <select className="form-select scoreboard-select" value={form.away_team} onChange={e => handleChange('away_team', e.target.value)}>
            <option value="">Team</option>
            {teams.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
          <input className="form-input scoreboard-score" type="number" min="0" value={form.away_score}
            onChange={e => handleChange('away_score', Math.max(0, +e.target.value))} />
        </div>
        <div className="scoreboard-vs">
          <div className="scoreboard-inning">{form.inning_topbot === 'Top' ? '▲' : '▼'} {form.inning}</div>
          <div className="scoreboard-count">{form.balls}-{form.strikes}, {form.outs} out{form.outs !== 1 ? 's' : ''}</div>
        </div>
        <div className="scoreboard-team">
          <span className="scoreboard-label">HOME</span>
          <select className="form-select scoreboard-select" value={form.home_team} onChange={e => handleChange('home_team', e.target.value)}>
            <option value="">Team</option>
            {teams.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
          <input className="form-input scoreboard-score" type="number" min="0" value={form.home_score}
            onChange={e => handleChange('home_score', Math.max(0, +e.target.value))} />
        </div>
      </div>

      <div className="main-grid">
        {/* ── Left: Form ────────────────────────────────────────── */}
        <div className="card">
          <div className="card-title"><span className="icon">🎯</span> Game Scenario</div>
          <div className="form-grid">
            <SearchableDropdown
              label="Pitcher"
              items={pitchers}
              value={form.pitcher_name}
              onChange={v => handleChange('pitcher_name', v)}
              placeholder="Search pitcher..."
            />
            <SearchableDropdown
              label="Batter"
              items={batters}
              value={form.batter_name}
              onChange={v => handleChange('batter_name', v)}
              placeholder="Search batter..."
            />

            {/* Count */}
            <div className="form-group">
              <label className="form-label">Balls</label>
              <select className="form-select" value={form.balls} onChange={e => handleChange('balls', +e.target.value)}>
                {[0,1,2,3].map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Strikes</label>
              <select className="form-select" value={form.strikes} onChange={e => handleChange('strikes', +e.target.value)}>
                {[0,1,2].map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Outs</label>
              <select className="form-select" value={form.outs} onChange={e => handleChange('outs', +e.target.value)}>
                {[0,1,2].map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Inning</label>
              <select className="form-select" value={form.inning} onChange={e => handleChange('inning', +e.target.value)}>
                {Array.from({ length: 18 }, (_, i) => i + 1).map(v => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Half</label>
              <select className="form-select" value={form.inning_topbot} onChange={e => handleChange('inning_topbot', e.target.value)}>
                <option value="Top">Top</option>
                <option value="Bot">Bottom</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Batter Stands</label>
              <select className="form-select" value={form.stand} onChange={e => handleChange('stand', e.target.value)}>
                <option value="R">Right</option>
                <option value="L">Left</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Pitcher Throws</label>
              <select className="form-select" value={form.p_throws} onChange={e => handleChange('p_throws', e.target.value)}>
                <option value="R">Right</option>
                <option value="L">Left</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Prev. Pitch</label>
              <select className="form-select" value={form.prev_pitch} onChange={e => handleChange('prev_pitch', e.target.value)}>
                {PITCH_TYPES.map(p => <option key={p} value={p}>{p}</option>)}
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Prev. Zone</label>
              <select className="form-select" value={form.prev_zone} onChange={e => handleChange('prev_zone', e.target.value)}>
                <option value="">None</option>
                {[1,2,3,4,5,6,7,8,9,11,12,13,14].map(z => <option key={z} value={z}>{z}</option>)}
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Prev 2 Pitch</label>
              <select className="form-select" value={form.prev2_pitch} onChange={e => handleChange('prev2_pitch', e.target.value)}>
                {PITCH_TYPES.map(p => <option key={p} value={p}>{p}</option>)}
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Prev 2 Zone</label>
              <select className="form-select" value={form.prev2_zone} onChange={e => handleChange('prev2_zone', e.target.value)}>
                <option value="">None</option>
                {[1,2,3,4,5,6,7,8,9,11,12,13,14].map(z => <option key={z} value={z}>{z}</option>)}
              </select>
            </div>

            {/* Base Runners */}
            <div className="form-group full-width">
              <label className="form-label">Runners on Base</label>
              <div className="toggle-row">
                {['on_1b', 'on_2b', 'on_3b'].map(b => (
                  <button key={b} className={`toggle-btn ${form[b] ? 'active' : ''}`} onClick={() => toggleBase(b)}>
                    {b.replace('on_', '').toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <button className="predict-btn" onClick={handlePredict} disabled={loading || !form.pitcher_name}>
            {loading && <span className="loading-spinner" />}
            {loading ? 'Predicting...' : 'Predict Next Pitch'}
          </button>
        </div>

        {/* ── Right: Zone + Results ─────────────────────────────── */}
        <div>
          <div className="card">
            <div className="card-title"><span className="icon">🎯</span> Strike Zone</div>
            <StrikeZone highlightedZones={zoneHighlights} />
          </div>

          {predictions && (
            <div className="results-section">
              <div className="card">
                <div className="card-title"><span className="icon">📊</span> Top 5 Predictions</div>
                {predictions.map((p, i) => (
                  <div className="result-card" key={i}>
                    <div className={`result-rank rank-${i + 1}`}>#{i + 1}</div>
                    <div className="result-info">
                      <div className="result-pitch-name">{p.pitch_type}</div>
                      <div className="result-zone-label">Zone {p.zone} — {p.zone_label}</div>
                    </div>
                    <div className="result-prob-wrapper">
                      <div className="result-prob">{p.probability}%</div>
                      <div className="result-prob-bar">
                        <div className="result-prob-bar-fill" style={{ width: `${Math.min(p.probability * 5, 100)}%` }} />
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* ── Live Game Outcome Panel ─────────────────────── */}
              {liveMode && (
                <div className="card live-panel">
                  <div className="card-title"><span className="icon">🎮</span> What Happened?</div>
                  
                  <div className="form-grid" style={{ marginBottom: '1.25rem', paddingBottom: '1.25rem', borderBottom: '1px solid var(--border)' }}>
                    <div className="form-group">
                      <label className="form-label">Actual Pitch</label>
                      <select className="form-select" value={actualPitchType} onChange={e => setActualPitchType(e.target.value)}>
                        <option value="">-- Select Pitch --</option>
                        {PITCH_TYPES.filter(p => p !== 'None').map(p => <option key={p} value={p}>{p}</option>)}
                      </select>
                    </div>
                    <div className="form-group">
                      <label className="form-label">Actual Zone</label>
                      <select className="form-select" value={actualZone} onChange={e => setActualZone(e.target.value)}>
                        <option value="">-- Select Zone --</option>
                        {[1,2,3,4,5,6,7,8,9,11,12,13,14].map(z => <option key={z} value={z}>Zone {z}</option>)}
                      </select>
                    </div>
                  </div>

                  <p className="live-hint" style={{ marginTop: 0 }}>Select the pitch outcome to auto-advance the game state.</p>
                  <div className="outcome-grid">
                    <div className="outcome-group">
                      <span className="outcome-group-label">No Contact</span>
                      <button className="outcome-btn outcome-ball" onClick={() => handleOutcome('ball')}>Ball</button>
                      <button className="outcome-btn outcome-strike" onClick={() => handleOutcome('called_strike')}>Called Strike</button>
                      <button className="outcome-btn outcome-strike" onClick={() => handleOutcome('swinging_strike')}>Swinging Strike</button>
                      <button className="outcome-btn outcome-foul" onClick={() => handleOutcome('foul')}>Foul</button>
                      <button className="outcome-btn outcome-hbp" onClick={() => handleOutcome('hbp')}>HBP</button>
                    </div>
                    <div className="outcome-group">
                      <span className="outcome-group-label">In Play — Hit</span>
                      <button className="outcome-btn outcome-hit" onClick={() => handleOutcome('single')}>Single</button>
                      <button className="outcome-btn outcome-hit" onClick={() => handleOutcome('double')}>Double</button>
                      <button className="outcome-btn outcome-hit" onClick={() => handleOutcome('triple')}>Triple</button>
                      <button className="outcome-btn outcome-hr" onClick={() => handleOutcome('home_run')}>Home Run 💣</button>
                    </div>
                    <div className="outcome-group">
                      <span className="outcome-group-label">In Play — Out</span>
                      <button className="outcome-btn outcome-out" onClick={() => handleOutcome('field_out')}>Field Out</button>
                      <button className="outcome-btn outcome-out" onClick={() => handleOutcome('double_play')}>Double Play</button>
                      <button className="outcome-btn outcome-out" onClick={() => handleOutcome('sac_fly')}>Sac Fly</button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;

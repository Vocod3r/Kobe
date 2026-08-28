import { useCallback, useEffect, useRef, useState } from 'react';
import BlockEditor from './BlockEditor';

const SLIDER_KEYS = ['curiosity', 'safety', 'comfort', 'efficiency'];
const TAB_KEYS = ['diagnostics', 'ir', 'generated', 'sliders'];

function App() {
  const [source, setSource] = useState('');
  const [compileResult, setCompileResult] = useState(null);
  const [activeTab, setActiveTab] = useState('diagnostics');
  const [generatedFile, setGeneratedFile] = useState('train');
  const [priorities, setPriorities] = useState(null);
  const [sliderDescriptions, setSliderDescriptions] = useState({});
  const [training, setTraining] = useState(false);
  const [trainProgress, setTrainProgress] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);
  const debounceRef = useRef(null);

  const compile = useCallback(async (text) => {
    if (!window.kobe) {
      setError('Not running in Electron — IPC unavailable.');
      return;
    }
    try {
      setError(null);
      const result = await window.kobe.compile({ source: text, trialLevel: 2, target: 'rl' });
      setCompileResult(result);
      if (result.priorities) setPriorities(result.priorities);
      if (result.sliderDescriptions) setSliderDescriptions(result.sliderDescriptions);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => compile(source), 400);
    return () => clearTimeout(debounceRef.current);
  }, [source, compile]);

  const handlePriorityChange = (key, value) => {
    setPriorities((prev) => ({ ...prev, [key]: parseFloat(value) }));
  };

  const handleTrain = async () => {
    if (!compileResult || compileResult.blocked || !compileResult.ir) return;
    setTraining(true);
    setTrainProgress(null);
    setMetrics(null);
    setError(null);

    const unsub = window.kobe.onTrainProgress((msg) => {
      if (msg.type === 'progress') setTrainProgress(msg);
    });

    try {
      const result = await window.kobe.train({
        ir: compileResult.ir,
        priorities: priorities || compileResult.priorities,
        algorithm: compileResult.algorithm || 'SAC',
        trialLevel: 2,
      });
      if (result.metrics) setMetrics(result.metrics);
    } catch (err) {
      setError(err.message);
    } finally {
      unsub();
      setTraining(false);
    }
  };

  const blocked = compileResult?.blocked;
  const diagnostics = compileResult?.diagnostics || [];
  const codeFiles = compileResult?.code || {};

  return (
    <div className="app">
      <header className="toolbar">
        <h1>Kobe IDE</h1>
        <span className="chip">{compileResult?.algorithm || 'SAC'}</span>
        <span className="chip">{compileResult?.hardware?.target || 'EV3'}</span>
        <div className="spacer" />
        <button
          className="secondary"
          onClick={() => compile(source)}
          disabled={training}
        >
          Recompile
        </button>
        <button
          onClick={handleTrain}
          disabled={training || blocked || !compileResult?.ir}
        >
          {training ? 'Training…' : 'Train Robot'}
        </button>
      </header>

      <div className="main">
        <section className="editor-pane">
          <div className="pane-header">Kobe Program</div>
          <div className="editor-wrap">
            <BlockEditor onSourceChange={setSource} />
          </div>
        </section>

        <aside className="inspector">
          <div className="pane-header">What did Kobe generate?</div>
          <div className="tabs">
            {TAB_KEYS.map((tab) => (
              <div
                key={tab}
                className={`tab ${activeTab === tab ? 'active' : ''}`}
                onClick={() => setActiveTab(tab)}
              >
                {tab === 'ir' ? 'IR' : tab.charAt(0).toUpperCase() + tab.slice(1)}
              </div>
            ))}
          </div>

          <div className="tab-panel">
            {activeTab === 'diagnostics' && (
              diagnostics.length === 0 ? (
                <p className="empty">No diagnostics — program looks good.</p>
              ) : (
                <ul className="diagnostics">
                  {diagnostics.map((d, i) => (
                    <li key={i} className={d.severity}>
                      <div>{d.message}</div>
                      <div className="loc">line {d.line}, col {d.col}</div>
                    </li>
                  ))}
                </ul>
              )
            )}

            {activeTab === 'ir' && (
              compileResult?.ir ? (
                <pre className="code-block">{JSON.stringify(compileResult.ir, null, 2)}</pre>
              ) : (
                <p className="empty">Compile to see IR.</p>
              )
            )}

            {activeTab === 'generated' && (
              <>
                <div className="meta-row">
                  {['client', 'environment', 'train'].map((f) => (
                    <button
                      key={f}
                      className={`secondary ${generatedFile === f ? '' : ''}`}
                      onClick={() => setGeneratedFile(f)}
                      style={{ opacity: generatedFile === f ? 1 : 0.6 }}
                    >
                      {f}.py
                    </button>
                  ))}
                </div>
                {codeFiles[generatedFile] ? (
                  <pre className="code-block">{codeFiles[generatedFile]}</pre>
                ) : (
                  <p className="empty">Generated code appears after a clean compile.</p>
                )}
              </>
            )}

            {activeTab === 'sliders' && (
              <div className="sliders">
                {SLIDER_KEYS.map((key) => {
                  const desc = sliderDescriptions[key] || {};
                  const val = priorities?.[key] ?? compileResult?.priorities?.[key] ?? 0.5;
                  return (
                    <div key={key} className="slider-row">
                      <label>
                        <span>{key}</span>
                        <span>{val.toFixed(2)}</span>
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={val}
                        onChange={(e) => handlePriorityChange(key, e.target.value)}
                      />
                      <div className="slider-effect">
                        <strong>{desc.param || '—'}</strong>
                        {desc.effect ? ` — ${desc.effect}` : ''}
                      </div>
                    </div>
                  );
                })}

                {metrics && (
                  <div className="metrics">
                    <div className="metric">
                      <div className="value">{metrics.speed}%</div>
                      <div className="label">Speed</div>
                    </div>
                    <div className="metric">
                      <div className="value">{metrics.safety}%</div>
                      <div className="label">Safety</div>
                    </div>
                    <div className="metric">
                      <div className="value">{metrics.convenience}%</div>
                      <div className="label">Comfort</div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </aside>
      </div>

      <footer className="status-bar">
        {error && <span style={{ color: 'var(--error)' }}>{error}</span>}
        {!error && blocked && <span className="blocked">Training blocked — fix warnings first</span>}
        {!error && !blocked && compileResult?.ir && <span className="ready">Ready to train</span>}
        {training && trainProgress && (
          <>
            <span>Step {trainProgress.step.toLocaleString()} / {trainProgress.total.toLocaleString()}</span>
            <div className="progress-bar">
              <div
                className="fill"
                style={{ width: `${(trainProgress.step / trainProgress.total) * 100}%` }}
              />
            </div>
          </>
        )}
      </footer>
    </div>
  );
}

export default App;
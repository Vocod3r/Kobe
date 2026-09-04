import { useCallback, useEffect, useRef, useState } from 'react';
import BlockEditor from './BlockEditor';

const TAB_KEYS = ['diagnostics', 'ir', 'generated'];
const POLICY_KEYS = ['curiosity', 'safety', 'comfort', 'efficiency'];

function App() {
  const [source, setSource] = useState('');
  const [compileResult, setCompileResult] = useState(null);
  const [activeTab, setActiveTab] = useState('diagnostics');
  const [generatedFile, setGeneratedFile] = useState('train');
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
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => compile(source), 400);
    return () => clearTimeout(debounceRef.current);
  }, [source, compile]);

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
        priorities: compileResult.priorities,
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
  const priorities = compileResult?.priorities;
  const algorithm = compileResult?.algorithm || 'SAC';

  const progressPct = trainProgress
    ? (trainProgress.step / trainProgress.total) * 100
    : 0;

  return (
    <div className="app">
      <header className="toolbar">
        <h1>Kobe IDE</h1>
        <span className="chip">{algorithm}</span>
        <span className="chip">{compileResult?.hardware?.target || 'EV3'}</span>
        <div className="spacer" />
        <button className="secondary" onClick={() => compile(source)} disabled={training}>
          Recompile
        </button>
        <button onClick={handleTrain} disabled={training || blocked || !compileResult?.ir}>
          {training ? 'Training…' : 'Train Robot'}
        </button>
      </header>

      <div className="main">
        <section className="editor-pane">
          <div className="pane-header">Kobe Program</div>
          <div className="editor-wrap">
            <BlockEditor onSourceChange={setSource} />
          </div>

          <div className="sim-panel">
            <div className="pane-header">Simulation</div>
            <div className="sim-body">
              {error && <p className="sim-error">{error}</p>}

              {!error && training && (
                <>
                  <div className="sim-line">
                    <span>Training {algorithm}…</span>
                    {trainProgress && (
                      <span>
                        Step {trainProgress.step.toLocaleString()} /{' '}
                        {trainProgress.total.toLocaleString()}
                      </span>
                    )}
                  </div>
                  <div className="progress-bar">
                    <div className="fill" style={{ width: `${progressPct}%` }} />
                  </div>
                </>
              )}

              {!error && !training && metrics && (
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

              {!error && !training && !metrics && (
                <div className="sim-idle">
                  <p className="empty">
                    {blocked
                      ? 'Fix warnings, then press Train Robot.'
                      : compileResult?.ir
                        ? 'Press Train Robot to run the simulator.'
                        : 'Build a program to begin.'}
                  </p>
                  {priorities && (
                    <div className="priority-readout">
                      {POLICY_KEYS.map((k) => (
                        <div key={k} className="priority-row">
                          <span>{k}</span>
                          <span>{Number(priorities[k] ?? 0).toFixed(2)}</span>
                        </div>
                      ))}
                      <span className="hint-text">Set these in the policy block.</span>
                    </div>
                  )}
                </div>
              )}
            </div>
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
                      className="secondary"
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
          </div>
        </aside>
      </div>

      <footer className="status-bar">
        {error && <span style={{ color: 'var(--error)' }}>{error}</span>}
        {!error && blocked && <span className="blocked">Training blocked — fix warnings first</span>}
        {!error && !blocked && compileResult?.ir && !training && (
          <span className="ready">Ready to train</span>
        )}
        {!error && training && <span>Running simulator…</span>}
      </footer>
    </div>
  );
}

export default App;
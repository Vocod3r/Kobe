import { app, BrowserWindow, ipcMain } from 'electron';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '..', '..');
const isDev = !app.isPackaged;

function pythonCmd() {
  return process.platform === 'win32' ? 'python' : 'python3';
}

function runPython(script, payload) {
  return new Promise((resolve, reject) => {
    const proc = spawn(
      pythonCmd(),
      [path.join(ROOT, 'ipc', script)],
      {
        cwd: ROOT,
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
      },
    );

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (chunk) => { stdout += chunk.toString(); });
    proc.stderr.on('data', (chunk) => { stderr += chunk.toString(); });

    proc.on('error', reject);
    proc.on('close', (code) => {
      if (code !== 0 && !stdout.trim()) {
        reject(new Error(stderr || `Python exited with code ${code}`));
        return;
      }
      resolve({ stdout, stderr, code });
    });

    proc.stdin.write(JSON.stringify(payload));
    proc.stdin.end();
  });
}

function parseJsonResponse(stdout) {
  const trimmed = stdout.trim();
  if (!trimmed) throw new Error('Empty response from Python');

  // Single JSON object (pipeline) or last line may be the final object
  try {
    return JSON.parse(trimmed);
  } catch {
    const lines = trimmed.split('\n').filter(Boolean);
    for (let i = lines.length - 1; i >= 0; i--) {
      try {
        return JSON.parse(lines[i]);
      } catch {
        // keep searching
      }
    }
    throw new Error(`Could not parse JSON from Python output: ${trimmed.slice(0, 200)}`);
  }
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  if (isDev) {
    win.loadURL('http://localhost:5173');
    win.webContents.openDevTools({ mode: 'detach' });
  } else {
    win.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
  }
}

ipcMain.handle('kobe:compile', async (_evt, payload) => {
  const { stdout, stderr } = await runPython('pipeline_ipc.py', payload);
  const result = parseJsonResponse(stdout);
  if (result.error) throw new Error(result.error);
  return result;
});

ipcMain.handle('kobe:train', async (evt, payload) => {
  return new Promise((resolve, reject) => {
    const proc = spawn(
      pythonCmd(),
      [path.join(ROOT, 'ipc', 'train_ipc.py')],
      {
        cwd: ROOT,
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
      },
    );

    let stderr = '';

    proc.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    proc.stdout.on('data', (chunk) => {
      const lines = chunk.toString().split('\n').filter(Boolean);
      for (const line of lines) {
        try {
          const msg = JSON.parse(line);
          if (msg.type === 'done') {
            resolve(msg);
          } else if (msg.type === 'error') {
            reject(new Error(msg.error || 'Training failed'));
          } else {
            evt.sender.send('kobe:train-progress', msg);
          }
        } catch {
          // ignore non-JSON noise
        }
      }
    });

    proc.on('error', reject);
    proc.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(stderr || `Training process exited with code ${code}`));
      }
    });

    proc.stdin.write(JSON.stringify(payload));
    proc.stdin.end();
  });
});

app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
const { app, BrowserWindow, ipcMain, shell, dialog } = require('electron');
const { spawn } = require('child_process');
const fs = require('fs');
const http = require('http');
const path = require('path');

let mainWindow = null;
let backendProcess = null;
let backendStartedByUi = false;
let backendLogBuffer = [];
let pendingLogBatch = [];
let logBatchTimer = null;
let pendingLogFileText = '';
let logFileFlushTimer = null;
let logFileWriteInFlight = false;
const logSubscribers = new Set();
let nextBackendLogId = 0;
let backendGeneration = 0;
let backendOperationQueue = Promise.resolve();
const LOG_FILE_MAX_BYTES = 5 * 1024 * 1024;
let backendStatus = {
  running: false,
  ready: false,
  starting: false,
  pid: null,
  url: 'http://127.0.0.1:8766',
  startedByUi: false,
  lastError: null,
  generation: 0,
};

const cliOptions = parseArgs(process.argv.slice(1));
const host = cliOptions.host || '127.0.0.1';
const port = Number(cliOptions.port || 8766);
const dashboardUrl = `http://${host}:${port}`;
backendStatus.url = dashboardUrl;

function parseArgs(args) {
  const parsed = {};
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (!arg.startsWith('--')) continue;
    const key = arg.slice(2);
    const next = args[i + 1];
    if (next && !next.startsWith('--')) {
      parsed[key] = next;
      i += 1;
    } else {
      parsed[key] = true;
    }
  }
  return parsed;
}

// Full edition first, then lite; --edition lite|full narrows the search.
const BACKEND_NAMES = (() => {
  const all = [
    { edition: 'full', exe: 'YOLO26n-VRC-Mocap.exe', dist: 'YOLO26n-VRC-Mocap', venv: '.venv' },
    { edition: 'lite', exe: 'YOLO26n-VRC-Mocap-Lite.exe', dist: 'YOLO26n-VRC-Mocap-Lite', venv: '.venv-lite' },
  ];
  const wanted = String(cliOptions.edition || process.env.YOLO26N_EDITION || '').toLowerCase();
  const filtered = all.filter(item => item.edition === wanted);
  return filtered.length ? filtered : all;
})();

function getProjectRoot() {
  if (cliOptions.root) {
    return path.resolve(String(cliOptions.root));
  }

  const candidates = [];
  if (process.env.PORTABLE_EXECUTABLE_DIR) {
    candidates.push(path.resolve(process.env.PORTABLE_EXECUTABLE_DIR));
  }
  if (process.execPath) {
    candidates.push(path.dirname(process.execPath));
  }
  if (process.resourcesPath) {
    candidates.push(path.dirname(process.resourcesPath));
  }
  candidates.push(path.resolve(__dirname, '..', '..', '..'));
  candidates.push(path.resolve(__dirname, '..', '..'));

  for (const candidate of candidates) {
    if (
      BACKEND_NAMES.some(item => fs.existsSync(path.join(candidate, item.exe))) ||
      fs.existsSync(path.join(candidate, 'bin', 'desktop_launcher.py'))
    ) {
      return candidate;
    }
  }

  return app.isPackaged ? path.dirname(process.execPath) : path.resolve(__dirname, '..', '..');
}

function getBackendCommand() {
  const root = getProjectRoot();
  if (cliOptions.backend) {
    return {
      command: path.resolve(String(cliOptions.backend)),
      args: ['--no-browser', '--host', host, '--port', String(port)],
      cwd: path.dirname(path.resolve(String(cliOptions.backend))),
    };
  }

  const backendArgs = ['--no-browser', '--host', host, '--port', String(port)];
  for (const item of BACKEND_NAMES) {
    const distExe = path.join(root, item.exe);
    if (fs.existsSync(distExe)) {
      return { command: distExe, args: backendArgs, cwd: root, edition: item.edition };
    }
  }

  for (const item of BACKEND_NAMES) {
    const projectExe = path.join(root, 'dist', item.dist, item.exe);
    if (fs.existsSync(projectExe)) {
      return { command: projectExe, args: backendArgs, cwd: path.dirname(projectExe), edition: item.edition };
    }
  }

  const sourceLauncher = path.join(root, 'bin', 'desktop_launcher.py');
  for (const item of BACKEND_NAMES) {
    const venvPython = process.platform === 'win32'
      ? path.join(root, item.venv, 'Scripts', 'python.exe')
      : path.join(root, item.venv, 'bin', 'python');
    if (fs.existsSync(venvPython)) {
      return { command: venvPython, args: [sourceLauncher, ...backendArgs], cwd: root, edition: item.edition };
    }
  }
  return {
    command: process.platform === 'win32' ? 'python' : 'python3',
    args: [sourceLauncher, ...backendArgs],
    cwd: root,
    edition: null,
  };
}

function appendLog(source, text) {
  const normalized = String(text).replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  const lines = normalized.split('\n');
  for (const line of lines) {
    if (!line) continue;
    const entry = {
      id: ++nextBackendLogId,
      ts: new Date().toISOString(),
      source,
      line,
    };
    backendLogBuffer.push(entry);
    if (backendLogBuffer.length > 2000) {
      backendLogBuffer = backendLogBuffer.slice(-1500);
    }
    queueLogEntry(entry);
    queueLogFileLine(`[${entry.ts}] [${entry.source}] ${entry.line}\n`);
  }
}

function queueLogFileLine(line) {
  pendingLogFileText += line;
  if (pendingLogFileText.length > 512 * 1024) {
    pendingLogFileText = pendingLogFileText.slice(-384 * 1024);
  }
  if (logFileFlushTimer !== null) return;
  logFileFlushTimer = setTimeout(flushLogFile, 1500);
}

async function flushLogFile() {
  logFileFlushTimer = null;
  if (!pendingLogFileText || logFileWriteInFlight) return;
  const text = pendingLogFileText;
  pendingLogFileText = '';
  logFileWriteInFlight = true;
  try {
    const logPath = path.join(app.getPath('userData'), 'backend-ui.log');
    const rotatedPath = `${logPath}.1`;
    try {
      const stat = await fs.promises.stat(logPath);
      if (stat.size + Buffer.byteLength(text, 'utf8') > LOG_FILE_MAX_BYTES) {
        await fs.promises.rm(rotatedPath, { force: true });
        await fs.promises.rename(logPath, rotatedPath);
      }
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
    await fs.promises.appendFile(logPath, text, 'utf8');
  } catch (_) {
    // Logging must never block startup or UI interaction.
  } finally {
    logFileWriteInFlight = false;
    if (pendingLogFileText) {
      logFileFlushTimer = setTimeout(flushLogFile, 250);
    }
  }
}

function queueLogEntry(entry) {
  pendingLogBatch.push(entry);
  if (pendingLogBatch.length > 500) {
    pendingLogBatch = pendingLogBatch.slice(-500);
  }
  if (logSubscribers.size <= 0 || logBatchTimer !== null) {
    return;
  }
  logBatchTimer = setTimeout(flushLogBatch, 1000);
}

function flushLogBatch() {
  logBatchTimer = null;
  if (logSubscribers.size <= 0 || pendingLogBatch.length === 0) {
    return;
  }
  const batch = pendingLogBatch;
  pendingLogBatch = [];
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('backend-log-batch', batch);
  }
}

function emitStatus() {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('backend-status', { ...backendStatus });
  }
}

function isBackendReady(timeoutMs = 800) {
  return new Promise((resolve) => {
    let done = false;
    const finish = (value) => {
      if (done) return;
      done = true;
      resolve(value);
    };
    const request = http.request({
      hostname: host === '0.0.0.0' ? '127.0.0.1' : host,
      port,
      path: '/api/config',
      method: 'GET',
    }, (response) => {
      response.resume();
      finish(response.statusCode >= 200 && response.statusCode < 300);
    });
    request.setTimeout(timeoutMs, () => {
      request.destroy();
      finish(false);
    });
    request.once('error', () => finish(false));
    request.end();
  });
}

async function waitForBackend(child, timeoutMs = 30000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (child !== backendProcess || child.exitCode !== null) {
      return false;
    }
    if (await isBackendReady(800)) {
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return false;
}

function waitForProcessExit(child, timeoutMs = 5000) {
  if (!child || child.exitCode !== null) return Promise.resolve(true);
  return new Promise((resolve) => {
    let settled = false;
    const finish = (exited) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      child.removeListener('exit', onExit);
      resolve(exited);
    };
    const onExit = () => finish(true);
    const timer = setTimeout(() => finish(false), timeoutMs);
    child.once('exit', onExit);
  });
}

function enqueueBackendOperation(operation) {
  const queued = backendOperationQueue.catch(() => undefined).then(operation);
  backendOperationQueue = queued;
  return queued;
}

async function startBackendInternal() {
  if (backendProcess && backendProcess.exitCode === null) {
    const ready = await isBackendReady();
    if (ready) {
      backendStatus = {
        ...backendStatus,
        running: true,
        ready: true,
        starting: false,
        lastError: null,
      };
      emitStatus();
      return { ok: true, alreadyRunning: true, status: backendStatus };
    }
    const error = backendStatus.lastError || 'Backend process exists but is not ready';
    return { ok: false, error, status: backendStatus };
  }

  if (await isBackendReady()) {
    backendStatus = {
      running: true,
      ready: true,
      starting: false,
      pid: null,
      url: dashboardUrl,
      startedByUi: false,
      lastError: null,
      generation: backendGeneration,
    };
    backendStartedByUi = false;
    appendLog('ui', `Detected existing backend at ${dashboardUrl}`);
    emitStatus();
    return { ok: true, alreadyRunning: true, status: backendStatus };
  }

  const backend = getBackendCommand();
  appendLog('ui', `Starting backend: ${backend.command} ${backend.args.join(' ')}`);

  let child;
  try {
    child = spawn(backend.command, backend.args, {
      cwd: backend.cwd,
      windowsHide: true,
      shell: false,
      env: {
        ...process.env,
        ...(backend.edition === 'lite' ? { YOLO26N_EDITION: 'lite' } : {}),
        PYTHONUNBUFFERED: '1',
        PYTHONUTF8: '1',
        PYTHONIOENCODING: 'utf-8',
      },
    });
  } catch (error) {
    backendStatus = {
      ...backendStatus,
      running: false,
      ready: false,
      starting: false,
      pid: null,
      startedByUi: false,
      lastError: error.message,
    };
    appendLog('error', `Backend spawn failed: ${error.message}`);
    emitStatus();
    return { ok: false, error: error.message, status: backendStatus };
  }

  backendProcess = child;
  backendStartedByUi = true;
  const generation = ++backendGeneration;
  backendStatus = {
    running: false,
    ready: false,
    starting: true,
    pid: child.pid,
    url: dashboardUrl,
    startedByUi: true,
    lastError: null,
    generation,
  };

  if (child.stdout) {
    child.stdout.on('data', (chunk) => appendLog('stdout', chunk));
  }
  if (child.stderr) {
    child.stderr.on('data', (chunk) => appendLog('stderr', chunk));
  }
  child.once('error', (error) => {
    appendLog('error', `Backend process error: ${error.message}`);
    if (backendProcess !== child) return;
    backendProcess = null;
    backendStartedByUi = false;
    backendStatus = {
      running: false,
      ready: false,
      starting: false,
      pid: null,
      url: dashboardUrl,
      startedByUi: false,
      lastError: error.message,
      generation,
    };
    emitStatus();
  });
  child.once('exit', (code, signal) => {
    appendLog('ui', `Backend exited: code=${code ?? 'null'} signal=${signal ?? 'null'}`);
    if (backendProcess !== child) return;
    backendProcess = null;
    backendStartedByUi = false;
    backendStatus = {
      running: false,
      ready: false,
      starting: false,
      pid: null,
      url: dashboardUrl,
      startedByUi: false,
      lastError: code === 0 || code === null ? null : `Exited with code ${code}`,
      generation,
    };
    emitStatus();
  });

  emitStatus();
  const ready = await waitForBackend(child);
  if (!ready) {
    const error = backendStatus.lastError || `Backend did not become ready at ${dashboardUrl} within timeout`;
    if (backendProcess === child) {
      backendStatus = {
        ...backendStatus,
        running: false,
        ready: false,
        starting: false,
        lastError: error,
      };
      appendLog('error', error);
      emitStatus();
    }
    return { ok: false, error, status: backendStatus };
  }
  if (backendProcess !== child) {
    const error = 'Backend process changed before startup completed';
    return { ok: false, error, status: backendStatus };
  }
  backendStatus = {
    ...backendStatus,
    running: true,
    ready: true,
    starting: false,
    lastError: null,
  };
  appendLog('ui', `Backend ready: ${dashboardUrl}`);
  emitStatus();
  return { ok: true, status: backendStatus };
}

async function stopBackendInternal(force = false) {
  if (!backendProcess) {
    return { ok: true, status: backendStatus };
  }
  if (!backendStartedByUi && !force) {
    appendLog('ui', 'Backend was not started by Electron UI; leaving it running.');
    return { ok: true, status: backendStatus };
  }

  const proc = backendProcess;
  appendLog('ui', `Stopping backend pid=${proc.pid}`);
  backendStatus = {
    ...backendStatus,
    running: false,
    ready: false,
    starting: false,
  };
  emitStatus();
  try {
    proc.kill();
  } catch (error) {
    appendLog('error', `Failed to stop backend: ${error.message}`);
    return { ok: false, error: error.message, status: backendStatus };
  }
  const exited = await waitForProcessExit(proc);
  if (!exited && backendProcess === proc) {
    const error = `Backend pid=${proc.pid} did not exit within timeout`;
    backendStatus.lastError = error;
    appendLog('error', error);
    emitStatus();
    return { ok: false, error, status: backendStatus };
  }
  return { ok: true, status: backendStatus };
}

function startBackend() {
  return enqueueBackendOperation(startBackendInternal);
}

function stopBackend(force = false) {
  return enqueueBackendOperation(() => stopBackendInternal(force));
}

function restartBackend() {
  return enqueueBackendOperation(async () => {
    const stopped = await stopBackendInternal(true);
    if (!stopped.ok) return stopped;
    return startBackendInternal();
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1420,
    height: 920,
    minWidth: 1000,
    minHeight: 680,
    backgroundColor: '#111827',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      webviewTag: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, 'renderer.html'));

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

ipcMain.handle('backend:start', async () => startBackend());
ipcMain.handle('backend:stop', async (_event, force) => stopBackend(Boolean(force)));
ipcMain.handle('backend:restart', async () => restartBackend());
ipcMain.handle('backend:status', async () => {
  const ready = await isBackendReady();
  if (!backendProcess && ready) {
    backendStatus = {
      running: true,
      ready: true,
      starting: false,
      pid: null,
      url: dashboardUrl,
      startedByUi: false,
      lastError: null,
      generation: backendGeneration,
    };
  } else if (!backendProcess && !ready) {
    backendStatus.running = false;
    backendStatus.ready = false;
    backendStatus.starting = false;
    backendStatus.pid = null;
    backendStatus.startedByUi = false;
  } else if (backendProcess) {
    backendStatus.running = ready;
    backendStatus.ready = ready;
    backendStatus.starting = !ready && backendProcess.exitCode === null;
  }
  emitStatus();
  return { ...backendStatus };
});
ipcMain.handle('backend:logs', async () => backendLogBuffer);
ipcMain.handle('backend:logs-subscribe', async (event) => {
  const senderId = event.sender.id;
  logSubscribers.add(senderId);
  event.sender.once('destroyed', () => logSubscribers.delete(senderId));
  flushLogBatch();
  return { ok: true, subscribers: logSubscribers.size };
});
ipcMain.handle('backend:logs-unsubscribe', async (event) => {
  logSubscribers.delete(event.sender.id);
  return { ok: true, subscribers: logSubscribers.size };
});
ipcMain.handle('app:open-external', async (_event, url) => {
  await shell.openExternal(url || dashboardUrl);
});
ipcMain.handle('app:show-error', async (_event, message) => {
  dialog.showErrorBox('YOLO26n VRC Mocap', String(message || 'Unknown error'));
});

app.whenReady().then(async () => {
  createWindow();
  if (!cliOptions['no-backend']) {
    await startBackend();
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  if (backendProcess && backendStartedByUi && !cliOptions['keep-backend-on-exit']) {
    try {
      backendProcess.kill();
    } catch (_) {
      // Best-effort shutdown.
    }
  }
});

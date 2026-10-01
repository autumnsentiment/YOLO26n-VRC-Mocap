const dashboardView = document.getElementById('dashboardView');
const loadingOverlay = document.getElementById('loadingOverlay');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const backendUrl = document.getElementById('backendUrl');
const logOutput = document.getElementById('logOutput');
const consolePanel = document.getElementById('consolePanel');
const layout = document.querySelector('.layout');
const toggleConsoleButton = document.getElementById('toggleConsole');

let currentUrl = 'http://127.0.0.1:8766';
let logLines = [];
let pendingLogEntries = [];
let logFlushTimer = null;
let logSubscribed = false;
let statusRefreshInFlight = false;
let lastLogId = 0;
const STATUS_POLL_MS = 10000;
const LOG_FLUSH_MS = 500;
const TEXT = {
  backendRunning: '后端运行中',
  externalBackend: '外部后端',
  backendStopped: '后端已停止',
  backendStartFailed: '后端启动失败',
  backendRestartFailed: '后端重启失败',
  expandConsole: '显示',
  collapseConsole: '折叠',
};

function formatTime(iso) {
  const date = new Date(iso);
  return date.toLocaleTimeString('zh-CN', { hour12: false });
}

function appendLog(entry) {
  const entryId = Number(entry.id || 0);
  if (entryId > 0 && entryId <= lastLogId) return;
  if (entryId > 0) lastLogId = entryId;
  const prefix = `[${formatTime(entry.ts)}] [${entry.source}]`;
  pendingLogEntries.push(`${prefix} ${entry.line}`);
  scheduleLogFlush();
}

function appendLogBatch(entries) {
  entries.forEach(appendLog);
}

function scheduleLogFlush() {
  if (logFlushTimer !== null) return;
  logFlushTimer = setTimeout(flushLogs, LOG_FLUSH_MS);
}

function flushLogs() {
  logFlushTimer = null;
  if (pendingLogEntries.length === 0) return;
  logLines.push(...pendingLogEntries);
  pendingLogEntries = [];
  if (logLines.length > 1200) {
    logLines = logLines.slice(-900);
  }
  logOutput.textContent = logLines.join('\n');
  logOutput.scrollTop = logOutput.scrollHeight;
}

function loadDashboard(force = false) {
  const target = force ? `${currentUrl}/?t=${Date.now()}` : currentUrl;
  if (force || dashboardView.src !== currentUrl && dashboardView.src !== `${currentUrl}/`) {
    loadingOverlay.classList.remove('hidden');
    dashboardView.src = target;
  }
}

function setStatus(status) {
  currentUrl = status.url || currentUrl;
  backendUrl.textContent = currentUrl;

  if (status.running) {
    statusDot.className = 'dot ok';
    const pidText = status.pid ? `PID ${status.pid}` : TEXT.externalBackend;
    statusText.textContent = `${TEXT.backendRunning} - ${pidText}`;
    loadDashboard(false);
  } else {
    statusDot.className = 'dot warn';
    statusText.textContent = status.lastError ? `${TEXT.backendStopped} - ${status.lastError}` : TEXT.backendStopped;
    loadingOverlay.classList.remove('hidden');
  }
}

async function refreshStatus() {
  if (statusRefreshInFlight || document.hidden) return;
  statusRefreshInFlight = true;
  try {
    const status = await window.mocapUi.getBackendStatus();
    setStatus(status);
  } finally {
    statusRefreshInFlight = false;
  }
}

async function loadInitialLogs() {
  const logs = await window.mocapUi.getBackendLogs();
  logLines = [];
  pendingLogEntries = [];
  lastLogId = 0;
  logOutput.textContent = '';
  logs.forEach(appendLog);
  flushLogs();
}

async function subscribeLogs() {
  if (logSubscribed) return;
  logSubscribed = true;
  await window.mocapUi.subscribeBackendLogs();
}

async function unsubscribeLogs() {
  if (!logSubscribed) return;
  logSubscribed = false;
  await window.mocapUi.unsubscribeBackendLogs();
}

document.getElementById('startBackend').addEventListener('click', async () => {
  const result = await window.mocapUi.startBackend();
  if (result.status) setStatus(result.status);
  if (!result.ok) window.mocapUi.showError(result.error || TEXT.backendStartFailed);
});

document.getElementById('restartBackend').addEventListener('click', async () => {
  loadingOverlay.classList.remove('hidden');
  const result = await window.mocapUi.restartBackend();
  if (result.status) setStatus(result.status);
  if (!result.ok) window.mocapUi.showError(result.error || TEXT.backendRestartFailed);
});

document.getElementById('stopBackend').addEventListener('click', async () => {
  const result = await window.mocapUi.stopBackend(false);
  if (result.status) setStatus(result.status);
});

document.getElementById('reloadDashboard').addEventListener('click', () => {
  loadDashboard(true);
});

document.getElementById('openBrowser').addEventListener('click', () => {
  window.mocapUi.openExternal(currentUrl);
});

document.getElementById('clearLogs').addEventListener('click', () => {
  logLines = [];
  pendingLogEntries = [];
  logOutput.textContent = '';
});

toggleConsoleButton.addEventListener('click', async () => {
  consolePanel.classList.toggle('collapsed');
  const collapsed = consolePanel.classList.contains('collapsed');
  layout.classList.toggle('console-collapsed', collapsed);
  toggleConsoleButton.textContent = collapsed ? TEXT.expandConsole : TEXT.collapseConsole;
  toggleConsoleButton.title = collapsed ? '显示日志区域' : '折叠日志区域';
  if (collapsed) {
    await unsubscribeLogs();
  } else {
    await loadInitialLogs();
    await subscribeLogs();
  }
});

dashboardView.addEventListener('load', () => {
  loadingOverlay.classList.add('hidden');
});

window.mocapUi.onBackendLogBatch(appendLogBatch);
window.mocapUi.onBackendStatus(setStatus);

async function initialize() {
  await loadInitialLogs();
  await subscribeLogs();
  await refreshStatus();
}

initialize().catch((error) => {
  window.mocapUi.showError(error.message || String(error));
});
setInterval(refreshStatus, STATUS_POLL_MS);
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) refreshStatus();
});

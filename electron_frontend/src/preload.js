const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('mocapUi', {
  startBackend: () => ipcRenderer.invoke('backend:start'),
  stopBackend: (force = false) => ipcRenderer.invoke('backend:stop', force),
  restartBackend: () => ipcRenderer.invoke('backend:restart'),
  getBackendStatus: () => ipcRenderer.invoke('backend:status'),
  getBackendLogs: () => ipcRenderer.invoke('backend:logs'),
  subscribeBackendLogs: () => ipcRenderer.invoke('backend:logs-subscribe'),
  unsubscribeBackendLogs: () => ipcRenderer.invoke('backend:logs-unsubscribe'),
  openExternal: (url) => ipcRenderer.invoke('app:open-external', url),
  showError: (message) => ipcRenderer.invoke('app:show-error', message),
  onBackendLogBatch: (callback) => {
    const listener = (_event, entries) => callback(entries);
    ipcRenderer.on('backend-log-batch', listener);
    return () => ipcRenderer.removeListener('backend-log-batch', listener);
  },
  onBackendStatus: (callback) => {
    const listener = (_event, status) => callback(status);
    ipcRenderer.on('backend-status', listener);
    return () => ipcRenderer.removeListener('backend-status', listener);
  },
});

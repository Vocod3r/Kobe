const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('kobe', {
  compile: (payload) => ipcRenderer.invoke('kobe:compile', payload),
  train: (payload) => ipcRenderer.invoke('kobe:train', payload),
  onTrainProgress: (callback) => {
    const listener = (_evt, data) => callback(data);
    ipcRenderer.on('kobe:train-progress', listener);
    return () => ipcRenderer.removeListener('kobe:train-progress', listener);
  },
});
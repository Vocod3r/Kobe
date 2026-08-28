import { useEffect, useRef } from 'react';
import * as Blockly from 'blockly/core';
import * as En from 'blockly/msg/en';
import { registerKobeBlocks, TOOLBOX, DEFAULT_WORKSPACE_XML } from './blockly/blocks';
import { workspaceToKobe } from './blockly/kobeGenerator';

Blockly.setLocale(En);
registerKobeBlocks();

const KOBE_THEME = Blockly.Theme.defineTheme('kobeDark', {
  base: Blockly.Themes.Classic,
  componentStyles: {
    workspaceBackgroundColour: '#0f1117',
    toolboxBackgroundColour: '#161b22',
    toolboxForegroundColour: '#e6edf3',
    flyoutBackgroundColour: '#161b22',
    flyoutForegroundColour: '#e6edf3',
    flyoutOpacity: 1,
    scrollbarColour: '#30363d',
    insertionMarkerColour: '#58a6ff',
    insertionMarkerOpacity: 0.4,
  },
});

export default function BlockEditor({ onSourceChange }) {
  const hostRef = useRef(null);
  const workspaceRef = useRef(null);

  useEffect(() => {
    const workspace = Blockly.inject(hostRef.current, {
      toolbox: TOOLBOX,
      theme: KOBE_THEME,
      grid: { spacing: 20, length: 3, colour: '#21262d', snap: true },
      zoom: { controls: true, wheel: true, startScale: 0.9, maxScale: 2, minScale: 0.4 },
      trashcan: true,
      move: { scrollbars: true, drag: true, wheel: true },
    });
    workspaceRef.current = workspace;

    try {
      const xml = Blockly.utils.xml.textToDom(DEFAULT_WORKSPACE_XML);
      Blockly.Xml.domToWorkspace(xml, workspace);
    } catch (e) {
      console.error('Failed to load starter blocks', e);
    }

    const emit = () => {
      try {
        onSourceChange(workspaceToKobe(workspace));
      } catch {
        // Mid-edit states (dangling connections, empty condition sockets)
        // are expected while the user is still building a block; just skip
        // emitting until the workspace is in a generatable state again.
      }
    };

    emit();
    const listener = (event) => {
      if (event.isUiEvent) return;
      if (event.type === Blockly.Events.FINISHED_LOADING) return;
      emit();
    };
    workspace.addChangeListener(listener);

    const handleResize = () => Blockly.svgResize(workspace);
    window.addEventListener('resize', handleResize);
    handleResize();

    return () => {
      window.removeEventListener('resize', handleResize);
      workspace.removeChangeListener(listener);
      workspace.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return <div ref={hostRef} className="blockly-host" />;
}
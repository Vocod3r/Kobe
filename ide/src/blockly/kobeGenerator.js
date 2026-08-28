const INDENT = '    ';

function pad(level) {
  return INDENT.repeat(level);
}

function conditionToKobe(block) {
  if (!block) return 'true';
  switch (block.type) {
    case 'kobe_cond_dist':
      return `dist ${block.getFieldValue('CMP')} ${block.getFieldValue('VAL')} ${block.getFieldValue('UNIT')}`;
    case 'kobe_cond_colour': {
      const neg = block.getFieldValue('NEG') === 'NOT' ? 'not ' : '';
      return `colour is ${neg}${block.getFieldValue('COLOUR')}`;
    }
    case 'kobe_cond_touch':
      return 'touch pressed';
    case 'kobe_cond_ir_detected':
      return 'IR detected';
    case 'kobe_cond_ir_signal':
      return `IR signal ${block.getFieldValue('CMP')} ${block.getFieldValue('VAL')}`;
    case 'kobe_cond_uv_detected':
      return 'UV detected';
    case 'kobe_cond_uv_index':
      return `UV index ${block.getFieldValue('CMP')} ${block.getFieldValue('VAL')}`;
    case 'kobe_cond_gyro':
      return `gyro tilt ${block.getFieldValue('CMP')} ${block.getFieldValue('VAL')} deg`;
    case 'kobe_cond_sound':
      return `sound ${block.getFieldValue('CMP')} ${block.getFieldValue('VAL')} db`;
    case 'kobe_cond_and':
      return `(${conditionToKobe(block.getInputTargetBlock('LEFT'))} and ${conditionToKobe(block.getInputTargetBlock('RIGHT'))})`;
    case 'kobe_cond_or':
      return `(${conditionToKobe(block.getInputTargetBlock('LEFT'))} or ${conditionToKobe(block.getInputTargetBlock('RIGHT'))})`;
    case 'kobe_cond_not':
      return `not (${conditionToKobe(block.getInputTargetBlock('OPERAND'))})`;
    default:
      throw new Error(`Unknown condition block: ${block.type}`);
  }
}

function statementsToKobe(firstBlock, level) {
  const lines = [];
  let block = firstBlock;
  while (block) {
    if (!block.disabled) {
      const code = statementToKobe(block, level);
      if (code) lines.push(code);
    }
    block = block.getNextBlock();
  }
  return lines.join('\n');
}

function branchStackToKobe(firstBlock, level) {
  const lines = [];
  let block = firstBlock;
  while (block) {
    if (block.type === 'kobe_branch' && !block.disabled) {
      lines.push(branchToKobe(block, level));
    }
    block = block.getNextBlock();
  }
  return lines.join('\n');
}

function branchToKobe(block, level) {
  const p = pad(level);
  const cond = conditionToKobe(block.getInputTargetBlock('CONDITION'));
  const thenCode = statementsToKobe(block.getInputTargetBlock('THEN'), level + 1);
  let out = `${p}${cond} then {\n${thenCode}\n${p}}`;
  const elseBlock = block.getInputTargetBlock('ELSE');
  if (elseBlock) {
    out += ` else {\n${statementsToKobe(elseBlock, level + 1)}\n${p}}`;
  }
  return out;
}

function statementToKobe(block, level) {
  const p = pad(level);
  switch (block.type) {
    case 'kobe_move': {
      const action = block.getFieldValue('ACTION');
      const direction = block.getFieldValue('DIRECTION');
      const speed = block.getFieldValue('SPEED');
      const dirPart = direction === 'NONE' ? '' : ` ${direction}`;
      return `${p}${action}${dirPart} ${speed};`;
    }
    case 'kobe_turn':
      return `${p}turn ${block.getFieldValue('DIRECTION')};`;
    case 'kobe_stop':
      return `${p}stop;`;
    case 'kobe_wait':
      return `${p}wait ${block.getFieldValue('VAL')} ${block.getFieldValue('UNIT')};`;
    case 'kobe_break':
      return `${p}break;`;
    case 'kobe_loop_for': {
      const body = statementsToKobe(block.getInputTargetBlock('DO'), level + 1);
      return `${p}loop (${block.getFieldValue('COUNT')}) {\n${body}\n${p}}`;
    }
    case 'kobe_loop_until': {
      const cond = conditionToKobe(block.getInputTargetBlock('CONDITION'));
      const body = statementsToKobe(block.getInputTargetBlock('DO'), level + 1);
      return `${p}loop until (${cond}) {\n${body}\n${p}}`;
    }
    case 'kobe_if': {
      const cond = conditionToKobe(block.getInputTargetBlock('CONDITION'));
      const thenCode = statementsToKobe(block.getInputTargetBlock('THEN'), level + 1);
      let out = `${p}if ${cond} then {\n${thenCode}\n${p}}`;
      const elseBlock = block.getInputTargetBlock('ELSE');
      if (elseBlock) {
        out += ` else {\n${statementsToKobe(elseBlock, level + 1)}\n${p}}`;
      }
      return out;
    }
    case 'kobe_observe': {
      const sensors = block.getFieldValue('SENSORS');
      const branchCode = branchStackToKobe(block.getInputTargetBlock('BRANCHES'), level + 1);
      return `${p}observe(${sensors}) {\n${branchCode}\n${p}}`;
    }
    case 'kobe_branch':
      // Branches are only ever walked from within an observe block
      // (see branchStackToKobe); reaching here means it's disconnected.
      return '';
    default:
      throw new Error(`Unknown statement block: ${block.type}`);
  }
}

/**
 * Serialize an entire Blockly workspace of Kobe blocks into Kobe source text.
 *
 * At most one kobe_algorithm / kobe_hardware / kobe_policy block is honored
 * (first one found, in workspace order) and always emitted first, in that
 * order, per the grammar. Every other top-level block chain is treated as
 * part of the main statement list, emitted in workspace order.
 */
export function workspaceToKobe(workspace) {
  const topBlocks = workspace.getTopBlocks(true);

  let algorithmBlock = null;
  let hardwareBlock = null;
  let policyBlock = null;
  const mainStarters = [];

  for (const block of topBlocks) {
    if (block.disabled) continue;
    if (block.type === 'kobe_algorithm' && !algorithmBlock) {
      algorithmBlock = block;
    } else if (block.type === 'kobe_hardware' && !hardwareBlock) {
      hardwareBlock = block;
    } else if (block.type === 'kobe_policy' && !policyBlock) {
      policyBlock = block;
    } else if (block.type !== 'kobe_branch') {
      mainStarters.push(block);
    }
  }

  const parts = [];

  if (algorithmBlock) {
    parts.push(`algorithm ${algorithmBlock.getFieldValue('NAME')}`);
  }

  if (hardwareBlock) {
    const target = hardwareBlock.getFieldValue('TARGET');
    const motors = (hardwareBlock.getFieldValue('MOTORS') || '')
      .split(',').map((s) => s.trim()).filter(Boolean);
    const sensors = (hardwareBlock.getFieldValue('SENSORS') || '')
      .split(',').map((s) => s.trim()).filter(Boolean);

    let hw = `hardware {\n${INDENT}target: ${target}\n`;
    if (motors.length) hw += `${INDENT}motors: [${motors.join(', ')}]\n`;
    if (sensors.length) hw += `${INDENT}sensors: [${sensors.join(', ')}]\n`;
    hw += '}';
    parts.push(hw);
  }

  if (policyBlock) {
    parts.push(
      `policy {\n` +
      `${INDENT}curiosity = ${policyBlock.getFieldValue('CURIOSITY')};\n` +
      `${INDENT}safety = ${policyBlock.getFieldValue('SAFETY')};\n` +
      `${INDENT}comfort = ${policyBlock.getFieldValue('COMFORT')};\n` +
      `${INDENT}efficiency = ${policyBlock.getFieldValue('EFFICIENCY')};\n` +
      `}`
    );
  }

  for (const starter of mainStarters) {
    const code = statementsToKobe(starter, 0);
    if (code) parts.push(code);
  }

  return parts.join('\n\n') + '\n';
}
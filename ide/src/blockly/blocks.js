import * as Blockly from 'blockly/core';

const CMP_OPTIONS = [
  ['<', '<'], ['<=', '<='], ['>', '>'], ['>=', '>='], ['==', '=='], ['!=', '!='],
];

const BLOCK_DEFS = [
  // ── Program setup (standalone — no connections) ──────────────────────────
  {
    type: 'kobe_algorithm',
    message0: 'algorithm %1',
    args0: [{
      type: 'field_dropdown', name: 'NAME',
      options: [['SAC', 'SAC'], ['TD3', 'TD3'], ['DroQ', 'DroQ'], ['random', 'random']],
    }],
    colour: 210,
    tooltip: 'Choose the RL algorithm used for training.',
  },
  {
    type: 'kobe_hardware',
    message0: '🔧 hardware',
    message1: 'target %1',
    args1: [{
      type: 'field_dropdown', name: 'TARGET',
      options: [['EV3', 'EV3'], ['Spike', 'Spike'], ['RPi', 'RPi'], ['RaspberryPi', 'RaspberryPi']],
    }],
    message2: 'motors (comma-separated ports) %1',
    args2: [{ type: 'field_input', name: 'MOTORS', text: 'A, B' }],
    message3: 'sensors (e.g. dist@1, colour@2) %1',
    args3: [{ type: 'field_input', name: 'SENSORS', text: 'dist@1, colour@2' }],
    colour: 210,
    tooltip: 'Declare the robot hardware: target platform, motor ports, sensor ports.',
  },
  {
    type: 'kobe_policy',
    message0: '🎚️ policy',
    message1: 'curiosity %1',
    args1: [{ type: 'field_number', name: 'CURIOSITY', value: 0.3, min: 0, max: 1, precision: 0.01 }],
    message2: 'safety %1',
    args2: [{ type: 'field_number', name: 'SAFETY', value: 0.5, min: 0, max: 1, precision: 0.01 }],
    message3: 'comfort %1',
    args3: [{ type: 'field_number', name: 'COMFORT', value: 0.5, min: 0, max: 1, precision: 0.01 }],
    message4: 'efficiency %1',
    args4: [{ type: 'field_number', name: 'EFFICIENCY', value: 0.5, min: 0, max: 1, precision: 0.01 }],
    colour: 210,
    tooltip: 'Tune reward priorities (0-1 sliders).',
  },

  // ── Actions ───────────────────────────────────────────────────────────────
  {
    type: 'kobe_move',
    message0: '%1 %2 %3',
    args0: [
      { type: 'field_dropdown', name: 'ACTION', options: [['walk', 'walk'], ['run', 'run']] },
      {
        type: 'field_dropdown', name: 'DIRECTION',
        options: [['forward', 'forward'], ['backward', 'backward'], ['left', 'left'], ['right', 'right'], ['(no direction)', 'NONE']],
      },
      { type: 'field_dropdown', name: 'SPEED', options: [['slowly', 'slowly'], ['normally', 'normally'], ['quickly', 'quickly']] },
    ],
    previousStatement: null,
    nextStatement: null,
    colour: 160,
  },
  {
    type: 'kobe_turn',
    message0: 'turn %1',
    args0: [{ type: 'field_dropdown', name: 'DIRECTION', options: [['left', 'left'], ['right', 'right']] }],
    previousStatement: null,
    nextStatement: null,
    colour: 160,
  },
  {
    type: 'kobe_stop',
    message0: 'stop',
    previousStatement: null,
    nextStatement: null,
    colour: 160,
  },
  {
    type: 'kobe_wait',
    message0: 'wait %1 %2',
    args0: [
      { type: 'field_number', name: 'VAL', value: 500, min: 0 },
      { type: 'field_dropdown', name: 'UNIT', options: [['ms', 'ms'], ['sec', 'sec']] },
    ],
    previousStatement: null,
    nextStatement: null,
    colour: 160,
  },
  {
    type: 'kobe_break',
    message0: 'break out of loop',
    previousStatement: null,
    nextStatement: null,
    colour: 0,
    tooltip: 'Only meaningful inside a loop block.',
  },

  // ── Control ───────────────────────────────────────────────────────────────
  {
    type: 'kobe_loop_for',
    message0: 'loop %1 times',
    args0: [{ type: 'field_number', name: 'COUNT', value: 3, min: 1, precision: 1 }],
    message1: 'do %1',
    args1: [{ type: 'input_statement', name: 'DO' }],
    previousStatement: null,
    nextStatement: null,
    colour: 290,
  },
  {
    type: 'kobe_loop_until',
    message0: 'loop until %1',
    args0: [{ type: 'input_value', name: 'CONDITION', check: 'Condition' }],
    message1: 'do %1',
    args1: [{ type: 'input_statement', name: 'DO' }],
    previousStatement: null,
    nextStatement: null,
    colour: 290,
  },
  {
    type: 'kobe_if',
    message0: 'if %1',
    args0: [{ type: 'input_value', name: 'CONDITION', check: 'Condition' }],
    message1: 'then %1',
    args1: [{ type: 'input_statement', name: 'THEN' }],
    message2: 'else %1',
    args2: [{ type: 'input_statement', name: 'ELSE' }],
    previousStatement: null,
    nextStatement: null,
    colour: 290,
  },
  {
    type: 'kobe_observe',
    message0: 'observe sensors: %1',
    args0: [{ type: 'field_input', name: 'SENSORS', text: 'dist' }],
    message1: 'branches %1',
    args1: [{ type: 'input_statement', name: 'BRANCHES', check: 'Branch' }],
    previousStatement: null,
    nextStatement: null,
    colour: 290,
    tooltip: "List sensors to read (comma-separated, matching hardware sensors), then stack 'when' branch blocks below.",
  },
  {
    type: 'kobe_branch',
    message0: 'when %1',
    args0: [{ type: 'input_value', name: 'CONDITION', check: 'Condition' }],
    message1: 'then %1',
    args1: [{ type: 'input_statement', name: 'THEN' }],
    message2: 'else %1',
    args2: [{ type: 'input_statement', name: 'ELSE' }],
    previousStatement: 'Branch',
    nextStatement: 'Branch',
    colour: 290,
    tooltip: 'Only connects inside an observe block.',
  },

  // ── Conditions (value blocks, output type "Condition") ────────────────────
  {
    type: 'kobe_cond_dist',
    message0: 'distance %1 %2 %3',
    args0: [
      { type: 'field_dropdown', name: 'CMP', options: CMP_OPTIONS },
      { type: 'field_number', name: 'VAL', value: 20, min: 0 },
      { type: 'field_dropdown', name: 'UNIT', options: [['cm', 'cm'], ['m', 'm'], ['in', 'in']] },
    ],
    output: 'Condition',
    colour: 20,
  },
  {
    type: 'kobe_cond_colour',
    message0: 'colour %1 %2',
    args0: [
      { type: 'field_dropdown', name: 'NEG', options: [['is', 'IS'], ['is not', 'NOT']] },
      {
        type: 'field_dropdown', name: 'COLOUR',
        options: [['red', 'red'], ['orange', 'orange'], ['yellow', 'yellow'], ['green', 'green'],
                  ['blue', 'blue'], ['indigo', 'indigo'], ['violet', 'violet'], ['white', 'white'],
                  ['black', 'black'], ['none', 'none']],
      },
    ],
    output: 'Condition',
    colour: 20,
  },
  { type: 'kobe_cond_touch', message0: 'touch pressed', output: 'Condition', colour: 20 },
  { type: 'kobe_cond_ir_detected', message0: 'IR detected', output: 'Condition', colour: 20 },
  {
    type: 'kobe_cond_ir_signal',
    message0: 'IR signal %1 %2',
    args0: [
      { type: 'field_dropdown', name: 'CMP', options: CMP_OPTIONS },
      { type: 'field_number', name: 'VAL', value: 50 },
    ],
    output: 'Condition',
    colour: 20,
  },
  { type: 'kobe_cond_uv_detected', message0: 'UV detected', output: 'Condition', colour: 20 },
  {
    type: 'kobe_cond_uv_index',
    message0: 'UV index %1 %2',
    args0: [
      { type: 'field_dropdown', name: 'CMP', options: CMP_OPTIONS },
      { type: 'field_number', name: 'VAL', value: 5 },
    ],
    output: 'Condition',
    colour: 20,
  },
  {
    type: 'kobe_cond_gyro',
    message0: 'gyro tilt %1 %2 deg',
    args0: [
      { type: 'field_dropdown', name: 'CMP', options: CMP_OPTIONS },
      { type: 'field_number', name: 'VAL', value: 30 },
    ],
    output: 'Condition',
    colour: 20,
  },
  {
    type: 'kobe_cond_sound',
    message0: 'sound %1 %2 db',
    args0: [
      { type: 'field_dropdown', name: 'CMP', options: CMP_OPTIONS },
      { type: 'field_number', name: 'VAL', value: 70 },
    ],
    output: 'Condition',
    colour: 20,
  },
  {
    type: 'kobe_cond_and',
    message0: '%1 and %2',
    args0: [
      { type: 'input_value', name: 'LEFT', check: 'Condition' },
      { type: 'input_value', name: 'RIGHT', check: 'Condition' },
    ],
    inputsInline: true,
    output: 'Condition',
    colour: 0,
  },
  {
    type: 'kobe_cond_or',
    message0: '%1 or %2',
    args0: [
      { type: 'input_value', name: 'LEFT', check: 'Condition' },
      { type: 'input_value', name: 'RIGHT', check: 'Condition' },
    ],
    inputsInline: true,
    output: 'Condition',
    colour: 0,
  },
  {
    type: 'kobe_cond_not',
    message0: 'not %1',
    args0: [{ type: 'input_value', name: 'OPERAND', check: 'Condition' }],
    inputsInline: true,
    output: 'Condition',
    colour: 0,
  },
];

let registered = false;

export function registerKobeBlocks() {
  if (registered) return;
  Blockly.defineBlocksWithJsonArray(BLOCK_DEFS);
  registered = true;
}

export const TOOLBOX = {
  kind: 'categoryToolbox',
  contents: [
    {
      kind: 'category', name: 'Program', colour: '210',
      contents: [
        { kind: 'block', type: 'kobe_algorithm' },
        { kind: 'block', type: 'kobe_hardware' },
        { kind: 'block', type: 'kobe_policy' },
      ],
    },
    {
      kind: 'category', name: 'Actions', colour: '160',
      contents: [
        { kind: 'block', type: 'kobe_move' },
        { kind: 'block', type: 'kobe_turn' },
        { kind: 'block', type: 'kobe_stop' },
        { kind: 'block', type: 'kobe_wait' },
        { kind: 'block', type: 'kobe_break' },
      ],
    },
    {
      kind: 'category', name: 'Control', colour: '290',
      contents: [
        { kind: 'block', type: 'kobe_loop_for' },
        { kind: 'block', type: 'kobe_loop_until' },
        { kind: 'block', type: 'kobe_if' },
        { kind: 'block', type: 'kobe_observe' },
        { kind: 'block', type: 'kobe_branch' },
      ],
    },
    {
      kind: 'category', name: 'Conditions', colour: '20',
      contents: [
        { kind: 'block', type: 'kobe_cond_dist' },
        { kind: 'block', type: 'kobe_cond_colour' },
        { kind: 'block', type: 'kobe_cond_touch' },
        { kind: 'block', type: 'kobe_cond_ir_detected' },
        { kind: 'block', type: 'kobe_cond_ir_signal' },
        { kind: 'block', type: 'kobe_cond_uv_detected' },
        { kind: 'block', type: 'kobe_cond_uv_index' },
        { kind: 'block', type: 'kobe_cond_gyro' },
        { kind: 'block', type: 'kobe_cond_sound' },
        { kind: 'block', type: 'kobe_cond_and' },
        { kind: 'block', type: 'kobe_cond_or' },
        { kind: 'block', type: 'kobe_cond_not' },
      ],
    },
  ],
};

// A starter workspace equivalent to the old DEFAULT_SOURCE text example, so
// the IDE opens with something already on the canvas.
export const DEFAULT_WORKSPACE_XML = `
<xml xmlns="https://developers.google.com/blockly/xml">
  <block type="kobe_algorithm" x="20" y="20">
    <field name="NAME">SAC</field>
  </block>
  <block type="kobe_hardware" x="20" y="90">
    <field name="TARGET">EV3</field>
    <field name="MOTORS">A, B</field>
    <field name="SENSORS">dist@1, colour@2</field>
  </block>
  <block type="kobe_policy" x="20" y="230">
    <field name="CURIOSITY">0.3</field>
    <field name="SAFETY">0.6</field>
    <field name="COMFORT">0.5</field>
    <field name="EFFICIENCY">0.4</field>
  </block>
  <block type="kobe_move" x="20" y="430">
    <field name="ACTION">walk</field>
    <field name="DIRECTION">forward</field>
    <field name="SPEED">slowly</field>
    <next>
      <block type="kobe_observe">
        <field name="SENSORS">dist, colour</field>
        <statement name="BRANCHES">
          <block type="kobe_branch">
            <value name="CONDITION">
              <block type="kobe_cond_dist">
                <field name="CMP">&lt;</field>
                <field name="VAL">20</field>
                <field name="UNIT">cm</field>
              </block>
            </value>
            <statement name="THEN">
              <block type="kobe_stop"></block>
            </statement>
            <next>
              <block type="kobe_branch">
                <value name="CONDITION">
                  <block type="kobe_cond_colour">
                    <field name="NEG">IS</field>
                    <field name="COLOUR">red</field>
                  </block>
                </value>
                <statement name="THEN">
                  <block type="kobe_stop"></block>
                </statement>
              </block>
            </next>
          </block>
        </statement>
        <next>
          <block type="kobe_turn">
            <field name="DIRECTION">left</field>
            <next>
              <block type="kobe_move">
                <field name="ACTION">walk</field>
                <field name="DIRECTION">forward</field>
                <field name="SPEED">normally</field>
              </block>
            </next>
          </block>
        </next>
      </block>
    </next>
  </block>
</xml>
`;
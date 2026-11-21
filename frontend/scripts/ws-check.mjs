import { io } from 'socket.io-client'

const URL = process.argv[2] || 'http://localhost:5173'
const socket = io(URL, { transports: ['polling', 'websocket'] })

function fail(msg) {
  console.log('FAIL:', msg)
  socket.disconnect()
  process.exit(1)
}

function isMatrix(m, rows, cols) {
  return Array.isArray(m) && m.length === rows && m.every((r) => Array.isArray(r) && r.length === cols)
}

const timer = setTimeout(() => fail('timed out after 20 seconds'), 20000)

socket.on('connect_error', (e) => fail('connect_error: ' + e.message))

socket.on('connect', () => {
  console.log('connected through', URL, '| first transport:', socket.io.engine.transport.name)
  socket.io.engine.on('upgrade', (t) => console.log('upgraded to:', t.name))
  socket.emit('new_game', { difficulty: 'easy', human_first: false })
})

socket.on('error', (e) => fail('server error event: ' + JSON.stringify(e)))

socket.on('ai_move', (ev) => {
  const b = ev.brain
  if (!b) fail('ai_move had no brain data')
  if (b.q_values.length !== 7) fail('q_values length is ' + b.q_values.length)
  if (b.input.length !== 84) fail('input length is ' + b.input.length)
  if (b.hidden1.length !== 12 || b.hidden2.length !== 12) fail('hidden sizes are wrong')

  // Fields the Brain panel needs (added in the newer brain.py)
  if (!Array.isArray(b.idx_h1) || b.idx_h1.length !== 12) fail('idx_h1 is missing or the wrong size')
  if (!Array.isArray(b.idx_h2) || b.idx_h2.length !== 12) fail('idx_h2 is missing or the wrong size')
  if (!isMatrix(b.contrib_h1_h2, 12, 12)) fail('contrib_h1_h2 is missing or not 12x12 (the server may be running the old brain.py)')
  if (!isMatrix(b.contrib_h2_out, 12, 7)) fail('contrib_h2_out is missing or not 12x7')
  if (typeof b.active_h1 !== 'number' || typeof b.active_h2 !== 'number') fail('active_h1 / active_h2 missing')
  for (const k of ['coverage_h1', 'coverage_out']) {
    if (typeof b[k] !== 'number' || b[k] < 0 || b[k] > 1) fail(k + ' missing or outside 0 to 1')
  }

  console.log('ai_move: column', ev.column, '| row', ev.row, '| player', ev.player)
  console.log('  q_values:', b.q_values.map((v) => v.toFixed(3)).join(' '))
  console.log('  neuron picks: layer 1', b.idx_h1.join(','), '| layer 2', b.idx_h2.join(','))
  console.log('  active neurons:', b.active_h1 + '/128 and', b.active_h2 + '/128')
  console.log('  coverage: layer 1->2', Math.round(b.coverage_h1 * 100) + '% | layer 2->output', Math.round(b.coverage_out * 100) + '%')
  setTimeout(() => {
    console.log('final transport:', socket.io.engine.transport.name)
    console.log('PASS')
    clearTimeout(timer)
    socket.disconnect()
    process.exit(0)
  }, 1500)
})

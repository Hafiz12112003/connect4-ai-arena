import { useMemo } from 'react'
import { motion, useReducedMotion } from 'framer-motion'

// Drawing space. The SVG scales to the panel width.
const W = 440
const H = 330
const IN_X0 = 5
const IN_Y0 = 127.5
const IN_STEP = 15
const FAN_X = IN_X0 + 6 * IN_STEP + 8
const FAN_Y = 165
const X1 = 165
const X2 = 270
const XO = 380
const R_H = 7
const R_O = 11
const N = 12

const hy = (i) => 30 + i * 24.5
const oy = (c) => 30 + c * (269.5 / 6)

const COLS = [0, 1, 2, 3, 4, 5, 6]
const SLOTS = Array.from({ length: N }, (_, i) => i)
const CELLS = Array.from({ length: 42 }, (_, k) => k)
const ZEROS = Array(N).fill(0)

const HEADS = [
  { x: 50, name: 'Input', sub: '84 values' },
  { x: X1, name: 'Layer 1', sub: '12 of 128' },
  { x: X2, name: 'Layer 2', sub: '12 of 128' },
  { x: XO, name: 'Output', sub: '7 columns' },
]

function maxAbs(matrix) {
  let a = 0
  for (const row of matrix) for (const v of row) a = Math.max(a, Math.abs(v))
  return a
}

// Every line comes from a real number sent by the server:
// source activation x trained weight. Very weak lines are skipped.
function buildLines(brain) {
  if (!brain) return { fan: [], l12: [], l2o: [] }

  const m1 = Math.max(...brain.hidden1, 1e-9)
  const fan = []
  brain.hidden1.forEach((v, j) => {
    const s = (v / m1) * 0.7
    if (s >= 0.02) fan.push({ key: `f${j}`, x1: FAN_X, y1: FAN_Y, x2: X1 - R_H, y2: hy(j), s, neg: false })
  })

  const l12 = []
  const mm12 = Math.max(maxAbs(brain.contrib_h1_h2), 1e-9)
  brain.contrib_h1_h2.forEach((row, i) => {
    row.forEach((v, j) => {
      const s = Math.abs(v) / mm12
      if (s >= 0.05) {
        l12.push({ key: `a${i}-${j}`, x1: X1 + R_H, y1: hy(i), x2: X2 - R_H, y2: hy(j), s, neg: v < 0 })
      }
    })
  })

  const l2o = []
  const mm2o = Math.max(maxAbs(brain.contrib_h2_out), 1e-9)
  brain.contrib_h2_out.forEach((row, i) => {
    row.forEach((v, c) => {
      const s = Math.abs(v) / mm2o
      if (s >= 0.05) {
        l2o.push({ key: `b${i}-${c}`, x1: X2 + R_H, y1: hy(i), x2: XO - R_O, y2: oy(c), s, neg: v < 0 })
      }
    })
  })

  return { fan, l12, l2o }
}

function Wire({ l, delay, reduce }) {
  return (
    <motion.line
      x1={l.x1}
      y1={l.y1}
      x2={l.x2}
      y2={l.y2}
      strokeWidth={0.5 + 2 * l.s}
      strokeLinecap="round"
      style={{ stroke: l.neg ? 'var(--danger)' : 'var(--ai)' }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 0.15 + 0.8 * l.s }}
      transition={reduce ? { duration: 0 } : { duration: 0.35, delay }}
    />
  )
}

function Network({ brain, fresh, pulsing, reduce }) {
  const lines = useMemo(() => buildLines(brain), [brain])
  const h1 = brain ? brain.hidden1 : ZEROS
  const h2 = brain ? brain.hidden2 : ZEROS
  const m1 = Math.max(...h1, 1e-9)
  const m2 = Math.max(...h2, 1e-9)

  let qMin = 0
  let qRange = 0
  if (brain) {
    const qs = brain.legal.map((c) => brain.q_values[c])
    qMin = Math.min(...qs)
    qRange = Math.max(...qs) - qMin
  }
  const rel = (c) => (qRange < 1e-9 ? 0.5 : (brain.q_values[c] - qMin) / qRange)
  const fade = reduce ? { duration: 0 } : { duration: 0.4 }

  const label = brain
    ? `Neural network diagram for the AI move. It chose column ${brain.column + 1}. The numbers are listed under the diagram.`
    : 'Neural network diagram. It fills in after the AI moves.'

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block w-full" role="img" aria-label={label}>
      <defs>
        <linearGradient id="brain-sweep" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0" style={{ stopColor: 'var(--ai)', stopOpacity: 0 }} />
          <stop offset="0.5" style={{ stopColor: 'var(--ai)', stopOpacity: 0.35 }} />
          <stop offset="1" style={{ stopColor: 'var(--ai)', stopOpacity: 0 }} />
        </linearGradient>
      </defs>

      <motion.g
        initial={false}
        animate={{ opacity: fresh || !brain ? 1 : 0.45 }}
        transition={{ duration: 0.3 }}
      >
        {/* Lines. The group is keyed, so every AI move lights them up again, layer by layer. */}
        <g key={brain ? brain.moveCount : 'none'}>
          {lines.fan.map((l) => (
            <Wire key={l.key} l={l} delay={0} reduce={reduce} />
          ))}
          {lines.l12.map((l) => (
            <Wire key={l.key} l={l} delay={0.25} reduce={reduce} />
          ))}
          {lines.l2o.map((l) => (
            <Wire key={l.key} l={l} delay={0.5} reduce={reduce} />
          ))}
        </g>

        {/* Input: the board as the AI saw it. Plane 0 is the AI, plane 1 is the opponent. */}
        {CELLS.map((k) => {
          const r = Math.floor(k / 7)
          const c = k % 7
          const mine = brain ? brain.input[k] > 0.5 : false
          const theirs = brain ? brain.input[42 + k] > 0.5 : false
          const fill = mine
            ? 'var(--ai)'
            : theirs
              ? 'var(--human)'
              : 'color-mix(in srgb, var(--muted) 35%, transparent)'
          return <circle key={k} cx={IN_X0 + c * IN_STEP} cy={IN_Y0 + r * IN_STEP} r={4.8} style={{ fill }} />
        })}

        {/* Layer 1: the 12 strongest neurons for this move, strongest at the top */}
        {SLOTS.map((i) => {
          const v = h1[i]
          return (
            <g key={`n1-${i}`}>
              <circle cx={X1} cy={hy(i)} r={R_H} strokeWidth="1.5" style={{ fill: 'var(--panel)', stroke: 'var(--line)' }} />
              <motion.circle
                cx={X1}
                cy={hy(i)}
                r={R_H}
                style={{ fill: 'var(--ai)' }}
                initial={false}
                animate={{ opacity: v > 0 ? 0.25 + 0.75 * (v / m1) : 0 }}
                transition={fade}
              />
              {brain && <title>{`Layer 1 neuron ${brain.idx_h1[i]}: activation ${v.toFixed(2)}`}</title>}
            </g>
          )
        })}

        {/* Layer 2 */}
        {SLOTS.map((i) => {
          const v = h2[i]
          return (
            <g key={`n2-${i}`}>
              <circle cx={X2} cy={hy(i)} r={R_H} strokeWidth="1.5" style={{ fill: 'var(--panel)', stroke: 'var(--line)' }} />
              <motion.circle
                cx={X2}
                cy={hy(i)}
                r={R_H}
                style={{ fill: 'var(--ai)' }}
                initial={false}
                animate={{ opacity: v > 0 ? 0.25 + 0.75 * (v / m2) : 0 }}
                transition={fade}
              />
              {brain && <title>{`Layer 2 neuron ${brain.idx_h2[i]}: activation ${v.toFixed(2)}`}</title>}
            </g>
          )
        })}

        {/* Output: one node per column. Full columns are dashed. */}
        {COLS.map((c) => {
          const legal = !brain || brain.legal.includes(c)
          const chosen = !!brain && brain.column === c
          return (
            <g key={`o-${c}`}>
              {chosen && fresh && !reduce && (
                <motion.circle
                  cx={XO}
                  cy={oy(c)}
                  r={R_O}
                  fill="none"
                  strokeWidth="2"
                  style={{ stroke: 'var(--ai)' }}
                  animate={{ r: [R_O, R_O + 7], opacity: [0.7, 0] }}
                  transition={{ duration: 1.2, repeat: Infinity, ease: 'easeOut' }}
                />
              )}
              <circle
                cx={XO}
                cy={oy(c)}
                r={R_O}
                strokeWidth={chosen ? 2.5 : 1.5}
                strokeDasharray={legal ? undefined : '3 3'}
                style={{ fill: 'var(--panel)', stroke: chosen ? 'var(--ai)' : 'var(--line)' }}
              />
              <motion.circle
                cx={XO}
                cy={oy(c)}
                r={R_O}
                style={{ fill: 'var(--ai)' }}
                initial={false}
                animate={{ opacity: !brain || !legal ? 0 : chosen ? 1 : 0.12 + 0.45 * rel(c) }}
                transition={fade}
              />
              <text
                x={XO}
                y={oy(c) + 4}
                textAnchor="middle"
                fontSize="12"
                fontWeight="700"
                className="num"
                style={{ fill: chosen ? '#0a0d16' : 'var(--ink)' }}
              >
                {c + 1}
              </text>
              {brain && (
                <title>{legal ? `Column ${c + 1}: Q-value ${brain.q_values[c].toFixed(3)}` : `Column ${c + 1} is full`}</title>
              )}
            </g>
          )
        })}
      </motion.g>

      {/* The thinking pulse: sweeps left to right through the layers while the AI works */}
      {pulsing && (
        <motion.rect
          y={0}
          width={70}
          height={H}
          fill="url(#brain-sweep)"
          style={{ pointerEvents: 'none' }}
          initial={{ x: -70 }}
          animate={{ x: W }}
          transition={{ duration: 0.7, repeat: Infinity, ease: 'linear' }}
        />
      )}
    </svg>
  )
}

export default function BrainPanel({ arena, span }) {
  const { game, brain, thinking } = arena
  const reduce = useReducedMotion()
  const fresh = !!(brain && game && brain.moveCount === game.move_count)
  const pulsing = !!thinking && thinking.who === 'ai' && !reduce

  return (
    <motion.section
      id="brain"
      aria-labelledby="brain-title"
      className={`panel scroll-mt-24 self-start p-6 ${span}`}
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
    >
      <div className="flex items-baseline justify-between gap-4">
        <h2 id="brain-title" className="text-xl font-semibold">AI Brain</h2>
        <span className="label">Real activations</span>
      </div>
      <p className="mt-2 text-sm" style={{ color: 'var(--muted)' }}>
        The network that picked the move, drawn from its real numbers. It is not an animation.
      </p>

      <div className="mt-5 rounded-xl border border-line p-3" style={{ background: 'var(--panel-2)' }}>
        <div className="relative mb-2" style={{ height: 34 }}>
          {HEADS.map((h) => (
            <div
              key={h.name}
              className="absolute text-center"
              style={{ left: `${(h.x / W) * 100}%`, transform: 'translateX(-50%)' }}
            >
              <p className="label" style={{ fontSize: '0.62rem' }}>{h.name}</p>
              <p className="num" style={{ fontSize: '0.62rem', color: 'var(--muted)' }}>{h.sub}</p>
            </div>
          ))}
        </div>
        <Network brain={brain} fresh={fresh} pulsing={pulsing} reduce={reduce} />
      </div>

      <div className="mt-4 space-y-1.5">
        {brain ? (
          <>
            <p className="text-sm">
              <span className="label">Chose</span>{' '}
              <span className="num font-semibold" style={{ color: 'var(--ai)' }}>
                column {brain.column + 1}
              </span>
              {!fresh && (
                <span className="ml-2 text-xs" style={{ color: 'var(--muted)' }}>(previous AI move)</span>
              )}
            </p>
            <p className="num text-xs" style={{ color: 'var(--muted)' }}>
              Active neurons: layer 1 {brain.active_h1}/128 · layer 2 {brain.active_h2}/128
            </p>
            <p className="num text-xs" style={{ color: 'var(--muted)' }}>
              Lines drawn carry {Math.round(brain.coverage_h1 * 100)}% (layer 1 to 2) and{' '}
              {Math.round(brain.coverage_out * 100)}% (layer 2 to output) of the real signal.
            </p>
          </>
        ) : (
          <p className="text-sm" style={{ color: 'var(--muted)' }}>
            Start a game. After the AI moves, its real activations light up here.
          </p>
        )}
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs" style={{ color: 'var(--muted)' }}>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: 'var(--ai)' }} /> AI discs, line raises
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: 'var(--human)' }} /> Opponent discs
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-0.5 w-4 rounded" style={{ background: 'var(--danger)' }} /> Line lowers
        </span>
      </div>
      <p className="mt-3 text-xs" style={{ color: 'var(--muted)' }}>
        Input shows the board as the AI saw it before moving. Each hidden layer shows its 12 strongest
        neurons for this move, strongest at the top. A line is a neuron activation times its trained
        weight. The fan from the board only shows each layer 1 neuron activation, because no per-line
        weights are drawn there.
      </p>
    </motion.section>
  )
}

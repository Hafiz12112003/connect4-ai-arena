import { useMemo } from 'react'
import { motion, useReducedMotion } from 'framer-motion'

const AMBER = '#ffb347'
const CYAN = '#3ef0d0'
const WHITE = '#ffffff'

// Small seeded random generator. The same seed always gives the same burst,
// so the component stays predictable and React never sees a changing result.
function seeded(seed) {
  let a = seed
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function makeParticles(count, colors) {
  const rand = seeded(7)
  return Array.from({ length: count }, (_, i) => {
    const angle = -Math.PI / 2 + (rand() - 0.5) * Math.PI * 1.4
    const speed = 150 + rand() * 280
    return {
      id: i,
      x: Math.cos(angle) * speed,
      up: Math.sin(angle) * speed,
      fall: 240 + rand() * 260,
      rotate: (rand() - 0.5) * 900,
      size: 6 + rand() * 7,
      delay: rand() * 0.18,
      duration: 1.7 + rand() * 0.8,
      color: colors[i % colors.length],
      round: rand() > 0.5,
    }
  })
}

function describe(over) {
  if (over.draw) {
    return { title: 'Draw', sub: 'The board is full.', accent: 'var(--ink)', burst: false }
  }
  const who = over.winner_who
  if (who === 'human') return { title: 'You win!', sub: 'Four in a row.', accent: 'var(--human)', burst: true }
  if (who === 'ai') return { title: 'The AI wins', sub: 'Four in a row.', accent: 'var(--ai)', burst: true }
  if (who === 'minimax') return { title: 'Minimax wins', sub: 'Four in a row.', accent: 'var(--human)', burst: true }
  if (who === 'llm') return { title: 'The LLM wins', sub: 'Four in a row.', accent: 'var(--human)', burst: true }
  return { title: 'Game over', sub: '', accent: 'var(--ink)', burst: false }
}

export default function Celebration({ over, onPlayAgain }) {
  const reduce = useReducedMotion()
  const info = describe(over)
  const aiWon = over.winner_who === 'ai'

  const particles = useMemo(
    () => makeParticles(48, aiWon ? [CYAN, WHITE, CYAN, '#7df5e0'] : [AMBER, WHITE, AMBER, '#ffd08a']),
    [aiWon],
  )

  return (
    <>
      {info.burst && !reduce && (
        <div
          className="pointer-events-none absolute inset-0 z-10 overflow-hidden"
          aria-hidden="true"
        >
          {particles.map((p) => (
            <motion.span
              key={p.id}
              className="absolute block"
              style={{
                left: '50%',
                top: '35%',
                width: p.size,
                height: p.round ? p.size : p.size * 0.5,
                borderRadius: p.round ? '50%' : 2,
                background: p.color,
              }}
              initial={{ x: 0, y: 0, opacity: 1, rotate: 0 }}
              animate={{
                x: [0, p.x, p.x * 1.2],
                y: [0, p.up, p.up + p.fall],
                opacity: [1, 1, 0],
                rotate: p.rotate,
              }}
              transition={{
                duration: p.duration,
                delay: p.delay,
                times: [0, 0.35, 1],
                ease: 'easeOut',
              }}
            />
          ))}
        </div>
      )}

      <motion.div
        role="status"
        className="absolute inset-x-0 top-3 z-20 flex justify-center px-3"
        initial={{ opacity: 0, y: -24, scale: 0.9 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ type: 'spring', stiffness: 260, damping: 20, delay: 0.1 }}
      >
        <div
          className="panel flex items-center gap-5 px-5 py-3"
          style={{
            borderColor: info.accent,
            boxShadow: `0 10px 40px -10px color-mix(in srgb, ${info.accent} 55%, transparent)`,
          }}
        >
          <div>
            <p className="label">Game over</p>
            <p className="text-xl font-bold leading-tight" style={{ color: info.accent }}>
              {info.title}
            </p>
            {info.sub && (
              <p className="text-xs" style={{ color: 'var(--muted)' }}>
                {info.sub}
              </p>
            )}
          </div>
          {onPlayAgain && (
            <button
              type="button"
              onClick={onPlayAgain}
              className="rounded-full px-4 py-2 text-sm font-semibold transition-transform hover:-translate-y-0.5"
              style={{ background: 'var(--human)', color: '#0a0d16' }}
            >
              Play again
            </button>
          )}
        </div>
      </motion.div>
    </>
  )
}

import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'

const ROWS = 6
const COLS = 7
const GAP = 8 // space between holes, in px
const PAD = 12 // space between board edge and holes, in px

// The board face is dark navy in both light and dark mode,
// so the discs use fixed bright colours.
const AMBER = '#ffb347'
const CYAN = '#3ef0d0'

const EMPTY = Array.from({ length: ROWS }, () => Array(COLS).fill(0))
const COLUMNS = Array.from({ length: COLS }, (_, c) => c)
const ROW_LIST = Array.from({ length: ROWS }, (_, r) => r)
const GRID = { gridTemplateColumns: `repeat(${COLS}, 1fr)`, gap: GAP }

function discBg(color) {
  return `radial-gradient(circle at 35% 30%, color-mix(in srgb, ${color} 55%, white), ${color} 55%, color-mix(in srgb, ${color} 65%, black))`
}

function Disc({ color, isNew, drop, winning }) {
  return (
    <motion.div
      className="absolute inset-0 rounded-full"
      style={{ zIndex: 2 }}
      initial={isNew ? { y: -drop } : false}
      animate={{ y: 0 }}
      transition={{ type: 'spring', stiffness: 260, damping: 15, mass: 0.8 }}
    >
      <motion.div
        className="h-full w-full rounded-full"
        animate={winning ? { scale: [1, 1.12, 1] } : { scale: 1 }}
        transition={
          winning
            ? { duration: 1.1, repeat: Infinity, ease: 'easeInOut' }
            : { duration: 0.2 }
        }
        style={{
          background: discBg(color),
          boxShadow: winning
            ? `0 0 22px 6px ${color}, inset 0 -3px 6px rgba(0,0,0,0.25)`
            : 'inset 0 -3px 6px rgba(0,0,0,0.3), 0 2px 4px rgba(0,0,0,0.35)',
        }}
      />
    </motion.div>
  )
}

// Shows where a disc will land if you click this column.
function LandingRing({ color }) {
  return (
    <motion.div
      className="absolute rounded-full"
      style={{
        inset: '10%',
        zIndex: 1,
        border: `3px solid ${color}`,
        boxShadow: `0 0 14px 2px ${color}`,
        pointerEvents: 'none',
      }}
      initial={{ opacity: 0 }}
      animate={{ opacity: [0.4, 0.9, 0.4] }}
      transition={{ duration: 1.2, repeat: Infinity, ease: 'easeInOut' }}
      aria-hidden="true"
    />
  )
}

function HeaderSlot({ cell, index, ghostColor, chosen, waving }) {
  return (
    <div
      className="flex flex-col items-center justify-center"
      style={{ height: cell }}
    >
      {ghostColor ? (
        <div
          className="w-full rounded-full"
          style={{ aspectRatio: '1', background: discBg(ghostColor), opacity: 0.85 }}
        />
      ) : (
        <>
          <motion.span
            className="block rounded-full"
            style={{
              width: chosen ? 14 : 8,
              height: chosen ? 14 : 8,
              background: chosen
                ? CYAN
                : 'color-mix(in srgb, var(--muted) 55%, transparent)',
              boxShadow: chosen ? `0 0 14px 3px ${CYAN}` : 'none',
            }}
            animate={waving ? { opacity: [0.25, 1, 0.25] } : { opacity: 1 }}
            transition={
              waving
                ? { duration: 1, repeat: Infinity, delay: index * 0.1 }
                : { duration: 0.2 }
            }
          />
          {chosen && (
            <motion.svg
              width="20"
              height="20"
              viewBox="0 0 18 18"
              fill="none"
              style={{ marginTop: 4 }}
              animate={{ y: [0, 4, 0] }}
              transition={{ duration: 0.9, repeat: Infinity, ease: 'easeInOut' }}
            >
              <polyline
                points="3,6 9,12 15,6"
                stroke={CYAN}
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </motion.svg>
          )}
        </>
      )}
    </div>
  )
}

function QBar({ q, pct, best, legal }) {
  return (
    <div className="flex flex-col items-center">
      <div className="flex items-end justify-center" style={{ height: 56, width: '100%' }}>
        {legal && (
          <motion.div
            initial={false}
            animate={{ height: `${pct}%` }}
            transition={{ type: 'spring', stiffness: 160, damping: 20 }}
            style={{
              width: '70%',
              maxWidth: 30,
              minHeight: 4,
              borderRadius: 6,
              background: best
                ? 'var(--ai)'
                : 'color-mix(in srgb, var(--muted) 45%, transparent)',
              boxShadow: best ? '0 0 12px color-mix(in srgb, var(--ai) 60%, transparent)' : 'none',
            }}
          />
        )}
      </div>
      <span
        className="num mt-1 text-[9px]"
        style={{ color: best ? 'var(--ai)' : 'var(--muted)' }}
      >
        {legal ? q.toFixed(3) : '–'}
      </span>
    </div>
  )
}

export default function Board({ game, lastMove, brain, thinking, canMove, onMove, over }) {
  const gridRef = useRef(null)
  const [cell, setCell] = useState(56)
  const [hoverCol, setHoverCol] = useState(null)
  const [focusCol, setFocusCol] = useState(3)
  const [focused, setFocused] = useState(false)

  // Measure the hole size so a falling disc knows how far to drop.
  useEffect(() => {
    const el = gridRef.current
    if (!el) return undefined
    const measure = () => setCell((el.clientWidth - GAP * (COLS - 1)) / COLS)
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const board = game ? game.board : EMPTY
  const legal = game ? game.legal : []
  const amberPlayer = game
    ? typeof game.human === 'number'
      ? game.human
      : -game.ai
    : 1
  const colorOf = (p) => (p === amberPlayer ? AMBER : CYAN)

  const winCells = (over && over.winning_cells) || (game && game.winning_cells) || []
  const winSet = new Set(winCells.map(([r, c]) => `${r}-${c}`))

  const playable = (c) => canMove && legal.includes(c)

  let ghostCol = null
  if (canMove) {
    const wanted = hoverCol !== null ? hoverCol : focused ? focusCol : null
    if (wanted !== null && legal.includes(wanted)) ghostCol = wanted
  }

  // The lowest empty square in the column you are aiming at.
  let landingRow = -1
  if (ghostCol !== null) {
    for (let r = ROWS - 1; r >= 0; r--) {
      if (board[r][ghostCol] === 0) {
        landingRow = r
        break
      }
    }
  }

  const fresh = !!(brain && game && brain.moveCount === game.move_count)
  const chosenCol = fresh ? brain.column : null
  const waving = !!(thinking && thinking.who === 'ai')

  // Q-bars are scaled per move: the Q-values of different brains have very different sizes.
  let qMin = 0
  let qRange = 0
  if (brain) {
    const qs = brain.legal.map((c) => brain.q_values[c])
    qMin = Math.min(...qs)
    qRange = Math.max(...qs) - qMin
  }
  const barPct = (c) => (qRange < 1e-9 ? 50 : 15 + (85 * (brain.q_values[c] - qMin)) / qRange)

  const onKeyDown = (e) => {
    if (e.key === 'ArrowLeft') {
      e.preventDefault()
      setFocusCol((c) => Math.max(0, c - 1))
    } else if (e.key === 'ArrowRight') {
      e.preventDefault()
      setFocusCol((c) => Math.min(COLS - 1, c + 1))
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      if (playable(focusCol)) onMove(focusCol)
    }
  }

  let status = ''
  if (over) {
    status = over.winner_who === 'human' ? 'You win.' : over.winner_who === 'ai' ? 'The AI wins.' : over.draw ? 'Draw.' : over.winner_who === 'minimax' ? 'Minimax wins.' : over.winner_who === 'llm' ? 'The LLM wins.' : 'Game over.'
  } else if (canMove) {
    status = `Your turn. Column ${focusCol + 1} selected.`
  } else if (thinking) {
    status = 'The AI is thinking.'
  }

  return (
    <div className="mx-auto w-full" style={{ maxWidth: 560 }}>
      <div
        tabIndex={0}
        role="group"
        aria-label="Connect Four board. Use the left and right arrow keys to choose a column and press Enter to drop a disc."
        onKeyDown={onKeyDown}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
      >
        {/* Row above the board: ghost disc, AI choice arrow, thinking pulse */}
        <div className="grid" style={{ ...GRID, paddingInline: PAD, marginBottom: GAP }} aria-hidden="true">
          {COLUMNS.map((c) => (
            <HeaderSlot
              key={c}
              cell={cell}
              index={c}
              ghostColor={ghostCol === c ? colorOf(amberPlayer) : null}
              chosen={chosenCol === c}
              waving={waving}
            />
          ))}
        </div>

        {/* The board */}
        <div className="board-surface" style={{ padding: PAD, borderRadius: 20 }}>
          <div ref={gridRef} className="grid" style={GRID}>
            {COLUMNS.map((c) => (
              <button
                key={c}
                type="button"
                tabIndex={-1}
                aria-label={`Drop a disc in column ${c + 1}`}
                aria-disabled={!playable(c)}
                onClick={() => playable(c) && onMove(c)}
                onPointerEnter={(e) => e.pointerType === 'mouse' && setHoverCol(c)}
                onPointerLeave={(e) => e.pointerType === 'mouse' && setHoverCol(null)}
                className="flex flex-col"
                style={{
                  gap: GAP,
                  padding: 0,
                  border: 0,
                  borderRadius: 12,
                  background: hoverCol === c && playable(c) ? 'rgba(255,255,255,0.07)' : 'transparent',
                  transition: 'background 0.15s',
                  cursor: playable(c) ? 'pointer' : 'default',
                }}
              >
                {ROW_LIST.map((r) => {
                  const v = board[r][c]
                  return (
                    <div
                      key={r}
                      className="relative w-full rounded-full"
                      style={{
                        aspectRatio: '1',
                        background: 'rgba(4, 8, 24, 0.55)',
                        boxShadow: 'inset 0 3px 8px rgba(0,0,0,0.6), 0 1px 0 rgba(120,160,255,0.15)',
                      }}
                    >
                      {ghostCol === c && landingRow === r && (
                        <LandingRing color={colorOf(amberPlayer)} />
                      )}
                      {v !== 0 && (
                        <Disc
                          color={colorOf(v)}
                          isNew={!!lastMove && lastMove.row === r && lastMove.column === c}
                          drop={(r + 1) * (cell + GAP)}
                          winning={winSet.has(`${r}-${c}`)}
                        />
                      )}
                    </div>
                  )
                })}
              </button>
            ))}
          </div>
        </div>

        {/* Q-value bars under each column */}
        <div style={{ minHeight: 96, marginTop: 12 }} aria-hidden="true">
          {brain && (
            <div style={{ opacity: fresh ? 1 : 0.55, transition: 'opacity 0.3s' }}>
              <div className="grid" style={{ ...GRID, paddingInline: PAD }}>
                {COLUMNS.map((c) => (
                  <QBar
                    key={c}
                    q={brain.q_values[c]}
                    pct={brain.legal.includes(c) ? barPct(c) : 0}
                    best={brain.column === c}
                    legal={brain.legal.includes(c)}
                  />
                ))}
              </div>
              <p className="label mt-1 text-center">AI Q-values for its last move · higher is better</p>
            </div>
          )}
        </div>
      </div>
      <p className="sr-only" aria-live="polite">
        {status}
      </p>
    </div>
  )
}

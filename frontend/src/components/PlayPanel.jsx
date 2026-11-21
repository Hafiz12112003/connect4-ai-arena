import { motion } from 'framer-motion'
import { useState } from 'react'
import Board from './Board.jsx'
import Celebration from './Confetti.jsx'

const LEVELS = [
  { id: 'easy', label: 'Easy', note: 'Early brain. Beats Random, loses to Minimax depth 1.' },
  { id: 'medium', label: 'Medium', note: 'Mid-training brain. Beat Minimax depth 1 about 58% in my tests.' },
  { id: 'hard', label: 'Hard', note: 'Final brain. Beat Minimax depth 1 about 77% in my tests.' },
]

const MODES = [
  { id: 'human', label: 'Human vs AI' },
  { id: 'minimax', label: 'AI vs Minimax' },
  { id: 'llm', label: 'AI vs LLM' },
]

const DEPTHS = [
  { id: 'minimax1', label: 'Depth 1' },
  { id: 'minimax2', label: 'Depth 2' },
]

const COLOURS = [
  { id: null, label: 'Alternate' },
  { id: true, label: 'AI first' },
  { id: false, label: 'AI second' },
]

const SPEEDS = [0.5, 1, 2, 5, 10]
const SPEED_LABELS = ['0.5×', '1×', '2×', '5×', '10×']

function Segmented({ label, value, options, onChange }) {
  return (
    <div>
      <p className="label mb-2">{label}</p>
      <div role="radiogroup" aria-label={label} className="flex gap-1 rounded-full border border-line p-1" style={{ background: 'var(--panel-2)' }}>
        {options.map((o) => {
          const active = o.id === value
          return (
            <button
              key={String(o.id)}
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => onChange(o.id)}
              className="flex-1 rounded-full px-2 py-2 text-xs font-semibold transition-colors sm:px-4 sm:text-sm"
              style={{
                background: active ? 'var(--human)' : 'transparent',
                color: active ? '#0a0d16' : 'var(--ink)',
              }}
            >
              {o.label}
            </button>
          )
        })}
      </div>
    </div>
  )
}

function statusText(arena) {
  const { connection, game, over, thinking, canMove, watching, stopped } = arena
  if (connection === 'connecting') return 'Connecting to the server. Free hosting sleeps when idle, so the first visit can take about a minute.'
  if (connection === 'offline') return 'Cannot reach the server. If it was asleep it may still be waking up. This retries on its own.'
  if (!game) return 'Pick your settings and press Start.'
  if (over) return 'Game over.'
  if (stopped) return 'Match stopped.'
  if (thinking) return thinking.who === 'llm' ? 'LLM is thinking...' : 'The AI is thinking...'
  if (canMove) return 'Your turn.'
  if (watching) return 'Watching the match...'
  return 'Sending your move...'
}

export default function PlayPanel({ arena, span }) {
  const {
    connection, game, brain, lastMove, thinking, over, error, canMove, watching,
    speed, llmStats, newGame, startMatch, stopMatch, setSpeed, humanMove, clearError,
  } = arena

  const [mode, setMode] = useState('human')
  const [difficulty, setDifficulty] = useState('medium')
  const [humanFirst, setHumanFirst] = useState(true)
  const [minimaxDepth, setMinimaxDepth] = useState('minimax2')
  const [aiLevel, setAiLevel] = useState('hard')
  const [aiFirst, setAiFirst] = useState(null)

  const online = connection === 'online'
  const isMatchMode = mode !== 'human'
  const hasHumanGame = !!game && game.mode === 'human_ai'
  const hasMatchGame = !!game && game.mode !== 'human_ai'
  const level = mode === 'human' ? difficulty : aiLevel
  const levelNote = LEVELS.find((l) => l.id === level).note

  let note = `${levelNote} The AI always plays its best move, so it never varies. The colour you choose matters.`
  if (mode === 'minimax') {
    note = `${levelNote} Minimax answers instantly. Pick Alternate to watch both colours: in my 100-game test per colour, the Hard brain won every game against depth 2 as first player and lost every game as second.`
  } else if (mode === 'llm') {
    note = `${levelNote} The LLM (gpt-oss-20b) plays through a free API with rate limits. Its moves can take several seconds, and in my saved runs about 1 in 5 of its moves fell back to a random move. Each LLM move uses one request of a free daily limit (1,000 on my key).`
  }

  const changeMode = (m) => {
    if (watching) stopMatch()
    setMode(m)
  }

  const start = () => {
    if (mode === 'human') {
      newGame({ difficulty, humanFirst })
    } else {
      startMatch({
        opponent: mode === 'llm' ? 'llm' : minimaxDepth,
        difficulty: aiLevel,
        aiFirst,
        speed,
      })
    }
  }

  let buttonLabel = hasHumanGame ? 'New game' : 'Start game'
  let onButton = start
  let buttonStyle = { background: 'var(--ai)', color: '#0a0d16' }
  if (isMatchMode) {
    if (watching) {
      buttonLabel = 'Stop match'
      onButton = stopMatch
      buttonStyle = { background: 'transparent', color: 'var(--danger)', border: '2px solid var(--danger)' }
    } else {
      buttonLabel = hasMatchGame ? 'Watch another match' : 'Start match'
    }
  }

  let speedIdx = 0
  SPEEDS.forEach((v, i) => {
    if (Math.abs(v - speed) < Math.abs(SPEEDS[speedIdx] - speed)) speedIdx = i
  })

  let opponentName = 'You'
  if (hasMatchGame) {
    opponentName = game.opponent === 'llm' ? 'LLM' : `Minimax depth ${String(game.opponent).slice(-1)}`
  }

  const showLlm = !!game && game.opponent === 'llm' && llmStats.moves > 0
  const llmPct = llmStats.moves > 0 ? Math.round((100 * llmStats.fallbacks) / llmStats.moves) : 0
  const llmLast = llmStats.last
  const modeLabel = MODES.find((m) => m.id === mode).label

  let dotColor = 'var(--muted)'
  if (connection === 'offline') dotColor = 'var(--danger)'
  else if (canMove) dotColor = 'var(--human)'
  else if (thinking) dotColor = thinking.who === 'llm' ? 'var(--human)' : 'var(--ai)'

  return (
    <motion.section
      id="play"
      aria-labelledby="play-title"
      className={`panel scroll-mt-24 p-6 ${span}`}
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
    >
      <div className="flex items-baseline justify-between gap-4">
        <h2 id="play-title" className="text-xl font-semibold">Play</h2>
        <span className="label">{modeLabel}</span>
      </div>

      <div className="mt-5">
        <Segmented label="Mode" value={mode} options={MODES} onChange={changeMode} />
      </div>

      <div className={`mt-4 grid gap-4 sm:grid-cols-2`}>
        {mode === 'human' && (
          <>
            <Segmented label="Difficulty" value={difficulty} options={LEVELS} onChange={setDifficulty} />
            <Segmented
              label="Who moves first"
              value={humanFirst}
              options={[{ id: true, label: 'You' }, { id: false, label: 'AI' }]}
              onChange={setHumanFirst}
            />
          </>
        )}
        {mode === 'minimax' && (
          <Segmented label="Minimax strength" value={minimaxDepth} options={DEPTHS} onChange={setMinimaxDepth} />
        )}
        {isMatchMode && (
          <>
            <Segmented label="AI level" value={aiLevel} options={LEVELS} onChange={setAiLevel} />
            <div className={mode === 'minimax' ? 'sm:col-span-2' : ''}><Segmented label="Who moves first" value={aiFirst} options={COLOURS} onChange={setAiFirst} /></div>
          </>
        )}
      </div>
      <p className="mt-3 text-sm" style={{ color: 'var(--muted)' }}>{note}</p>

      {isMatchMode && (
        <div className="mt-4">
          <p className="label mb-2">Speed</p>
          <input
            type="range"
            min={0}
            max={SPEEDS.length - 1}
            step={1}
            value={speedIdx}
            onChange={(e) => setSpeed(SPEEDS[Number(e.target.value)])}
            aria-label="Match speed"
            aria-valuetext={`${SPEED_LABELS[speedIdx]} speed`}
            className="w-full"
            style={{ accentColor: 'var(--human)' }}
          />
          <div className="mt-1 flex justify-between">
            {SPEED_LABELS.map((s, i) => (
              <span
                key={s}
                className="label"
                style={{ color: i === speedIdx ? 'var(--human)' : 'var(--muted)', fontWeight: i === speedIdx ? 700 : 400 }}
              >
                {s}
              </span>
            ))}
          </div>
          <p className="mt-1 text-xs" style={{ color: 'var(--muted)' }}>
            {mode === 'llm'
              ? 'Speed shortens the pauses between moves and the AI thinking pulse. The LLM answers at its own pace, so it cannot be sped up.'
              : 'Speed shortens the pause between moves and the AI thinking pulse.'}
          </p>
        </div>
      )}

      <div className="mt-5 flex flex-wrap items-center gap-4">
        <button
          type="button"
          onClick={onButton}
          disabled={!online}
          className="rounded-full px-6 py-3 font-semibold transition-transform enabled:hover:-translate-y-0.5 disabled:opacity-50"
          style={buttonStyle}
        >
          {buttonLabel}
        </button>
        <p className="flex items-center gap-2 text-sm" role="status">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: dotColor }} />
          {statusText(arena)}
        </p>
        {game && (
          <p className="label ml-auto flex items-center gap-2">
            <span className="h-3 w-3 rounded-full" style={{ background: '#ffb347' }} /> {opponentName}
            <span className="ml-2 h-3 w-3 rounded-full" style={{ background: '#3ef0d0' }} /> AI
          </p>
        )}
      </div>

      {showLlm && (
        <div className="mt-4 rounded-xl border border-line px-4 py-3 text-sm" style={{ background: 'var(--panel-2)' }}>
          <p className="num text-xs" style={{ color: 'var(--muted)' }}>
            LLM moves so far: {llmStats.moves} · random fallbacks: {llmStats.fallbacks} ({llmPct}%)
          </p>
          {llmLast && llmLast.fallback && (
            <p role="alert" className="mt-2 font-semibold" style={{ color: 'var(--danger)' }}>
              The latest LLM move was a random fallback{llmLast.reason ? ` (${llmLast.reason})` : ''}. The LLM did not choose it.
            </p>
          )}
        </div>
      )}

      {error && (
        <div
          role="alert"
          className="mt-4 flex items-start justify-between gap-3 rounded-xl border px-4 py-3 text-sm"
          style={{ borderColor: 'var(--danger)', color: 'var(--danger)' }}
        >
          <span>{error}</span>
          <button type="button" onClick={clearError} className="font-semibold underline underline-offset-4">
            Dismiss
          </button>
        </div>
      )}

      <div className="relative mt-6">
        {over && <Celebration over={over} onPlayAgain={start} />}
        <Board
          game={game}
          lastMove={lastMove}
          brain={brain}
          thinking={thinking}
          canMove={canMove}
          onMove={humanMove}
          over={over}
        />
      </div>
    </motion.section>
  )
}

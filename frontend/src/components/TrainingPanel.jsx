import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import {
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

// When deployed, set VITE_API_URL to the backend address. In development the Vite proxy handles it.
const API = import.meta.env.VITE_API_URL || ''

const SERIES = [
  { key: 'random', label: 'vs Random', color: 'var(--muted)' },
  { key: 'minimax1', label: 'vs Minimax depth 1', color: 'var(--human)' },
  { key: 'minimax2', label: 'vs Minimax depth 2', color: 'var(--ai)' },
]

// The game levels that use these checkpoints (see LEVELS in backend/games.py).
const LEVEL_OF = { 'ep_500.pt': 'Easy', 'ep_5000.pt': 'Medium', 'final.pt': 'Hard' }

const WINDOW = 2 // moving average over up to 2 points on each side
const AXIS = { fill: 'var(--muted)', fontSize: 11 }
const fmtEp = (v) => (v >= 1000 ? `${v / 1000}k` : String(v))

function prepare(history) {
  const rows = history.map((h) => ({
    episode: h.episode,
    random: h.random_win_rate * 100,
    minimax1: h.minimax1_win_rate * 100,
    minimax2: h.minimax2_win_rate * 100,
    epsilon: h.epsilon,
    loss: h.loss,
  }))
  return rows.map((row, i) => {
    const out = { ...row }
    // The window shrinks near the two ends, so the line starts and ends on the real tests.
    const w = Math.min(WINDOW, i, rows.length - 1 - i)
    for (const s of SERIES) {
      let sum = 0
      let n = 0
      for (let j = i - w; j <= i + w; j++) {
        sum += rows[j][s.key]
        n++
      }
      out[`${s.key}Avg`] = sum / n
    }
    return out
  })
}

function WinTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null
  const row = payload[0].payload
  return (
    <div className="panel px-3 py-2 text-xs" style={{ background: 'var(--panel)' }}>
      <p className="label mb-1">Test at episode {row.episode.toLocaleString()}</p>
      {SERIES.map((s) => (
        <p key={s.key} className="num" style={{ color: s.color }}>
          {s.label}: {row[s.key].toFixed(0)}%
        </p>
      ))}
    </div>
  )
}

function MiniChart({ title, data, dataKey, color, domain, fmt, maxEp, marker }) {
  return (
    <div>
      <p className="label mb-1">{title}</p>
      <ResponsiveContainer width="100%" height={110}>
        <LineChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="episode" type="number" domain={[0, maxEp]} hide />
          <YAxis domain={domain} tick={AXIS} tickLine={false} axisLine={false} width={52} tickFormatter={fmt} />
          <Line dataKey={dataKey} stroke={color} strokeWidth={2} dot={false} type="monotone" isAnimationActive={false} />
          <ReferenceLine x={marker} stroke="var(--human)" strokeWidth={1.5} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function TrainingPanel({ span }) {
  const [status, setStatus] = useState('loading')
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)
  const [pick, setPick] = useState(null)

  useEffect(() => {
    const ctrl = new AbortController()
    fetch(`${API}/api/metrics`, { signal: ctrl.signal })
      .then(async (r) => {
        if (!r.ok) {
          const body = await r.json().catch(() => null)
          throw new Error((body && body.error) || `The server answered ${r.status}`)
        }
        return r.json()
      })
      .then((json) => {
        setData(json)
        setStatus('ready')
      })
      .catch((e) => {
        if (e.name === 'AbortError') return
        setError(e.message || 'Could not load the training data.')
        setStatus('error')
      })
    return () => ctrl.abort()
  }, [attempt])

  const rows = useMemo(() => (data ? prepare(data.history) : []), [data])
  const stops = useMemo(
    () => (data ? [...data.checkpoints].sort((a, b) => a.episode - b.episode) : []),
    [data],
  )

  const retry = () => {
    setStatus('loading')
    setAttempt((a) => a + 1)
  }

  let body = null
  if (status === 'loading') {
    body = (
      <div
        className="mt-6 animate-pulse rounded-xl"
        style={{ height: 280, background: 'var(--panel-2)' }}
        role="status"
        aria-label="Loading training data"
      />
    )
  } else if (status === 'error') {
    body = (
      <div
        role="alert"
        className="mt-6 flex items-start justify-between gap-3 rounded-xl border px-4 py-3 text-sm"
        style={{ borderColor: 'var(--danger)', color: 'var(--danger)' }}
      >
        <span>{error}</span>
        <button type="button" onClick={retry} className="font-semibold underline underline-offset-4">
          Retry
        </button>
      </div>
    )
  } else if (data && rows.length > 0 && stops.length > 0) {
    const maxEp = rows[rows.length - 1].episode
    const idx = pick === null ? stops.length - 1 : pick
    const stop = stops[idx]
    const row = rows.find((r) => r.episode === stop.episode) || rows[0]
    const n = data.config.eval_games
    const finalStop = stops.find((s) => s.file === 'final.pt')
    const level = LEVEL_OF[stop.file]
    const stopName = (s) => (s.file === 'final.pt' ? 'Final' : s.episode.toLocaleString())

    body = (
      <>
        <div className="mt-5 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs">
          {SERIES.map((s) => (
            <span key={s.key} className="flex items-center gap-1.5">
              <span className="h-0.5 w-5 rounded" style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
          <span style={{ color: 'var(--muted)' }}>Dots: each real test. Line: 5-point average.</span>
        </div>

        <div
          className="mt-3"
          role="img"
          aria-label={`Line chart of the win rate against Random, Minimax depth 1 and Minimax depth 2 over ${maxEp.toLocaleString()} training episodes. The numbers for the selected checkpoint are listed below the chart.`}
        >
          <ResponsiveContainer width="100%" height={290}>
            <ComposedChart data={rows} margin={{ top: 24, right: 12, bottom: 0, left: -8 }}>
              <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="episode"
                type="number"
                domain={[0, maxEp]}
                tick={AXIS}
                tickLine={false}
                axisLine={{ stroke: 'var(--line)' }}
                tickFormatter={fmtEp}
              />
              <YAxis
                domain={[0, 100]}
                tick={AXIS}
                tickLine={false}
                axisLine={false}
                width={44}
                tickFormatter={(v) => `${v}%`}
              />
              <Tooltip content={<WinTooltip />} cursor={{ stroke: 'var(--line)' }} />
              {SERIES.map((s) => (
                <Line
                  key={`${s.key}-dots`}
                  dataKey={s.key}
                  stroke="none"
                  dot={{ r: 2.5, fill: s.color, fillOpacity: 0.45, stroke: 'none' }}
                  activeDot={{ r: 5, fill: s.color, stroke: 'var(--ink)', strokeWidth: 1.5 }}
                  isAnimationActive={false}
                />
              ))}
              {SERIES.map((s) => (
                <Line
                  key={s.key}
                  dataKey={`${s.key}Avg`}
                  stroke={s.color}
                  strokeWidth={2.5}
                  dot={false}
                  activeDot={false}
                  type="monotone"
                  animationDuration={1400}
                />
              ))}
              {finalStop && (
                <ReferenceLine
                  x={finalStop.episode}
                  stroke="var(--ink)"
                  strokeDasharray="2 4"
                  label={{ value: 'final brain', position: 'top', fill: 'var(--muted)', fontSize: 10 }}
                />
              )}
              <ReferenceLine x={stop.episode} stroke="var(--human)" strokeWidth={2} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-5 rounded-xl border border-line p-4" style={{ background: 'var(--panel-2)' }}>
          <p className="label mb-3">Replay the training: pick a checkpoint</p>
          <input
            type="range"
            min={0}
            max={stops.length - 1}
            step={1}
            value={idx}
            onChange={(e) => setPick(Number(e.target.value))}
            aria-label="Training checkpoint"
            aria-valuetext={stop.file === 'final.pt' ? 'Final brain' : `Episode ${stop.episode}`}
            className="w-full"
            style={{ accentColor: 'var(--human)' }}
          />
          <div className="mt-1 flex justify-between">
            {stops.map((s, i) => (
              <button
                key={s.file}
                type="button"
                onClick={() => setPick(i)}
                className="label px-1 py-1"
                style={{ color: i === idx ? 'var(--human)' : 'var(--muted)', fontWeight: i === idx ? 700 : 400 }}
              >
                {stopName(s)}
              </button>
            ))}
          </div>

          <p className="mt-3 text-sm">
            <span className="num font-semibold">{stop.file}</span>
            <span style={{ color: 'var(--muted)' }}>
              {' '}
              ·{' '}
              {level
                ? `used for the ${level} level in the game`
                : 'not offered as a game level'}
              {stop.file === 'final.pt' && ` · picked as the best-scoring of ${rows.length} tests`}
            </span>
          </p>

          <div className="mt-3 grid gap-3 sm:grid-cols-3">
            {SERIES.map((s) => (
              <div key={s.key} className="rounded-xl border border-line p-3" style={{ background: 'var(--panel)' }}>
                <p className="label">{s.label}</p>
                <p className="num text-2xl font-bold" style={{ color: s.color }}>
                  {Math.round(row[s.key])}%
                </p>
                <p className="num text-xs" style={{ color: 'var(--muted)' }}>
                  {Math.round((row[s.key] * n) / 100)} of {n} test games won
                </p>
              </div>
            ))}
          </div>
          <p className="num mt-3 text-xs" style={{ color: 'var(--muted)' }}>
            Exploring rate (epsilon) {row.epsilon.toFixed(2)} · training loss{' '}
            {row.loss === null || row.loss === undefined ? 'not measured yet' : row.loss.toFixed(4)}
          </p>
        </div>

        <div className="mt-5 grid gap-4 sm:grid-cols-2">
          <MiniChart
            title="Exploring rate (epsilon)"
            data={rows}
            dataKey="epsilon"
            color="var(--human)"
            domain={[0, 1]}
            fmt={(v) => v.toFixed(2)}
            maxEp={maxEp}
            marker={stop.episode}
          />
          <MiniChart
            title="Training loss (axis starts at 0)"
            data={rows}
            dataKey="loss"
            color="var(--danger)"
            domain={[0, 'auto']}
            fmt={(v) => v.toFixed(3)}
            maxEp={maxEp}
            marker={stop.episode}
          />
        </div>

        <p className="mt-5 text-xs" style={{ color: 'var(--muted)' }}>
          This is a saved run, not live training: {data.config.episodes.toLocaleString()} episodes, about 29
          minutes on a CPU. Each test point plays {n} games against each opponent, {n / 2} as first player
          and {n / 2} as second, and a single point is noisy. Both sides play almost the same way every
          time, so many games repeat: for the final brain against Minimax depth 2, 100 games per colour
          gave only 7 and 14 different games. The final brain was picked as the best-scoring of{' '}
          {rows.length} tests, which flatters it, so the leaderboard uses separate games. Against Minimax
          depth 2 the final brain won every game as first player and lost every game as second (100 games
          each), so its 50% is two opposite results added together. Training loss is not a score of how
          well the AI plays.
        </p>
      </>
    )
  }

  return (
    <motion.section
      id="training"
      aria-labelledby="training-title"
      className={`panel scroll-mt-24 self-start p-6 ${span}`}
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
    >
      <div className="flex items-baseline justify-between gap-4">
        <h2 id="training-title" className="text-xl font-semibold">Training</h2>
        <span className="label">Win rate in test matches</span>
      </div>
      <p className="mt-2 text-sm" style={{ color: 'var(--muted)' }}>
        How the AI went from losing to Minimax to holding its own, checked every 500 episodes.
      </p>
      {body}
    </motion.section>
  )
}

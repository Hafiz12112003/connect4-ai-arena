import { useEffect, useMemo, useState } from 'react'
import { motion, useReducedMotion } from 'framer-motion'

// When deployed, set VITE_API_URL to the backend address. In development the Vite proxy handles it.
const API = import.meta.env.VITE_API_URL || ''

// Below this many games a win rate is shown as a small sample and is not ranked.
const MIN_GAMES = 100

const KIND_COLOR = { ai: 'var(--ai)', human: 'var(--human)' }
const OTHER_BAR = 'color-mix(in srgb, var(--ink) 55%, transparent)'
const barColor = (kind) => KIND_COLOR[kind] || OTHER_BAR
const textColor = (kind) => KIND_COLOR[kind] || 'var(--ink)'

function fmtDate(iso) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return null
  return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

function PlayerRow({ p, rank, reduce }) {
  const none = p.games === 0
  const r = none ? 0 : (100 * p.wins) / p.games
  return (
    <li className="rounded-xl border border-line p-4" style={{ background: 'var(--panel-2)' }}>
      <div className="flex items-start justify-between gap-3">
        <p className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 font-semibold">
          {rank !== null && (
            <span className="num text-xs" style={{ color: 'var(--muted)' }}>
              #{rank}
            </span>
          )}
          <span>{p.name}</span>
          {rank === null && !none && (
            <span className="label rounded-full border border-line px-2 py-0.5" style={{ fontSize: '0.6rem' }}>
              small sample
            </span>
          )}
        </p>
        <p className="num text-2xl font-bold" style={{ color: textColor(p.kind) }}>
          {none ? '–' : `${Math.round(r)}%`}
        </p>
      </div>
      <div
        className="mt-3 h-2.5 w-full overflow-hidden rounded-full"
        style={{ background: 'color-mix(in srgb, var(--muted) 25%, transparent)' }}
        role="img"
        aria-label={none ? 'No games yet' : `Win rate ${Math.round(r)} percent`}
      >
        <motion.div
          className="h-full rounded-full"
          style={{ background: barColor(p.kind) }}
          initial={reduce ? false : { width: 0 }}
          animate={{ width: `${r}%` }}
          transition={reduce ? { duration: 0 } : { duration: 0.8, ease: 'easeOut' }}
        />
      </div>
      <p className="num mt-2 text-xs" style={{ color: 'var(--muted)' }}>
        {none
          ? 'No games yet. Finish a game in the Play panel and your result appears here.'
          : `${p.wins} won · ${p.losses} lost · ${p.draws} drawn · ${p.games} games`}
      </p>
    </li>
  )
}

function HeadToHead({ matches, kindOf }) {
  return (
    <ul className="mt-3 space-y-4">
      {matches.map((m) => {
        const total = m.games || 1
        const a = (100 * m.a_wins) / total
        const d = (100 * m.draws) / total
        const b = (100 * m.b_wins) / total
        return (
          <li key={`${m.a}|${m.b}`}>
            <div className="flex items-baseline justify-between gap-3 text-sm">
              <span className="min-w-0">
                {m.a} <span className="num font-bold">{m.a_wins}</span>
              </span>
              <span className="min-w-0 text-right">
                <span className="num font-bold">{m.b_wins}</span> {m.b}
              </span>
            </div>
            <div
              className="mt-1 flex h-2 w-full overflow-hidden rounded-full"
              style={{ background: 'color-mix(in srgb, var(--muted) 20%, transparent)' }}
              role="img"
              aria-label={`${m.a} won ${m.a_wins}, ${m.b} won ${m.b_wins}, ${m.draws} drawn`}
            >
              <div style={{ width: `${a}%`, background: barColor(kindOf[m.a]), borderRight: b > 0 ? '2px solid var(--panel-2)' : 'none' }} />
              <div style={{ width: `${d}%`, background: 'var(--line)' }} />
              <div style={{ width: `${b}%`, background: 'color-mix(in srgb, var(--muted) 45%, transparent)' }} />
            </div>
            <p className="num mt-1 text-xs" style={{ color: 'var(--muted)' }}>
              {m.games} games{m.draws > 0 ? ` · ${m.draws} drawn` : ''}
            </p>
          </li>
        )
      })}
    </ul>
  )
}

export default function LeaderboardPanel({ arena, span }) {
  const over = arena.over
  const reduce = useReducedMotion()
  const [status, setStatus] = useState('loading')
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)

  // Loads on start and again whenever a game ends, so the Human row stays current.
  useEffect(() => {
    const ctrl = new AbortController()
    fetch(`${API}/api/leaderboard`, { signal: ctrl.signal })
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
        setError(e.message || 'Could not load the leaderboard.')
        // If results are already on screen, keep showing them.
        setStatus((s) => (s === 'ready' ? 'ready' : 'error'))
      })
    return () => ctrl.abort()
  }, [attempt, over])

  const { ranked, small, kindOf } = useMemo(() => {
    const players = (data && data.players) || []
    return {
      ranked: players
        .filter((p) => p.games >= MIN_GAMES)
        .sort((x, y) => y.wins / y.games - x.wins / x.games),
      small: players.filter((p) => p.games < MIN_GAMES),
      kindOf: Object.fromEntries(players.map((p) => [p.name, p.kind])),
    }
  }, [data])

  const retry = () => {
    setStatus('loading')
    setAttempt((a) => a + 1)
  }

  let body = null
  if (status === 'loading') {
    body = (
      <div
        className="mt-6 animate-pulse rounded-xl"
        style={{ height: 320, background: 'var(--panel-2)' }}
        role="status"
        aria-label="Loading the leaderboard"
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
  } else if (data) {
    const llm = data.llm && data.llm.calls > 0 ? data.llm : null
    const date = fmtDate(data.generated_at)
    const matches = data.matches || []

    body = (
      <>
        <p className="num mt-4 text-xs" style={{ color: 'var(--muted)' }}>
          Saved results{date ? ` from ${date}` : ''}
          {data.checkpoint ? `, final brain (${data.checkpoint}${data.checkpoint_episode ? `, episode ${data.checkpoint_episode}` : ''})` : ''}
          {data.games_per_pair ? `. ${data.games_per_pair} games per pair` : ''}
          {data.llm_games_per_pair ? `, ${data.llm_games_per_pair} per pair with the LLM` : ''}.
        </p>

        {ranked.length > 0 && (
          <>
            <p className="label mt-5">Ranked: {MIN_GAMES}+ games</p>
            <ul className="mt-2 space-y-3">
              {ranked.map((p, i) => (
                <PlayerRow key={p.name} p={p} rank={i + 1} reduce={reduce} />
              ))}
            </ul>
          </>
        )}

        {small.length > 0 && (
          <>
            <p className="label mt-5">Small samples: under {MIN_GAMES} games, not ranked</p>
            <ul className="mt-2 space-y-3">
              {small.map((p) => (
                <PlayerRow key={p.name} p={p} rank={null} reduce={reduce} />
              ))}
            </ul>
          </>
        )}

        {llm && (
          <div className="mt-5 rounded-xl border border-line p-4" style={{ background: 'var(--panel-2)' }}>
            <p className="label">LLM fallback rate ({llm.model})</p>
            <p className="num mt-1 text-2xl font-bold" style={{ color: 'var(--danger)' }}>
              {Math.round((100 * llm.fallbacks) / llm.calls)}%
            </p>
            <p className="num text-xs" style={{ color: 'var(--muted)' }}>
              {llm.fallbacks} of {llm.calls} moves were random fallbacks
            </p>
            <ul className="num mt-2 text-xs" style={{ color: 'var(--muted)' }}>
              {Object.entries(llm.fallback_reasons || {}).map(([reason, count]) => (
                <li key={reason}>
                  {count} × {reason}
                </li>
              ))}
            </ul>
            <p className="mt-2 text-xs" style={{ color: 'var(--muted)' }}>
              A fallback is a random legal move played when the LLM did not give a usable answer, so the
              LLM result mixes real LLM moves with random ones.
            </p>
          </div>
        )}

        {matches.length > 0 && (
          <>
            <p className="label mt-5">Head to head</p>
            <HeadToHead matches={matches} kindOf={kindOf} />
          </>
        )}

        <p className="mt-5 text-xs" style={{ color: 'var(--muted)' }}>
          A win rate is wins divided by all games a player played, so it depends on who they faced. Read
          the head-to-head list for a fair comparison. The final brain against Minimax depth 2 is an even
          split, and I checked directly that it is two opposite results: it won all 100 games as first
          player and lost all 100 as second. Both sides play almost the same way every time, so many games
          repeat (7 and 14 different games out of 100 per colour in that match-up) and these results are
          less certain than the game counts suggest. Human results count finished games at any difficulty
          and are kept in a file on the server, which free hosting erases when it restarts or goes to sleep, so this row can reset. This page only reads the saved results. A full run with
          the LLM takes hours, so it is never re-run here.
        </p>
      </>
    )
  }

  return (
    <motion.section
      id="leaderboard"
      aria-labelledby="leaderboard-title"
      className={`panel scroll-mt-24 self-start p-6 ${span}`}
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
    >
      <div className="flex items-baseline justify-between gap-4">
        <h2 id="leaderboard-title" className="text-xl font-semibold">Leaderboard</h2>
        <span className="label">Saved tournament</span>
      </div>
      <p className="mt-2 text-sm" style={{ color: 'var(--muted)' }}>
        Trained AI, Minimax, the LLM, Random and you, with every number shown.
      </p>
      {body}
    </motion.section>
  )
}

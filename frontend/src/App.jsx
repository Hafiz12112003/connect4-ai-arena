import { useState } from 'react'
import { motion } from 'framer-motion'
import { useArena } from './hooks/useArena.js'
import PlayPanel from './components/PlayPanel.jsx'
import BrainPanel from './components/BrainPanel.jsx'
import TrainingPanel from './components/TrainingPanel.jsx'
import LeaderboardPanel from './components/LeaderboardPanel.jsx'
import ErrorBoundary from './components/ErrorBoundary.jsx'

const SECTIONS = [
  { id: 'play', title: 'Play', span: 'lg:col-span-7' },
  { id: 'brain', title: 'AI Brain', span: 'lg:col-span-5' },
  { id: 'training', title: 'Training', span: 'lg:col-span-7' },
  { id: 'leaderboard', title: 'Leaderboard', span: 'lg:col-span-5' },
]

function useTheme() {
  const [theme, setTheme] = useState(
    () => document.documentElement.getAttribute('data-theme') || 'dark',
  )
  const toggle = () => {
    const next = theme === 'dark' ? 'light' : 'dark'
    document.documentElement.setAttribute('data-theme', next)
    try {
      localStorage.setItem('c4-theme', next)
    } catch {
      /* storage can be blocked; the theme still changes for this visit */
    }
    setTheme(next)
  }
  return [theme, toggle]
}

function Logo() {
  return (
    <span className="flex items-center" aria-hidden="true">
      <span className="h-5 w-5 rounded-full" style={{ background: 'var(--human)' }} />
      <span className="-ml-1.5 h-5 w-5 rounded-full" style={{ background: 'var(--ai)' }} />
    </span>
  )
}

function StatusPill({ status }) {
  const map = {
    connecting: { text: 'Connecting', color: 'var(--muted)' },
    online: { text: 'Server online', color: 'var(--ai)' },
    offline: { text: 'Server offline', color: 'var(--danger)' },
  }
  const s = map[status]
  return (
    <span
      className="label hidden items-center gap-2 rounded-full border border-line px-3 py-1.5 sm:flex"
      role="status"
    >
      <span className="h-2 w-2 rounded-full" style={{ background: s.color }} />
      {s.text}
    </span>
  )
}

function ThemeToggle({ theme, onToggle }) {
  const isLight = theme === 'light'
  return (
    <button
      onClick={onToggle}
      aria-label={isLight ? 'Switch to dark mode' : 'Switch to light mode'}
      className="label flex h-10 min-w-10 items-center justify-center rounded-full border border-line px-3 text-ink transition-colors hover:bg-panel2"
    >
      {isLight ? 'Dark' : 'Light'}
    </button>
  )
}

function Header({ theme, onToggle, status }) {
  return (
    <header className="sticky top-0 z-20 border-b border-line backdrop-blur-md" style={{ background: 'color-mix(in srgb, var(--page) 80%, transparent)' }}>
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-4 px-4 sm:px-6">
        <a href="#top" className="flex items-center gap-3 font-semibold tracking-tight">
          <Logo />
          <span>Connect Four AI Arena</span>
        </a>
        <nav className="hidden items-center gap-6 md:flex" aria-label="Sections">
          {SECTIONS.map((s) => (
            <a key={s.id} href={`#${s.id}`} className="label transition-colors hover:text-ink">
              {s.title}
            </a>
          ))}
        </nav>
        <div className="flex items-center gap-3">
          <StatusPill status={status} />
          <ThemeToggle theme={theme} onToggle={onToggle} />
        </div>
      </div>
    </header>
  )
}

function Hero() {
  const chips = ['84 → 128 → 128 → 7', '28,295 parameters', 'Hand-written PyTorch DQN', 'No RL libraries']
  return (
    <section id="top" className="mx-auto max-w-7xl px-4 pb-10 pt-14 sm:px-6 sm:pt-20">
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
      >
        <p className="label mb-4">Portfolio project</p>
        <h1 className="max-w-3xl text-4xl font-bold leading-tight tracking-tight sm:text-6xl">
          A Connect Four AI that{' '}
          <span style={{ color: 'var(--ai)' }}>taught itself</span> to play.
        </h1>
        <p className="mt-5 max-w-2xl text-lg" style={{ color: 'var(--muted)' }}>
          Watch its brain think on every move, see how it learned, then try to beat it.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <a
            href="#play"
            className="rounded-full px-6 py-3 font-semibold transition-transform hover:-translate-y-0.5"
            style={{ background: 'var(--human)', color: '#0a0d16' }}
          >
            Play now
          </a>
          <a
            href="#training"
            className="rounded-full border border-line px-6 py-3 font-semibold transition-colors hover:bg-panel2"
          >
            See how it learned
          </a>
        </div>
        <ul className="mt-10 flex flex-wrap gap-2">
          {chips.map((c) => (
            <li key={c} className="num rounded-full border border-line bg-panel px-3 py-1.5 text-xs">
              {c}
            </li>
          ))}
        </ul>
      </motion.div>
    </section>
  )
}

function Footer() {
  return (
    <footer className="mx-auto max-w-7xl px-4 pb-12 pt-4 sm:px-6">
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line pt-6 text-sm" style={{ color: 'var(--muted)' }}>
        <span>Every learning step is hand-written PyTorch. No Stable-Baselines3.</span>
        <a href="https://hafiz-portfolio-v2.vercel.app/" target="_blank" rel="noopener noreferrer" className="underline underline-offset-4 hover:text-ink">
          Built by Hafiz
        </a>
      </div>
    </footer>
  )
}

function renderSection(s, arena) {
  if (s.id === 'play') {
    return (
      <ErrorBoundary key={s.id} id={s.id} name="Play" span={s.span}>
        <PlayPanel arena={arena} span={s.span} />
      </ErrorBoundary>
    )
  }
  if (s.id === 'brain') {
    // The outer box stretches to the height of the Play row. The panel inside it sticks
    // while you scroll, but only within that box, so it never reaches the Leaderboard.
    return (
      <div key={s.id} className={s.span}>
        <div className="lg:sticky lg:top-24">
          <ErrorBoundary id={s.id} name="AI Brain" span="">
            <BrainPanel arena={arena} span="" />
          </ErrorBoundary>
        </div>
      </div>
    )
  }
  if (s.id === 'training') {
    return (
      <ErrorBoundary key={s.id} id={s.id} name="Training" span={s.span}>
        <TrainingPanel span={s.span} />
      </ErrorBoundary>
    )
  }
  return (
    <ErrorBoundary key={s.id} id={s.id} name="Leaderboard" span={s.span}>
      <LeaderboardPanel arena={arena} span={s.span} />
    </ErrorBoundary>
  )
}

export default function App() {
  const [theme, toggleTheme] = useTheme()
  const arena = useArena()
  return (
    <>
      <Header theme={theme} onToggle={toggleTheme} status={arena.connection} />
      <main>
        <Hero />
        <div className="mx-auto grid max-w-7xl grid-cols-1 gap-6 px-4 pb-16 sm:px-6 lg:grid-cols-12">
          {SECTIONS.map((s) => renderSection(s, arena))}
        </div>
      </main>
      <Footer />
    </>
  )
}

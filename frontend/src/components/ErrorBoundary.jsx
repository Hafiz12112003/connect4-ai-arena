import { Component } from 'react'

// Catches a crash inside one panel so the rest of the page keeps working.
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.error(`[${this.props.name || 'panel'}] crashed:`, error, info.componentStack)
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children
    const { id, name = 'This panel', span = '' } = this.props
    return (
      <section id={id} className={`panel scroll-mt-24 p-6 ${span}`} role="alert">
        <h2 className="text-xl font-semibold" style={{ color: 'var(--danger)' }}>
          {name} stopped
        </h2>
        <p className="mt-2 text-sm" style={{ color: 'var(--muted)' }}>
          Something went wrong while drawing this panel. The rest of the page still works.
        </p>
        <p className="num mt-3 rounded-lg border border-line px-3 py-2 text-xs" style={{ background: 'var(--panel-2)' }}>
          {String(error && error.message ? error.message : error)}
        </p>
        <button
          type="button"
          onClick={() => this.setState({ error: null })}
          className="mt-4 rounded-full border border-line px-5 py-2 text-sm font-semibold transition-colors hover:bg-panel2"
        >
          Try again
        </button>
      </section>
    )
  }
}

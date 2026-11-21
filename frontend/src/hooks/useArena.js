import { useCallback, useEffect, useRef, useState } from 'react'
import { io } from 'socket.io-client'

// In development the Vite proxy forwards to Flask, so no address is needed.
// When deployed, set VITE_API_URL to the backend address.
const SERVER = import.meta.env.VITE_API_URL || undefined

const NO_LLM = { moves: 0, fallbacks: 0, last: null }

export function useArena() {
  const socketRef = useRef(null)
  const [connection, setConnection] = useState('connecting')
  const [game, setGame] = useState(null)
  const [brain, setBrain] = useState(null)
  const [lastMove, setLastMove] = useState(null)
  const [thinking, setThinking] = useState(null)
  const [over, setOver] = useState(null)
  const [error, setError] = useState(null)
  const [pending, setPending] = useState(false)
  const [speed, setSpeedState] = useState(1)
  const [stopped, setStopped] = useState(false)
  const [llmStats, setLlmStats] = useState(NO_LLM)

  useEffect(() => {
    const options = { transports: ['polling', 'websocket'] }
    const socket = SERVER ? io(SERVER, options) : io(options)
    socketRef.current = socket

    socket.on('connect', () => setConnection('online'))

    socket.on('connect_error', () => setConnection('offline'))

    // The server forgets a game or match when the connection drops, so we do too.
    socket.on('disconnect', () => {
      setConnection('offline')
      setGame(null)
      setBrain(null)
      setLastMove(null)
      setThinking(null)
      setOver(null)
      setPending(false)
      setStopped(false)
      setLlmStats(NO_LLM)
      setError('Connection lost. Start a new game when the server is back.')
    })

    // A fresh game or match (move_count 0), or the board after a human move.
    socket.on('game_state', (s) => {
      setPending(false)
      setError(null)
      setThinking(null)
      if (s.move_count === 0) {
        setBrain(null)
        setLastMove(null)
        setOver(null)
        setStopped(false)
        setLlmStats(NO_LLM)
        setGame(s)
        return
      }
      setGame((prev) => ({ ...prev, ...s }))
      const [row, column] = s.last_move
      setLastMove({
        id: s.move_count,
        who: 'human',
        player: s.board[row][column],
        row,
        column,
        fallback: false,
      })
    })

    // who is "ai" or "llm". The LLM can take many seconds to answer.
    socket.on('ai_thinking', (data) => setThinking(data))

    // who is "ai", "minimax" or "llm".
    socket.on('ai_move', (ev) => {
      setThinking(null)
      setGame((prev) => ({ ...prev, ...ev.state }))
      const fallback = !!(ev.llm && ev.llm.fallback)
      setLastMove({
        id: ev.state.move_count,
        who: ev.who,
        player: ev.player,
        row: ev.row,
        column: ev.column,
        fallback,
      })
      if (ev.who === 'llm') {
        setLlmStats((s) => ({
          moves: s.moves + 1,
          fallbacks: s.fallbacks + (fallback ? 1 : 0),
          last: ev.llm
            ? { fallback, reason: ev.llm.reason || '', moveId: ev.state.move_count }
            : null,
        }))
      }
      if (ev.brain) {
        setBrain({
          ...ev.brain,
          column: ev.column,
          player: ev.player,
          moveCount: ev.state.move_count,
        })
      }
    })

    socket.on('game_over', (p) => {
      setThinking(null)
      setGame((prev) => ({ ...prev, ...p.state }))
      setOver({
        winner: p.winner,
        draw: p.draw,
        winning_cells: p.winning_cells,
        winner_who: p.winner_who,
      })
    })

    socket.on('match_stopped', () => {
      setThinking(null)
      setStopped(true)
    })

    socket.on('error', (e) => {
      setPending(false)
      setError(e && e.message ? e.message : 'Something went wrong.')
    })

    return () => {
      socket.removeAllListeners()
      socket.disconnect()
    }
  }, [])

  const canMove =
    connection === 'online' &&
    !!game &&
    game.mode === 'human_ai' &&
    !game.done &&
    !over &&
    game.current_player === game.human &&
    !thinking &&
    !pending

  // True while a spectator match is running (not finished, not stopped).
  const watching =
    connection === 'online' && !!game && game.mode !== 'human_ai' && !over && !stopped

  const resetView = useCallback(() => {
    setError(null)
    setOver(null)
    setBrain(null)
    setLastMove(null)
    setThinking(null)
    setPending(false)
    setStopped(false)
    setLlmStats(NO_LLM)
  }, [])

  const newGame = useCallback(
    ({ difficulty = 'medium', humanFirst = true } = {}) => {
      resetView()
      socketRef.current?.emit('new_game', { difficulty, human_first: humanFirst })
    },
    [resetView],
  )

  // aiFirst: true or false picks the colour. Leave it null and the server alternates every match.
  const startMatch = useCallback(
    ({ opponent = 'minimax2', difficulty = 'hard', aiFirst = null, speed: sp = 1 } = {}) => {
      resetView()
      setSpeedState(sp)
      const payload = { opponent, difficulty, speed: sp }
      if (aiFirst === true || aiFirst === false) payload.ai_first = aiFirst
      socketRef.current?.emit('start_match', payload)
    },
    [resetView],
  )

  const setSpeed = useCallback((value) => {
    setSpeedState(value)
    socketRef.current?.emit('set_speed', { speed: value })
  }, [])

  const stopMatch = useCallback(() => {
    socketRef.current?.emit('stop_match')
  }, [])

  const humanMove = useCallback(
    (col) => {
      if (!canMove) return
      setPending(true)
      setError(null)
      socketRef.current?.emit('human_move', { col })
    },
    [canMove],
  )

  const clearError = useCallback(() => setError(null), [])

  return {
    connection,
    game,
    brain,
    lastMove,
    thinking,
    over,
    error,
    canMove,
    watching,
    stopped,
    speed,
    llmStats,
    newGame,
    startMatch,
    stopMatch,
    setSpeed,
    humanMove,
    clearError,
  }
}

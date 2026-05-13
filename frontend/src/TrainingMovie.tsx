/**
 * C-15 — Real-time training visualization.
 *
 * Animates the 24 demo galaxies' positions in the Zoobot
 * penultimate-feature UMAP space across all training epochs of the
 * T3.6 run. All epochs are projected through the SAME UMAP fit (the
 * final epoch's) so what the user sees is genuine training-driven
 * motion, not UMAP re-fit jitter.
 *
 * Controls:
 *  - epoch slider (or scrub)
 *  - Play / Pause toggle (auto-advance at ~5 fps)
 *
 * Backend:
 *   GET /api/training_movie -> { epochs, label_names, frames }
 */

import { useEffect, useMemo, useRef, useState } from 'react'
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore -- factory lacks first-class TS types
import * as factoryNs from 'react-plotly.js/factory'
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import Plotly from 'plotly.js-dist-min'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const factory: any =
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (factoryNs as any).default?.default ??
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (factoryNs as any).default ??
  factoryNs
const Plot = factory(Plotly)

interface TrainingFrame {
  epoch: number
  galaxy_id: string
  umap_x: number
  umap_y: number
  label_name: string
}

interface TrainingResponse {
  epochs: number[]
  label_names: string[]
  frames: TrainingFrame[]
}

const CLASS_COLORS: Record<string, string> = {
  smooth: '#0072B2',
  'featured-or-disk': '#E69F00',
  artifact: '#CC79A7',
  unknown: '#94a3b8',
}

const PLAY_INTERVAL_MS = 200

export function TrainingMovie() {
  const [data, setData] = useState<TrainingResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [epochIdx, setEpochIdx] = useState(0)
  const [playing, setPlaying] = useState(false)
  const playRef = useRef<number | null>(null)

  useEffect(() => {
    let cancelled = false
    fetch('/api/training_movie')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}: ${r.statusText}`)
        return r.json() as Promise<TrainingResponse>
      })
      .then((body) => {
        if (cancelled) return
        setData(body)
        setEpochIdx(0)
      })
      .catch((e) => {
        if (cancelled) return
        setError(e instanceof Error ? e.message : String(e))
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Auto-advance loop when playing == true.
  useEffect(() => {
    if (!playing || !data) return
    playRef.current = window.setInterval(() => {
      setEpochIdx((i) => (i + 1) % data.epochs.length)
    }, PLAY_INTERVAL_MS)
    return () => {
      if (playRef.current !== null) window.clearInterval(playRef.current)
      playRef.current = null
    }
  }, [playing, data])

  const framesByEpoch = useMemo(() => {
    if (!data) return new Map<number, TrainingFrame[]>()
    const map = new Map<number, TrainingFrame[]>()
    for (const f of data.frames) {
      const list = map.get(f.epoch) ?? []
      list.push(f)
      map.set(f.epoch, list)
    }
    return map
  }, [data])

  const currentFrames = useMemo(() => {
    if (!data || data.epochs.length === 0) return []
    const epoch = data.epochs[epochIdx] ?? data.epochs[0]
    return framesByEpoch.get(epoch) ?? []
  }, [data, framesByEpoch, epochIdx])

  // Bounding box across ALL frames so the axes stay stable while the
  // slider moves.
  const bounds = useMemo(() => {
    if (!data || data.frames.length === 0) return null
    let xMin = Infinity
    let xMax = -Infinity
    let yMin = Infinity
    let yMax = -Infinity
    for (const f of data.frames) {
      if (f.umap_x < xMin) xMin = f.umap_x
      if (f.umap_x > xMax) xMax = f.umap_x
      if (f.umap_y < yMin) yMin = f.umap_y
      if (f.umap_y > yMax) yMax = f.umap_y
    }
    const padX = (xMax - xMin) * 0.05
    const padY = (yMax - yMin) * 0.05
    return {
      x: [xMin - padX, xMax + padX],
      y: [yMin - padY, yMax + padY],
    }
  }, [data])

  const traces = useMemo(() => {
    if (!data) return []
    const byLabel = new Map<string, TrainingFrame[]>()
    for (const f of currentFrames) {
      const list = byLabel.get(f.label_name) ?? []
      list.push(f)
      byLabel.set(f.label_name, list)
    }
    return Array.from(byLabel.entries()).map(([label, items]) => ({
      type: 'scatter' as const,
      mode: 'markers' as const,
      x: items.map((f) => f.umap_x),
      y: items.map((f) => f.umap_y),
      text: items.map((f) => `#${f.galaxy_id}`),
      name: `${label} (n=${items.length})`,
      marker: {
        color: CLASS_COLORS[label] ?? '#94a3b8',
        size: 14,
        opacity: 0.85,
        line: { color: '#0f172a', width: 1 },
      },
      hovertemplate: '<b>%{text}</b><br>UMAP-1=%{x:.2f}<br>UMAP-2=%{y:.2f}<extra></extra>',
    }))
  }, [data, currentFrames])

  if (error) {
    return (
      <div className="text-sm text-red-400 whitespace-pre-wrap">
        Could not load training movie: {error}
      </div>
    )
  }
  if (!data) {
    return <div className="text-sm text-slate-400">Loading training movie…</div>
  }

  const epoch = data.epochs[epochIdx]
  const epochLabel = epoch === -1 ? 'pretrained init' : `epoch ${epoch + 1}`

  return (
    <div className="space-y-6">
      <section>
        <h2 className="text-lg font-medium text-slate-100 mb-1">
          Training movie
        </h2>
        <p className="text-sm text-slate-400 mb-4">
          {data.frames.length === 0
            ? 'No frames available yet — re-run the T3.6 trainer with logging.per_epoch_features_path set.'
            : `${data.epochs.length} snapshots × ${
                currentFrames.length
              } demo galaxies. All epochs projected through the same final-epoch UMAP fit so the motion you see is genuine training progress.`}
        </p>
      </section>

      <section className="flex items-center gap-3 flex-wrap">
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          className="text-xs rounded-md bg-violet-600 hover:bg-violet-500 text-white px-3 py-1"
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(0, data.epochs.length - 1)}
          value={epochIdx}
          onChange={(e) => {
            setPlaying(false)
            setEpochIdx(parseInt(e.target.value, 10))
          }}
          className="flex-1 min-w-[16rem]"
        />
        <span className="text-xs text-slate-300 tabular-nums w-32 text-right">
          {epochLabel}
        </span>
      </section>

      <section>
        <Plot
          data={traces}
          layout={{
            width: 880,
            height: 520,
            margin: { l: 50, r: 20, t: 20, b: 40 },
            xaxis: {
              title: { text: 'UMAP-1' },
              zeroline: false,
              range: bounds?.x,
            },
            yaxis: {
              title: { text: 'UMAP-2' },
              zeroline: false,
              range: bounds?.y,
            },
            paper_bgcolor: '#0f172a',
            plot_bgcolor: '#0f172a',
            font: { color: '#cbd5e1', size: 11 },
            legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(15,23,42,0.6)' },
            transition: { duration: 250, easing: 'cubic-in-out' },
          }}
          config={{
            displaylogo: false,
            modeBarButtonsToRemove: ['toImage'],
            responsive: true,
          }}
          style={{ borderRadius: 8 }}
        />
      </section>
    </div>
  )
}

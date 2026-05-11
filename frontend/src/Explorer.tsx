/**
 * T4.2 — Interactive UMAP Explorer tab.
 *
 * 2,462 DR8 test galaxies in their Zoobot penultimate-feature UMAP space
 * (from T2.4). Features per DEVPLAN T4.2:
 *
 *  - **scattergl plot** colored by the smooth-or-featured 3-class label
 *  - **hover thumbnail**: float a small image preview at the cursor on hover
 *  - **lasso**: rectangular / freehand selection
 *  - **sample grid**: lassoed points render as a grid of thumbnails below
 *  - **color-by selector**: switch palette between the canonical class or
 *    "by UMAP-y" (continuous, for visual sanity)
 *  - **click-to-posterior** (substitutes DEVPLAN's click-to-Aladin --
 *    HF dataset doesn't ship per-galaxy RA/Dec): clicking a point
 *    fetches the full PosteriorResponse + renders bars in a side panel
 *
 * Data layer:
 *   GET  /api/umap_points                          -> {points, label_names}
 *   GET  /api/test_thumbs/{idx}/thumbnail          -> JPEG (lazy on hover)
 *   POST /api/test_thumbs/{idx}/posteriors         -> PosteriorResponse
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import Plot from 'react-plotly.js'

interface UMAPPoint {
  idx: number
  x: number
  y: number
  label: number
  label_name: string
}

interface UMAPPointsResponse {
  points: UMAPPoint[]
  label_names: string[]
}

interface PosteriorAnswer {
  name: string
  mean: number
  ci_lower: number
  ci_upper: number
}

interface PosteriorQuestion {
  question: string
  answers: PosteriorAnswer[]
  plurality_answer: string
  plurality_index: number
  n_effective: number
  active: boolean
  parent_question: string | null
  parent_answer: string | null
}

interface PosteriorResponse {
  questions: PosteriorQuestion[]
  calibration_regime: string
  temperature: number
}

type ColorBy = 'class' | 'umap_y'

const CLASS_COLORS: Record<string, string> = {
  smooth: '#0072B2',
  'featured-or-disk': '#E69F00',
  artifact: '#CC79A7',
}

const SAMPLE_GRID_MAX = 60 // cap the thumb grid to a sane number

export function Explorer() {
  const [points, setPoints] = useState<UMAPPoint[]>([])
  const [labelNames, setLabelNames] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
  const [selectedIdxs, setSelectedIdxs] = useState<number[] | null>(null)
  const [hoverIdx, setHoverIdx] = useState<number | null>(null)
  const [hoverXY, setHoverXY] = useState<{ x: number; y: number } | null>(null)
  const [clickedIdx, setClickedIdx] = useState<number | null>(null)
  const [posterior, setPosterior] = useState<PosteriorResponse | null>(null)
  const [colorBy, setColorBy] = useState<ColorBy>('class')
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let cancelled = false
    fetch('/api/umap_points')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}: ${r.statusText}`)
        return r.json() as Promise<UMAPPointsResponse>
      })
      .then((body) => {
        if (cancelled) return
        setPoints(body.points)
        setLabelNames(body.label_names)
      })
      .catch((e) => {
        if (cancelled) return
        setError(
          `Could not load UMAP points (${e instanceof Error ? e.message : String(e)})`,
        )
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Lazy-fetch the clicked galaxy's posterior whenever clickedIdx changes.
  useEffect(() => {
    if (clickedIdx === null) {
      setPosterior(null)
      return
    }
    let cancelled = false
    fetch(`/api/test_thumbs/${clickedIdx}/posteriors`, { method: 'POST' })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}: ${r.statusText}`)
        return r.json() as Promise<PosteriorResponse>
      })
      .then((body) => {
        if (!cancelled) setPosterior(body)
      })
      .catch((e) => {
        if (cancelled) return
        setError(
          `Could not load posterior for ${clickedIdx} (${e instanceof Error ? e.message : String(e)})`,
        )
      })
    return () => {
      cancelled = true
    }
  }, [clickedIdx])

  const traces = useMemo(() => {
    if (points.length === 0) return []
    if (colorBy === 'class') {
      return labelNames.map((name) => {
        const cls_pts = points.filter((p) => p.label_name === name)
        return {
          type: 'scattergl' as const,
          mode: 'markers' as const,
          x: cls_pts.map((p) => p.x),
          y: cls_pts.map((p) => p.y),
          customdata: cls_pts.map((p) => p.idx),
          name: `${name} (n=${cls_pts.length})`,
          marker: {
            color: CLASS_COLORS[name] ?? '#999999',
            size: 4,
            opacity: 0.65,
            line: { width: 0 },
          },
          hoverinfo: 'skip' as const, // We render our own hover thumbnail.
        }
      })
    }
    // colorBy === 'umap_y': single trace colored continuously
    return [
      {
        type: 'scattergl' as const,
        mode: 'markers' as const,
        x: points.map((p) => p.x),
        y: points.map((p) => p.y),
        customdata: points.map((p) => p.idx),
        name: 'all',
        marker: {
          color: points.map((p) => p.y),
          colorscale: 'Viridis' as const,
          size: 4,
          opacity: 0.65,
          showscale: true,
          colorbar: { title: { text: 'UMAP-2' }, thickness: 10 },
          line: { width: 0 },
        },
        hoverinfo: 'skip' as const,
      },
    ]
  }, [points, labelNames, colorBy])

  function onPlotHover(ev: { points: Array<{ customdata?: unknown; bbox?: { x0: number; y0: number } }> }) {
    const p = ev?.points?.[0]
    if (!p) return
    const idx = typeof p.customdata === 'number' ? p.customdata : null
    if (idx === null) return
    setHoverIdx(idx)
    if (p.bbox) setHoverXY({ x: p.bbox.x0, y: p.bbox.y0 })
  }

  function onPlotUnhover() {
    setHoverIdx(null)
    setHoverXY(null)
  }

  function onPlotClick(ev: { points: Array<{ customdata?: unknown }> }) {
    const p = ev?.points?.[0]
    if (!p) return
    const idx = typeof p.customdata === 'number' ? p.customdata : null
    if (idx === null) return
    setClickedIdx(idx)
  }

  function onPlotSelected(ev: { points?: Array<{ customdata?: unknown }> } | undefined) {
    if (!ev || !ev.points) {
      setSelectedIdxs(null)
      return
    }
    const idxs = ev.points
      .map((p) => p.customdata)
      .filter((c): c is number => typeof c === 'number')
    setSelectedIdxs(idxs)
  }

  const selectedSample =
    selectedIdxs && selectedIdxs.length > 0
      ? selectedIdxs.slice(0, SAMPLE_GRID_MAX)
      : []

  return (
    <div className="space-y-6">
      <section>
        <h2 className="text-lg font-medium text-slate-100 mb-1">
          UMAP feature-space explorer
        </h2>
        <p className="text-sm text-slate-400 mb-4">
          {points.length > 0
            ? `${points.length} DR8 test-set galaxies in the Zoobot penultimate-feature UMAP. Hover for thumbnail, click to fetch posterior, lasso to build a sample grid.`
            : error ?? 'Loading…'}
        </p>
      </section>

      {error && (
        <div className="text-sm text-red-400 whitespace-pre-wrap">{error}</div>
      )}

      {points.length > 0 && (
        <>
          <section className="flex items-center gap-3">
            <span className="text-xs text-slate-400">Color by:</span>
            <select
              value={colorBy}
              onChange={(e) => setColorBy(e.target.value as ColorBy)}
              className="text-xs bg-slate-800 border border-slate-700 text-slate-100 px-2 py-1 rounded"
            >
              <option value="class">smooth-or-featured class</option>
              <option value="umap_y">UMAP-2 (continuous)</option>
            </select>
            <span className="text-xs text-slate-500">
              {selectedIdxs && selectedIdxs.length > 0
                ? `${selectedIdxs.length} points lassoed`
                : 'lasso a region for the sample grid'}
            </span>
          </section>

          <section ref={containerRef} className="relative">
            <Plot
              data={traces}
              layout={{
                width: 880,
                height: 540,
                margin: { l: 50, r: 20, t: 20, b: 40 },
                xaxis: { title: { text: 'UMAP-1' }, zeroline: false },
                yaxis: { title: { text: 'UMAP-2' }, zeroline: false },
                paper_bgcolor: '#0f172a',
                plot_bgcolor: '#0f172a',
                font: { color: '#cbd5e1', size: 11 },
                legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(15,23,42,0.6)' },
                dragmode: 'lasso',
                hovermode: 'closest',
              }}
              config={{
                displaylogo: false,
                modeBarButtonsToRemove: ['toImage'],
                responsive: true,
              }}
              onHover={onPlotHover}
              onUnhover={onPlotUnhover}
              onClick={onPlotClick}
              onSelected={onPlotSelected}
              onDeselect={() => setSelectedIdxs(null)}
              style={{ borderRadius: 8 }}
            />

            {/* Hover thumbnail floating near the cursor */}
            {hoverIdx !== null && hoverXY && (
              <div
                className="pointer-events-none absolute rounded shadow-lg border border-slate-700 bg-slate-900 p-1"
                style={{
                  left: Math.min(hoverXY.x + 12, 760),
                  top: Math.min(hoverXY.y + 12, 420),
                  zIndex: 10,
                }}
              >
                <img
                  src={`/api/test_thumbs/${hoverIdx}/thumbnail`}
                  alt={`galaxy-${hoverIdx}`}
                  className="w-24 h-24 object-cover rounded-sm"
                />
                <div className="text-[10px] text-slate-400 text-center pt-1">
                  #{hoverIdx}
                </div>
              </div>
            )}
          </section>

          {selectedSample.length > 0 && (
            <section className="border-t border-slate-800 pt-4">
              <h3 className="text-sm font-medium text-slate-300 mb-2">
                Sample grid ({selectedSample.length}
                {selectedIdxs && selectedIdxs.length > SAMPLE_GRID_MAX
                  ? ` of ${selectedIdxs.length}`
                  : ''})
              </h3>
              <div className="grid grid-cols-6 sm:grid-cols-10 gap-1">
                {selectedSample.map((idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => setClickedIdx(idx)}
                    className={`rounded border ${
                      clickedIdx === idx
                        ? 'border-indigo-400'
                        : 'border-slate-800 hover:border-slate-600'
                    }`}
                    title={`galaxy ${idx}`}
                  >
                    <img
                      src={`/api/test_thumbs/${idx}/thumbnail`}
                      alt={`g-${idx}`}
                      className="w-full aspect-square object-cover rounded-[2px]"
                    />
                  </button>
                ))}
              </div>
            </section>
          )}

          {clickedIdx !== null && (
            <section className="border-t border-slate-800 pt-4">
              <h3 className="text-sm font-medium text-slate-300 mb-2">
                Galaxy #{clickedIdx} — posterior
              </h3>
              {!posterior && <div className="text-sm text-slate-400">Predicting…</div>}
              {posterior && (
                <div className="grid grid-cols-2 gap-2">
                  {posterior.questions.map((q) => (
                    <div
                      key={q.question}
                      className={`rounded border border-slate-800 px-3 py-2 ${
                        q.active ? 'bg-slate-900/40' : 'bg-slate-900/20 opacity-50'
                      }`}
                    >
                      <div className="flex justify-between items-baseline mb-1">
                        <span className="text-xs font-medium text-slate-100">
                          {q.question}
                        </span>
                        <span className="text-[10px] text-slate-500">
                          plurality: {q.plurality_answer}
                        </span>
                      </div>
                      <div className="text-[10px] text-slate-400">
                        {q.answers
                          .map(
                            (a) =>
                              `${a.name}=${(a.mean * 100).toFixed(0)}%`,
                          )
                          .join(' · ')}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </section>
          )}
        </>
      )}
    </div>
  )
}

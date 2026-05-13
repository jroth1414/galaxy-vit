/**
 * S-2 — Sky map + Aladin Lite embed.
 *
 * Two stacked sub-views in a single tab:
 *
 * 1. **RA/Dec scatter** of the ~14 k DR8 galaxies that have both an
 *    inference row and volunteer-catalog coordinates. Plotly scattergl
 *    handles the point count without trouble. Color by the model's
 *    predicted smooth-or-featured class or by predictive entropy.
 *    Hover + click forward (ra, dec) to the Aladin sub-view below.
 *
 * 2. **Aladin Lite** iframe pointed at DECaLS DR10 imagery. The iframe
 *    URL accepts a `target=` query param (RA Dec); when the user
 *    clicks a scatter point we re-mount the iframe at that target so
 *    Aladin centres on the galaxy.
 *
 * The "click anywhere on the sky" → /api/predict_sdss flow is wired
 * via a small "Predict at coords" button that POSTs the current Aladin
 * target to /api/predict_sdss. Cross-origin postMessage from the
 * Aladin iframe is intentionally NOT wired -- it's brittle on
 * embedded mode (CDS doesn't expose the click handler over postMessage
 * in their iframe build) and the manual flow gives the user the same
 * outcome with a clearer affordance.
 *
 * Backend:
 *   GET /api/sky_points          -> {points, label_names}
 *   GET /api/predict_sdss?ra&dec -> PredictResponse  (existing endpoint)
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

interface SkyPoint {
  dr8_id: string
  ra: number
  dec: number
  label: number
  label_name: string
  entropy: number
}

interface SkyPointsResponse {
  points: SkyPoint[]
  label_names: string[]
}

interface TopKItem {
  class_id: number
  class_name: string
  probability: number
}

interface PredictResponse {
  top_k: TopKItem[]
  attention_id: string
}

type ColorBy = 'class' | 'entropy'

const CLASS_COLORS: Record<string, string> = {
  smooth: '#0072B2',
  'featured-or-disk': '#E69F00',
  artifact: '#CC79A7',
}

const ALADIN_BASE = 'https://aladin.cds.unistra.fr/AladinLite/'

function aladinUrl(ra: number, dec: number): string {
  // The Aladin Lite v3 iframe API accepts target / fov / survey via URL.
  // DECaLS DR10 colour: P/DECaPS/DR1/color.
  const params = new URLSearchParams({
    target: `${ra.toFixed(6)} ${dec.toFixed(6)}`,
    fov: '0.1',
    survey: 'P/DECaPS/DR1/color',
  })
  return `${ALADIN_BASE}?${params.toString()}`
}

export function Sky() {
  const [points, setPoints] = useState<SkyPoint[]>([])
  const [labelNames, setLabelNames] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
  const [colorBy, setColorBy] = useState<ColorBy>('class')
  const [target, setTarget] = useState<{ ra: number; dec: number } | null>(
    null,
  )
  const [pred, setPred] = useState<PredictResponse | null>(null)
  const [predError, setPredError] = useState<string | null>(null)
  const [predLoading, setPredLoading] = useState(false)
  const [nameInput, setNameInput] = useState<string>('')
  const [nameResolved, setNameResolved] = useState<string | null>(null)
  const [nameError, setNameError] = useState<string | null>(null)
  const [nameLoading, setNameLoading] = useState(false)
  const aladinRef = useRef<HTMLIFrameElement>(null)

  useEffect(() => {
    let cancelled = false
    fetch('/api/sky_points')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}: ${r.statusText}`)
        return r.json() as Promise<SkyPointsResponse>
      })
      .then((body) => {
        if (cancelled) return
        setPoints(body.points)
        setLabelNames(body.label_names)
      })
      .catch((e) => {
        if (cancelled) return
        setError(
          `Could not load sky points (${
            e instanceof Error ? e.message : String(e)
          })`,
        )
      })
    return () => {
      cancelled = true
    }
  }, [])

  const traces = useMemo(() => {
    if (points.length === 0) return []
    if (colorBy === 'class') {
      return labelNames.map((name) => {
        const cls_pts = points.filter((p) => p.label_name === name)
        return {
          type: 'scattergl' as const,
          mode: 'markers' as const,
          x: cls_pts.map((p) => p.ra),
          y: cls_pts.map((p) => p.dec),
          customdata: cls_pts.map((p) => [p.dr8_id, p.entropy]),
          name: `${name} (n=${cls_pts.length})`,
          marker: {
            color: CLASS_COLORS[name] ?? '#999999',
            size: 3,
            opacity: 0.55,
            line: { width: 0 },
          },
          hovertemplate:
            '<b>%{customdata[0]}</b><br>' +
            'RA = %{x:.3f}<br>' +
            'Dec = %{y:.3f}<br>' +
            'entropy = %{customdata[1]:.2f}<extra></extra>',
        }
      })
    }
    return [
      {
        type: 'scattergl' as const,
        mode: 'markers' as const,
        x: points.map((p) => p.ra),
        y: points.map((p) => p.dec),
        customdata: points.map((p) => [p.dr8_id, p.entropy]),
        name: 'all',
        marker: {
          color: points.map((p) => p.entropy),
          colorscale: 'Viridis' as const,
          size: 3,
          opacity: 0.55,
          showscale: true,
          colorbar: { title: { text: 'entropy' }, thickness: 10 },
          line: { width: 0 },
        },
        hovertemplate:
          '<b>%{customdata[0]}</b><br>' +
          'RA = %{x:.3f}<br>' +
          'Dec = %{y:.3f}<br>' +
          'entropy = %{customdata[1]:.2f}<extra></extra>',
      },
    ]
  }, [points, labelNames, colorBy])

  function onPlotClick(ev: {
    points: Array<{ x?: number; y?: number }>
  }) {
    const p = ev?.points?.[0]
    if (!p || typeof p.x !== 'number' || typeof p.y !== 'number') return
    setTarget({ ra: p.x, dec: p.y })
    setPred(null)
    setPredError(null)
  }

  async function resolveAndCenter(query: string) {
    setNameLoading(true)
    setNameError(null)
    setNameResolved(null)
    try {
      const r = await fetch(
        `/api/resolve_name?name=${encodeURIComponent(query)}`,
      )
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`)
      const body = (await r.json()) as {
        ra: number
        dec: number
        source: string
      }
      setTarget({ ra: body.ra, dec: body.dec })
      setNameResolved(
        `${body.source === 'coords' ? 'coords' : 'sesame'}: ` +
          `RA=${body.ra.toFixed(4)}° Dec=${body.dec.toFixed(4)}°`,
      )
      setPred(null)
      setPredError(null)
    } catch (e) {
      setNameError(e instanceof Error ? e.message : String(e))
    } finally {
      setNameLoading(false)
    }
  }

  function onSubmitName(e: React.FormEvent) {
    e.preventDefault()
    if (!nameInput.trim()) return
    resolveAndCenter(nameInput.trim())
  }

  async function predictAtTarget() {
    if (!target) return
    setPredLoading(true)
    setPredError(null)
    setPred(null)
    try {
      const r = await fetch(
        `/api/predict_sdss?ra=${target.ra}&dec=${target.dec}`,
      )
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`)
      const body = (await r.json()) as PredictResponse
      setPred(body)
    } catch (e) {
      setPredError(e instanceof Error ? e.message : String(e))
    } finally {
      setPredLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <section>
        <h2 className="text-lg font-medium text-slate-100 mb-1">
          Sky map
        </h2>
        <p className="text-sm text-slate-400 mb-4">
          {points.length > 0
            ? `${points.length} DR8 galaxies (joined inference + volunteer catalog) on RA / Dec. Click a point to centre Aladin Lite below; press "Predict at coords" to fetch a fresh DECaLS cutout + Galaxy10 prediction.`
            : (error ?? 'Loading…')}
        </p>
      </section>

      {error && (
        <div className="text-sm text-red-400 whitespace-pre-wrap">{error}</div>
      )}

      {/* A-8: object-name / coords resolver. Always visible (doesn't
       * require sky_points to be loaded). */}
      <section className="border-t border-slate-800 pt-4">
        <h3 className="text-sm font-medium text-slate-300 mb-2">
          Jump to a galaxy
        </h3>
        <form
          onSubmit={onSubmitName}
          className="flex items-center gap-2 flex-wrap"
        >
          <input
            type="text"
            value={nameInput}
            onChange={(e) => setNameInput(e.target.value)}
            placeholder="e.g. M31, NGC 1300, or '10.68 41.27'"
            className="text-xs bg-slate-800 border border-slate-700 text-slate-100 px-2 py-1 rounded flex-1 min-w-[16rem]"
          />
          <button
            type="submit"
            disabled={nameLoading || !nameInput.trim()}
            className="text-xs rounded-md bg-violet-600 hover:bg-violet-500 disabled:bg-slate-800 disabled:text-slate-500 text-white px-3 py-1"
          >
            {nameLoading ? 'Resolving…' : 'Resolve & centre'}
          </button>
          {nameResolved && (
            <span className="text-xs text-slate-500">{nameResolved}</span>
          )}
        </form>
        {nameError && (
          <div className="mt-2 text-xs text-red-400 whitespace-pre-wrap">
            {nameError}
          </div>
        )}
      </section>

      {points.length > 0 && (
        <>
          <section className="flex items-center gap-3 flex-wrap">
            <span className="text-xs text-slate-400">Color by:</span>
            <select
              value={colorBy}
              onChange={(e) => setColorBy(e.target.value as ColorBy)}
              className="text-xs bg-slate-800 border border-slate-700 text-slate-100 px-2 py-1 rounded"
            >
              <option value="class">smooth-or-featured class</option>
              <option value="entropy">predictive entropy</option>
            </select>
            {target && (
              <span className="text-xs text-slate-500">
                target: RA={target.ra.toFixed(3)}° Dec={target.dec.toFixed(3)}°
              </span>
            )}
          </section>

          <section>
            <Plot
              data={traces}
              layout={{
                width: 880,
                height: 480,
                margin: { l: 60, r: 20, t: 20, b: 50 },
                xaxis: {
                  title: { text: 'RA (deg)' },
                  zeroline: false,
                  range: [0, 360],
                },
                yaxis: {
                  title: { text: 'Dec (deg)' },
                  zeroline: false,
                  range: [-30, 35],
                  // GZ DESI footprint is approximately Dec [-30, 35].
                },
                paper_bgcolor: '#0f172a',
                plot_bgcolor: '#0f172a',
                font: { color: '#cbd5e1', size: 11 },
                legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(15,23,42,0.6)' },
                hovermode: 'closest',
              }}
              config={{
                displaylogo: false,
                modeBarButtonsToRemove: ['toImage'],
                responsive: true,
              }}
              onClick={onPlotClick}
              style={{ borderRadius: 8 }}
            />
          </section>

          <section className="border-t border-slate-800 pt-4">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-slate-300">
                Aladin Lite — DECaLS DR10
              </h3>
              {target && (
                <button
                  type="button"
                  onClick={predictAtTarget}
                  disabled={predLoading}
                  className="text-xs rounded-md bg-violet-600 hover:bg-violet-500 disabled:bg-slate-800 disabled:text-slate-500 text-white px-3 py-1"
                >
                  {predLoading ? 'Predicting…' : 'Predict at coords'}
                </button>
              )}
            </div>
            {!target ? (
              <p className="text-xs text-slate-500 italic">
                Click a point above to centre the Aladin view here.
              </p>
            ) : (
              <iframe
                ref={aladinRef}
                title="Aladin Lite"
                src={aladinUrl(target.ra, target.dec)}
                className="w-full h-[480px] rounded border border-slate-800"
                /* CDS attribution stays visible inside the iframe -
                 * required by the Aladin Lite license; do not strip. */
              />
            )}
          </section>

          {(pred || predError) && (
            <section className="border-t border-slate-800 pt-4">
              <h3 className="text-sm font-medium text-slate-300 mb-2">
                Galaxy10 prediction at{' '}
                {target
                  ? `(${target.ra.toFixed(3)}°, ${target.dec.toFixed(3)}°)`
                  : ''}
              </h3>
              {predError && (
                <div className="text-sm text-red-400 whitespace-pre-wrap">
                  {predError}
                </div>
              )}
              {pred && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <img
                    src={`/api/attention/${pred.attention_id}`}
                    alt="GradCAM"
                    className="w-full aspect-square object-cover rounded-md border border-slate-800"
                  />
                  <div className="space-y-2">
                    {pred.top_k.map((it, i) => (
                      <div
                        key={it.class_id}
                        className="flex items-center gap-3"
                      >
                        <span className="text-xs text-slate-500 w-6 text-right">
                          {i + 1}
                        </span>
                        <span className="text-sm text-slate-300 w-44 truncate">
                          {it.class_name}
                        </span>
                        <div className="flex-1 h-3 bg-slate-800 rounded-sm overflow-hidden">
                          <div
                            className="h-full bg-violet-500"
                            style={{ width: `${it.probability * 100}%` }}
                          />
                        </div>
                        <span className="text-xs font-mono text-slate-300 w-14 text-right">
                          {(it.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>
          )}
        </>
      )}
    </div>
  )
}

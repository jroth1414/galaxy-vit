/**
 * S-3 — "Most interesting galaxies" outlier panel.
 *
 * Three side-by-side columns, one per metric, each showing the top-K
 * test-set galaxies sorted descending by the metric value:
 *
 *   - **Predictive entropy** — model is categorically uncertain.
 *   - **BALD** — model is "confidently uncertain" (Houlsby+11 mutual info).
 *   - **|model - volunteer|** — biggest disagreement vs the volunteer
 *     consensus on questions where votes are available.
 *
 * Each thumbnail is captioned with `metric=value (median X.YY)` so
 * the user can see how outlier-y the top items are relative to the
 * population median. Clicking a thumbnail forwards the test-thumb
 * idx via `onSelect` (typically wired by Posteriors to load that
 * galaxy's full posterior bars in-place).
 *
 * Backend: `GET /api/outliers?metric=entropy|bald|disagreement&k=8`.
 */

import { useEffect, useState } from 'react'

interface OutlierItem {
  idx: number
  value: number
  thumbnail_url: string
}

interface OutliersResponse {
  metric: string
  median: number
  items: OutlierItem[]
}

const METRICS: { key: string; label: string; tooltip: string }[] = [
  {
    key: 'entropy',
    label: 'Most uncertain',
    tooltip: 'Predictive entropy summed across all 10 questions',
  },
  {
    key: 'bald',
    label: 'Most BALD',
    tooltip:
      'Bayesian Active Learning by Disagreement — "confidently uncertain"',
  },
  {
    key: 'disagreement',
    label: 'Disagrees most with volunteers',
    tooltip:
      'Mean L1 distance between model posterior and volunteer fractions',
  },
]

const PER_METRIC_K = 8

export function Outliers({
  onSelect,
}: {
  onSelect?: (idx: number) => void
}) {
  const [data, setData] = useState<Record<string, OutliersResponse>>({})
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    Promise.all(
      METRICS.map((m) =>
        fetch(`/api/outliers?metric=${m.key}&k=${PER_METRIC_K}`)
          .then((r) => {
            if (!r.ok) throw new Error(`${m.key}: HTTP ${r.status}`)
            return r.json() as Promise<OutliersResponse>
          })
          .then((body) => [m.key, body] as const),
      ),
    )
      .then((entries) => {
        if (cancelled) return
        const obj = Object.fromEntries(entries)
        setData(obj)
        setLoading(false)
      })
      .catch((e) => {
        if (cancelled) return
        setError(e instanceof Error ? e.message : String(e))
        setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  if (loading) {
    return <div className="text-sm text-slate-400">Loading outliers…</div>
  }
  if (error) {
    return (
      <div className="text-sm text-red-400 whitespace-pre-wrap">
        Could not load outliers: {error}
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {METRICS.map((m) => {
        const block = data[m.key]
        if (!block) return null
        return (
          <div
            key={m.key}
            className="rounded-md border border-slate-800 px-3 py-3 bg-slate-900/40"
          >
            <h4
              className="text-sm font-medium text-slate-100 mb-1"
              title={m.tooltip}
            >
              {m.label}
            </h4>
            <p className="text-[10px] text-slate-500 mb-2">
              median: {block.median.toFixed(2)} · top {block.items.length}
            </p>
            <div className="grid grid-cols-4 gap-1">
              {block.items.map((it) => (
                <button
                  key={it.idx}
                  type="button"
                  onClick={() => onSelect?.(it.idx)}
                  className="group relative rounded border border-slate-800 hover:border-indigo-400"
                  title={`#${it.idx} · ${m.key}=${it.value.toFixed(3)}`}
                >
                  <img
                    src={it.thumbnail_url}
                    alt={`g-${it.idx}`}
                    className="w-full aspect-square object-cover rounded-[2px]"
                    loading="lazy"
                  />
                  <span className="absolute bottom-0 left-0 right-0 text-[9px] bg-slate-900/80 text-slate-300 px-1 truncate">
                    {it.value.toFixed(2)}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}

/**
 * S-1 — Similar-galaxy kNN search tab.
 *
 * Query modes:
 *  - **Upload** an image: encoded server-side through the same Zoobot
 *    encoder that built the cache; cosine-kNN against the 2,462 cached
 *    test-set features.
 *  - **By cache idx**: type a row number (0..N-1) to query the cache
 *    row directly; the result always contains that idx first with
 *    distance ≈ 0 (sanity-check property).
 *  - **Preset query** (driven from other tabs): when the user clicks
 *    "Find similar" in Classify / Posteriors / Explorer, the App
 *    flips to this tab AND passes either an idx or an uploaded file.
 *
 * Backend endpoints:
 *   GET  /api/similar/{idx}?k=20
 *   POST /api/similar           (multipart upload)
 *
 * Visually: 4×5 thumbnail grid below the search controls, captioned
 * with the cosine distance.
 */

import { useEffect, useRef, useState } from 'react'
import { SampleGrid } from './SampleGrid'

export interface SimilarHit {
  idx: number
  distance: number
  thumbnail_url: string
}

interface SimilarResponse {
  query_idx: number | null
  hits: SimilarHit[]
}

export type SimilarPresetQuery =
  | { kind: 'idx'; value: number }
  | { kind: 'file'; value: File }

type Status = 'idle' | 'loading' | 'ok' | 'error'

const DEFAULT_K = 20
const MAX_K = 60

export function SimilarGalaxies({
  presetQuery,
  onPresetConsumed,
}: {
  presetQuery: SimilarPresetQuery | null
  onPresetConsumed: () => void
}) {
  const [k, setK] = useState<number>(DEFAULT_K)
  const [idxInput, setIdxInput] = useState<string>('0')
  const [hits, setHits] = useState<SimilarHit[]>([])
  const [queryIdx, setQueryIdx] = useState<number | null>(null)
  const [uploadName, setUploadName] = useState<string | null>(null)
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  function reset() {
    setHits([])
    setError(null)
  }

  async function queryByIdx(idx: number, kVal: number = k) {
    reset()
    setQueryIdx(idx)
    setUploadName(null)
    setStatus('loading')
    try {
      const r = await fetch(`/api/similar/${idx}?k=${kVal}`)
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`)
      const body = (await r.json()) as SimilarResponse
      setHits(body.hits)
      setStatus('ok')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setStatus('error')
    }
  }

  async function queryByFile(file: File, kVal: number = k) {
    reset()
    setQueryIdx(null)
    setUploadName(file.name)
    setStatus('loading')
    const form = new FormData()
    form.append('file', file)
    try {
      const r = await fetch(`/api/similar?k=${kVal}`, {
        method: 'POST',
        body: form,
      })
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`)
      const body = (await r.json()) as SimilarResponse
      setHits(body.hits)
      setStatus('ok')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setStatus('error')
    }
  }

  // Consume the preset query (from a "Find similar" button on another
  // tab). On first render with a non-null preset, fire the
  // corresponding query, then signal back so App can clear the preset.
  useEffect(() => {
    if (!presetQuery) return
    if (presetQuery.kind === 'idx') {
      setIdxInput(String(presetQuery.value))
      queryByIdx(presetQuery.value)
    } else {
      queryByFile(presetQuery.value)
    }
    onPresetConsumed()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presetQuery])

  function onPickFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (!f) return
    queryByFile(f)
  }

  function onSubmitIdx(e: React.FormEvent) {
    e.preventDefault()
    const parsed = parseInt(idxInput, 10)
    if (!Number.isFinite(parsed) || parsed < 0) {
      setError(`Invalid idx: ${idxInput}`)
      setStatus('error')
      return
    }
    queryByIdx(parsed)
  }

  const items = hits.map((h) => ({
    idx: h.idx,
    caption: `d=${h.distance.toFixed(3)}`,
  }))

  return (
    <div className="space-y-6">
      <section>
        <h2 className="text-lg font-medium text-slate-100 mb-1">
          Similar-galaxy search
        </h2>
        <p className="text-sm text-slate-400 mb-4">
          Cosine-kNN over the 2,462 DR8 test-set galaxies in the Zoobot
          ConvNeXt-nano feature space. Either upload an image or query
          by cache row index. Lower distance = morphologically more
          similar (0 = identical features, 2 = anti-parallel).
        </p>
      </section>

      <section className="flex flex-wrap items-center gap-3">
        <form onSubmit={onSubmitIdx} className="flex items-center gap-2">
          <label className="text-xs text-slate-400">Cache idx:</label>
          <input
            type="number"
            min={0}
            value={idxInput}
            onChange={(e) => setIdxInput(e.target.value)}
            className="text-xs bg-slate-800 border border-slate-700 text-slate-100 px-2 py-1 rounded w-20"
          />
          <button
            type="submit"
            className="text-xs rounded-md bg-violet-600 hover:bg-violet-500 text-white px-3 py-1"
          >
            Search by idx
          </button>
        </form>

        <span className="text-xs text-slate-500">or</span>

        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="text-xs rounded-md bg-slate-800 hover:bg-slate-700 text-slate-100 px-3 py-1 border border-slate-700"
        >
          Upload an image
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/png,image/jpeg"
          onChange={onPickFile}
          className="hidden"
        />

        <span className="text-xs text-slate-500 ml-auto">k =</span>
        <select
          value={k}
          onChange={(e) => setK(parseInt(e.target.value, 10))}
          className="text-xs bg-slate-800 border border-slate-700 text-slate-100 px-2 py-1 rounded"
        >
          {[5, 10, 20, 30, 50, MAX_K].map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      </section>

      {(queryIdx !== null || uploadName) && (
        <section className="text-xs text-slate-400 border-t border-slate-800 pt-3">
          {queryIdx !== null && (
            <>
              Query: cache row <span className="text-slate-200">#{queryIdx}</span>{' '}
              <img
                src={`/api/test_thumbs/${queryIdx}/thumbnail`}
                alt={`g-${queryIdx}`}
                className="inline-block w-12 h-12 object-cover rounded align-middle ml-2 mr-2 border border-slate-700"
              />
            </>
          )}
          {uploadName && (
            <>
              Query: uploaded image{' '}
              <span className="text-slate-200">{uploadName}</span>
            </>
          )}
        </section>
      )}

      {status === 'loading' && (
        <div className="text-sm text-slate-400">Searching…</div>
      )}
      {status === 'error' && error && (
        <div className="text-sm text-red-400 whitespace-pre-wrap">{error}</div>
      )}
      {status === 'ok' && (
        <section>
          <h3 className="text-sm font-medium text-slate-300 mb-2">
            Top {hits.length} nearest neighbours
          </h3>
          <SampleGrid items={items} />
        </section>
      )}
    </div>
  )
}

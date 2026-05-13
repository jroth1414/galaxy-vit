import { useRef, useState } from 'react'
import type { SimilarPresetQuery } from './SimilarGalaxies'

interface TopKItem {
  class_id: number
  class_name: string
  probability: number
}

interface PredictResponse {
  top_k: TopKItem[]
  attention_id: string
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

interface CompareResponse {
  m1: PredictResponse
  m3: PosteriorResponse
}

type Status = 'idle' | 'loading' | 'ok' | 'error'

export function Classify({
  onFindSimilar,
}: {
  onFindSimilar?: (q: SimilarPresetQuery) => void
}) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [response, setResponse] = useState<PredictResponse | null>(null)
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState<string | null>(null)
  // C-16: M1 vs M3 compare. Loaded lazily after the user opts in.
  const [compare, setCompare] = useState<CompareResponse | null>(null)
  const [compareStatus, setCompareStatus] = useState<Status>('idle')
  const [compareError, setCompareError] = useState<string | null>(null)

  function onPickFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (!f) return
    setImageFile(f)
    setImageUrl(URL.createObjectURL(f))
    setResponse(null)
    setStatus('idle')
    setError(null)
    setCompare(null)
    setCompareStatus('idle')
    setCompareError(null)
  }

  async function runCompare() {
    if (!imageFile) return
    setCompareStatus('loading')
    setCompareError(null)
    const form = new FormData()
    form.append('file', imageFile)
    try {
      const r = await fetch('/api/compare', { method: 'POST', body: form })
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`)
      const body = (await r.json()) as CompareResponse
      setCompare(body)
      // Bring M1's prediction into the standard slots so the existing
      // GradCAM tile + TopK render too.
      setResponse(body.m1)
      setStatus('ok')
      setCompareStatus('ok')
    } catch (e) {
      setCompareError(e instanceof Error ? e.message : String(e))
      setCompareStatus('error')
    }
  }

  async function classify() {
    if (!imageFile) return
    setStatus('loading')
    setError(null)
    const form = new FormData()
    form.append('file', imageFile)
    try {
      const r = await fetch('/api/predict', { method: 'POST', body: form })
      if (!r.ok) {
        throw new Error(`HTTP ${r.status}: ${await r.text()}`)
      }
      const body = (await r.json()) as PredictResponse
      setResponse(body)
      setStatus('ok')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setStatus('error')
    }
  }

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-lg font-medium text-slate-100 mb-1">
          Upload a galaxy image
        </h2>
        <p className="text-sm text-slate-400 mb-4">
          The model expects a 256×256 RGB DECaLS-style thumbnail. Larger or
          differently-sized images are center-cropped on the server.
        </p>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="rounded-md bg-slate-800 hover:bg-slate-700 text-slate-100 px-4 py-2 text-sm border border-slate-700"
          >
            {imageFile ? 'Pick a different image' : 'Pick an image'}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg"
            onChange={onPickFile}
            className="hidden"
          />
          <button
            type="button"
            onClick={classify}
            disabled={!imageFile || status === 'loading'}
            className="rounded-md bg-violet-600 hover:bg-violet-500 disabled:bg-slate-800 disabled:text-slate-500 text-white px-4 py-2 text-sm"
          >
            {status === 'loading' ? 'Classifying…' : 'Classify'}
          </button>
          {imageFile && (
            <button
              type="button"
              onClick={runCompare}
              disabled={compareStatus === 'loading'}
              className="rounded-md bg-slate-800 hover:bg-slate-700 text-slate-100 px-3 py-2 text-sm border border-slate-700"
              title="Run M1 (Galaxy10 plurality) and M3 (Dirichlet) on the same image"
            >
              {compareStatus === 'loading'
                ? 'Comparing…'
                : 'Compare with M3'}
            </button>
          )}
          {imageFile && onFindSimilar && (
            <button
              type="button"
              onClick={() => onFindSimilar({ kind: 'file', value: imageFile })}
              className="rounded-md bg-slate-800 hover:bg-slate-700 text-slate-100 px-3 py-2 text-sm border border-slate-700"
              title="Cosine-kNN against the 2,462 DR8 test-set galaxies"
            >
              Find similar →
            </button>
          )}
          {imageFile && (
            <span className="text-xs text-slate-500 truncate">
              {imageFile.name}
            </span>
          )}
        </div>
      </section>

      {imageUrl && (
        <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Tile title="Input">
            <img
              src={imageUrl}
              alt="upload preview"
              className="w-full aspect-square object-cover rounded-md border border-slate-800"
            />
          </Tile>

          <Tile title="GradCAM overlay">
            {response ? (
              <img
                src={`/api/attention/${response.attention_id}`}
                alt="GradCAM saliency overlay"
                className="w-full aspect-square object-cover rounded-md border border-slate-800"
              />
            ) : (
              <div className="aspect-square rounded-md border border-dashed border-slate-800 flex items-center justify-center text-slate-500 text-sm">
                Run classify to see where the model is looking.
              </div>
            )}
          </Tile>
        </section>
      )}

      {response && <TopK items={response.top_k} />}

      {compareStatus === 'error' && compareError && (
        <div className="rounded-md bg-red-950/40 border border-red-900 px-4 py-3 text-sm text-red-200">
          <strong>Compare error:</strong> {compareError}
        </div>
      )}

      {compare && (
        <section className="border-t border-slate-800 pt-6">
          <div className="flex items-baseline justify-between mb-3">
            <h3 className="text-sm font-medium text-slate-300">
              M3 — GZ DESI Dirichlet posterior
            </h3>
            <a
              href="https://github.com/jroth1414/galaxy-vit/blob/main/docs/m1_to_m3_mapping.md"
              target="_blank"
              rel="noreferrer"
              className="text-[11px] text-slate-500 hover:text-slate-300 underline"
            >
              M1 ↔ M3 mapping
            </a>
          </div>
          <p className="text-[11px] text-slate-500 mb-3">
            Same input image; M3 answers all 10 GZ DESI questions with
            full posterior bars. Plurality answers shown highlighted.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {compare.m3.questions.map((q) => (
              <div
                key={q.question}
                className={`rounded-md border border-slate-800 px-3 py-2 ${
                  q.active ? 'bg-slate-900/40' : 'bg-slate-900/20 opacity-50'
                }`}
              >
                <div className="flex items-baseline justify-between mb-1">
                  <span className="text-xs font-medium text-slate-100">
                    {q.question}
                  </span>
                  <span className="text-[10px] text-slate-500">
                    plurality: {q.plurality_answer}
                  </span>
                </div>
                <div className="space-y-1">
                  {q.answers.map((a, idx) => (
                    <div
                      key={a.name}
                      className="grid grid-cols-[7rem_1fr_3rem] items-center gap-2"
                    >
                      <span className="text-[10px] text-slate-400 truncate">
                        {a.name}
                      </span>
                      <div className="relative h-2 bg-slate-800 rounded">
                        <div
                          className={`absolute top-0 bottom-0 rounded ${
                            idx === q.plurality_index
                              ? 'bg-indigo-400'
                              : 'bg-indigo-700'
                          }`}
                          style={{ width: `${a.mean * 100}%` }}
                        />
                      </div>
                      <span className="text-[10px] font-mono text-slate-400 text-right">
                        {(a.mean * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {status === 'error' && error && (
        <div className="rounded-md bg-red-950/40 border border-red-900 px-4 py-3 text-sm text-red-200">
          <strong>Error:</strong> {error}
        </div>
      )}
    </div>
  )
}

function Tile({
  title,
  children,
}: {
  title: string
  children: React.ReactNode
}) {
  return (
    <div>
      <h3 className="text-xs uppercase tracking-wider text-slate-500 mb-2">
        {title}
      </h3>
      {children}
    </div>
  )
}

function TopK({ items }: { items: TopKItem[] }) {
  const max = Math.max(...items.map((i) => i.probability), 1e-6)
  return (
    <section>
      <h3 className="text-xs uppercase tracking-wider text-slate-500 mb-2">
        Top {items.length} classes
      </h3>
      <div className="space-y-2">
        {items.map((it) => (
          <div key={it.class_id} className="flex items-center gap-3">
            <span className="text-xs text-slate-500 w-6 text-right">
              {it.class_id}
            </span>
            <span className="text-sm text-slate-300 w-44 truncate">
              {it.class_name}
            </span>
            <div className="flex-1 h-3 bg-slate-800 rounded-sm overflow-hidden">
              <div
                className="h-full bg-violet-500"
                style={{ width: `${(it.probability / max) * 100}%` }}
              />
            </div>
            <span className="text-xs font-mono text-slate-300 w-14 text-right">
              {(it.probability * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </section>
  )
}

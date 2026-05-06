import { useRef, useState } from 'react'

interface TopKItem {
  class_id: number
  class_name: string
  probability: number
}

interface PredictResponse {
  top_k: TopKItem[]
  attention_id: string
}

type Status = 'idle' | 'loading' | 'ok' | 'error'

export function Classify() {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [response, setResponse] = useState<PredictResponse | null>(null)
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState<string | null>(null)

  function onPickFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (!f) return
    setImageFile(f)
    setImageUrl(URL.createObjectURL(f))
    setResponse(null)
    setStatus('idle')
    setError(null)
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

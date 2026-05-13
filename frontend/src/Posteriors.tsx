/**
 * T4.3 — Multi-Question Posteriors tab.
 *
 * Renders per-question Dirichlet posterior bars + 95% CI whiskers for one
 * galaxy at a time. Surfaces:
 *
 *  - Galaxy picker: pre-computed DR8 demo galaxies (with volunteer-overlay
 *    fractions from artifacts/demo_galaxies/manifest.json) OR upload.
 *  - 10 question groups, one panel each. Inactive panels (parent-dependency
 *    gating predicts a different branch) are greyed out per the GZ DESI
 *    decision tree.
 *  - Compare-to-volunteers overlay (when a demo galaxy is selected):
 *    empirical vote fractions rendered as a thin tick alongside the
 *    predicted bar.
 *
 * Backend endpoints driven by this file (added in T4.3 part 1):
 *   GET /api/demo_galaxies
 *   GET /api/demo_galaxies/{id}/thumbnail
 *   GET /api/demo_galaxies/{id}/posteriors
 *   POST /api/posteriors  (multipart upload)
 */

import { useEffect, useRef, useState } from 'react'
import { Outliers } from './Outliers'
import type { SimilarPresetQuery } from './SimilarGalaxies'

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

interface VolunteerOverlay {
  question: string
  valid: boolean
  fractions: number[]
}

interface DemoGalaxyPosteriorResponse {
  posterior: PosteriorResponse
  volunteer: VolunteerOverlay[]
}

interface DemoGalaxy {
  id: string
  smooth_or_featured_plurality: string
  thumbnail_url: string
}

type Status = 'idle' | 'loading' | 'ok' | 'error'

export function Posteriors({
  onFindSimilar,
}: {
  onFindSimilar?: (q: SimilarPresetQuery) => void
}) {
  const [demoGalaxies, setDemoGalaxies] = useState<DemoGalaxy[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [posterior, setPosterior] = useState<PosteriorResponse | null>(null)
  const [volunteer, setVolunteer] = useState<VolunteerOverlay[] | null>(null)
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState<string | null>(null)
  const [uploadName, setUploadName] = useState<string | null>(null)
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [outlierIdx, setOutlierIdx] = useState<number | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Initial fetch of the demo galaxies catalog.
  useEffect(() => {
    let cancelled = false
    fetch('/api/demo_galaxies')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}: ${r.statusText}`)
        return r.json() as Promise<{ galaxies: DemoGalaxy[] }>
      })
      .then((body) => {
        if (cancelled) return
        setDemoGalaxies(body.galaxies)
      })
      .catch((e) => {
        if (cancelled) return
        setError(
          `Could not load demo galaxies (${e instanceof Error ? e.message : String(e)})`,
        )
      })
    return () => {
      cancelled = true
    }
  }, [])

  async function selectDemo(id: string) {
    setSelectedId(id)
    setUploadName(null)
    setUploadFile(null)
    setOutlierIdx(null)
    setStatus('loading')
    setError(null)
    setPosterior(null)
    setVolunteer(null)
    try {
      const r = await fetch(`/api/demo_galaxies/${id}/posteriors`)
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`)
      const body = (await r.json()) as DemoGalaxyPosteriorResponse
      setPosterior(body.posterior)
      setVolunteer(body.volunteer)
      setStatus('ok')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setStatus('error')
    }
  }

  async function classifyUpload(file: File) {
    setSelectedId(null)
    setOutlierIdx(null)
    setUploadName(file.name)
    setUploadFile(file)
    setStatus('loading')
    setError(null)
    setPosterior(null)
    setVolunteer(null)
    const form = new FormData()
    form.append('file', file)
    try {
      const r = await fetch('/api/posteriors', { method: 'POST', body: form })
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`)
      const body = (await r.json()) as PosteriorResponse
      setPosterior(body)
      setStatus('ok')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setStatus('error')
    }
  }

  function onPickFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (!f) return
    classifyUpload(f)
  }

  async function selectOutlier(idx: number) {
    setOutlierIdx(idx)
    setSelectedId(null)
    setUploadName(null)
    setUploadFile(null)
    setStatus('loading')
    setError(null)
    setPosterior(null)
    setVolunteer(null)
    try {
      const r = await fetch(`/api/test_thumbs/${idx}/posteriors`, {
        method: 'POST',
      })
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`)
      const body = (await r.json()) as PosteriorResponse
      setPosterior(body)
      // Test-thumb endpoint doesn't carry volunteer overlay (those
      // come bundled with the 24 demo galaxies). Leave volunteer as
      // null so the bars render without the comparison tick.
      setStatus('ok')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setStatus('error')
    }
  }

  const selectedDemo = demoGalaxies.find((g) => g.id === selectedId) ?? null
  const calibrationLabel =
    posterior?.calibration_regime === 'single_T'
      ? `T=${posterior.temperature.toFixed(2)}`
      : 'raw'

  async function findSimilarToCurrent() {
    if (!onFindSimilar) return
    if (outlierIdx !== null) {
      onFindSimilar({ kind: 'idx', value: outlierIdx })
      return
    }
    if (uploadFile) {
      onFindSimilar({ kind: 'file', value: uploadFile })
      return
    }
    if (selectedDemo) {
      // Demo galaxies aren't in the test_thumbs cache (different index
      // space), so we have to POST the thumbnail as an upload.
      try {
        const r = await fetch(selectedDemo.thumbnail_url)
        if (!r.ok) throw new Error(`HTTP ${r.status} fetching thumbnail`)
        const blob = await r.blob()
        const file = new File([blob], `${selectedDemo.id}.jpg`, {
          type: blob.type || 'image/jpeg',
        })
        onFindSimilar({ kind: 'file', value: file })
      } catch (e) {
        setError(
          `Could not load demo galaxy for similarity search: ${
            e instanceof Error ? e.message : String(e)
          }`,
        )
      }
    }
  }

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-lg font-medium text-slate-100 mb-1">
          Multi-question posteriors
        </h2>
        <p className="text-sm text-slate-400 mb-4">
          Per-question Dirichlet-Multinomial posterior bars with 95%
          credible-interval whiskers. Questions greyed-out are gated off by
          the model's predicted decision-tree branch. Volunteer fractions
          shown as ticks when a demo galaxy is selected.
        </p>
      </section>

      <section className="border-t border-slate-800 pt-4">
        <h3 className="text-sm font-medium text-slate-300 mb-2">
          Most interesting galaxies
        </h3>
        <p className="text-[11px] text-slate-500 mb-3">
          Top-K test-set galaxies ranked by predictive entropy, BALD
          (Houlsby+11), and |model − volunteer| disagreement. Click a
          thumbnail to load its full posterior.
        </p>
        <Outliers onSelect={selectOutlier} />
      </section>

      <section>
        <h3 className="text-sm font-medium text-slate-300 mb-2">
          Demo galaxies (DR8 test set, stratified)
        </h3>
        <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2">
          {demoGalaxies.map((g) => (
            <button
              key={g.id}
              type="button"
              onClick={() => selectDemo(g.id)}
              className={`group relative rounded border-2 transition-all ${
                selectedId === g.id
                  ? 'border-indigo-400'
                  : 'border-slate-800 hover:border-slate-600'
              }`}
              title={`${g.id} · ${g.smooth_or_featured_plurality}`}
            >
              <img
                src={g.thumbnail_url}
                alt={g.id}
                className="w-full aspect-square object-cover rounded-[2px]"
              />
              <span className="absolute bottom-0 left-0 right-0 text-[10px] bg-slate-900/80 text-slate-300 px-1 truncate">
                {g.smooth_or_featured_plurality}
              </span>
            </button>
          ))}
        </div>

        <div className="mt-4 flex items-center gap-3">
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="px-4 py-1.5 text-sm rounded-md bg-slate-800 text-slate-100 hover:bg-slate-700"
          >
            …or upload your own
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={onPickFile}
          />
          {uploadName && (
            <span className="text-xs text-slate-400">{uploadName}</span>
          )}
        </div>
      </section>

      {(selectedDemo || uploadName || outlierIdx !== null) && (
        <section className="border-t border-slate-800 pt-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-slate-300">
              {selectedDemo
                ? `Galaxy ${selectedDemo.id} (volunteer plurality: ${selectedDemo.smooth_or_featured_plurality})`
                : uploadName
                  ? `Uploaded: ${uploadName}`
                  : `Test-set galaxy #${outlierIdx}`}
            </h3>
            <div className="flex items-center gap-3">
              {posterior && onFindSimilar && (selectedDemo || uploadFile || outlierIdx !== null) && (
                <button
                  type="button"
                  onClick={findSimilarToCurrent}
                  className="text-xs rounded-md bg-slate-800 hover:bg-slate-700 text-slate-100 px-3 py-1 border border-slate-700"
                  title="Cosine-kNN against the 2,462 DR8 test-set galaxies"
                >
                  Find similar →
                </button>
              )}
              {posterior && (
                <span className="text-xs text-slate-500">
                  calibration: {calibrationLabel}
                </span>
              )}
            </div>
          </div>

          {status === 'loading' && (
            <div className="text-sm text-slate-400">Predicting…</div>
          )}
          {status === 'error' && error && (
            <div className="text-sm text-red-400 whitespace-pre-wrap">
              {error}
            </div>
          )}
          {status === 'ok' && posterior && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {posterior.questions.map((q) => (
                <QuestionPanel
                  key={q.question}
                  question={q}
                  volunteer={
                    volunteer?.find((v) => v.question === q.question) ?? null
                  }
                />
              ))}
            </div>
          )}
        </section>
      )}
    </div>
  )
}

function QuestionPanel({
  question,
  volunteer,
}: {
  question: PosteriorQuestion
  volunteer: VolunteerOverlay | null
}) {
  const active = question.active
  return (
    <div
      className={`rounded-md border border-slate-800 px-4 py-3 ${
        active ? 'bg-slate-900/40' : 'bg-slate-900/20 opacity-50'
      }`}
    >
      <div className="flex items-baseline justify-between mb-2">
        <h4 className="text-sm font-medium text-slate-100">
          {question.question}
        </h4>
        <span className="text-[10px] text-slate-500">
          n_eff={question.n_effective.toFixed(1)}
          {!active && question.parent_question && (
            <span className="ml-2">
              (gated off: {question.parent_question}≠{question.parent_answer})
            </span>
          )}
        </span>
      </div>
      <div className="space-y-1.5">
        {question.answers.map((a, idx) => {
          const isPlurality = idx === question.plurality_index
          const vFrac = volunteer && volunteer.valid ? volunteer.fractions[idx] : null
          return (
            <AnswerBar
              key={a.name}
              name={a.name}
              mean={a.mean}
              ciLower={a.ci_lower}
              ciUpper={a.ci_upper}
              isPlurality={isPlurality}
              volunteerFraction={vFrac}
            />
          )
        })}
      </div>
    </div>
  )
}

function AnswerBar({
  name,
  mean,
  ciLower,
  ciUpper,
  isPlurality,
  volunteerFraction,
}: {
  name: string
  mean: number
  ciLower: number
  ciUpper: number
  isPlurality: boolean
  volunteerFraction: number | null
}) {
  return (
    <div className="grid grid-cols-[8rem_1fr_3rem] items-center gap-2">
      <span className="text-xs text-slate-300 truncate">{name}</span>
      <div className="relative h-4 bg-slate-800 rounded">
        {/* 95% CI whisker (lower..upper) */}
        <div
          className="absolute top-1/2 -translate-y-1/2 h-0.5 bg-slate-500"
          style={{
            left: `${ciLower * 100}%`,
            width: `${(ciUpper - ciLower) * 100}%`,
          }}
        />
        {/* Lower / upper end caps */}
        <div
          className="absolute top-0 bottom-0 w-px bg-slate-500"
          style={{ left: `${ciLower * 100}%` }}
        />
        <div
          className="absolute top-0 bottom-0 w-px bg-slate-500"
          style={{ left: `${ciUpper * 100}%` }}
        />
        {/* Posterior mean bar */}
        <div
          className={`absolute top-1 bottom-1 rounded ${
            isPlurality ? 'bg-indigo-400' : 'bg-indigo-700'
          }`}
          style={{ left: '0%', width: `${mean * 100}%` }}
        />
        {/* Volunteer-overlay tick */}
        {volunteerFraction !== null && (
          <div
            className="absolute top-0 bottom-0 w-1 bg-amber-300 rounded-sm"
            style={{ left: `calc(${volunteerFraction * 100}% - 2px)` }}
            title={`volunteer: ${(volunteerFraction * 100).toFixed(1)}%`}
          />
        )}
      </div>
      <span className="text-xs text-slate-400 tabular-nums text-right">
        {(mean * 100).toFixed(1)}%
      </span>
    </div>
  )
}

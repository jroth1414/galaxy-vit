/**
 * Reusable thumbnail-grid component (extracted from Explorer.tsx for S-1).
 *
 * Renders a responsive grid of test-thumbnail buttons. Each item is
 * keyed by `idx` and produces the URL `/api/test_thumbs/{idx}/thumbnail`.
 * Optional `caption` (e.g. cosine distance for kNN hits) is rendered as
 * a small overlay at the bottom of each tile.
 *
 * Used by: Explorer (lasso sample grid), SimilarGalaxies (kNN hits),
 * and S-3 Outliers panel (to be wired in a later commit).
 */

interface SampleGridItem {
  idx: number
  caption?: string
}

export function SampleGrid({
  items,
  selectedIdx,
  onSelect,
  emptyMessage = 'No items.',
  className = '',
}: {
  items: SampleGridItem[]
  selectedIdx?: number | null
  onSelect?: (idx: number) => void
  emptyMessage?: string
  className?: string
}) {
  if (items.length === 0) {
    return (
      <div className="text-sm text-slate-500 italic">{emptyMessage}</div>
    )
  }
  return (
    <div
      className={`grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-10 gap-1 ${className}`}
    >
      {items.map((it) => (
        <button
          key={it.idx}
          type="button"
          onClick={() => onSelect?.(it.idx)}
          className={`group relative rounded border ${
            selectedIdx === it.idx
              ? 'border-indigo-400'
              : 'border-slate-800 hover:border-slate-600'
          }`}
          title={`galaxy ${it.idx}${it.caption ? ` · ${it.caption}` : ''}`}
        >
          <img
            src={`/api/test_thumbs/${it.idx}/thumbnail`}
            alt={`g-${it.idx}`}
            className="w-full aspect-square object-cover rounded-[2px]"
            loading="lazy"
          />
          {it.caption && (
            <span className="absolute bottom-0 left-0 right-0 text-[10px] bg-slate-900/80 text-slate-300 px-1 truncate">
              {it.caption}
            </span>
          )}
        </button>
      ))}
    </div>
  )
}

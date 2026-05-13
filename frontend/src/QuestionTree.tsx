/**
 * A-5 — GZ DESI question-tree Sankey diagram.
 *
 * Builds two layers of nodes from /api/tree_flow{,/test_thumbs/{idx}}:
 *
 *   - question nodes (10): "smooth-or-featured", "disk-edge-on", ...
 *   - answer nodes (34):   one per (question, answer) pair
 *
 * Links:
 *   - question → its answer children, valued by reach(question, answer)
 *   - parent-question's gating answer → child question, valued by
 *     reach(child question)
 *
 * Nodes with reach below FADE_THRESHOLD are rendered with reduced
 * opacity so the user sees the parts of the tree the model's
 * prediction would have skipped. This is the "greying-out branches"
 * acceptance criterion from the plan.
 *
 * The Plotly Sankey layout doesn't always lay out a deep tree
 * cleanly; the trace is parameterised with `arrangement: 'snap'` so
 * the user can drag nodes manually if a particular galaxy's tree
 * comes out cluttered.
 */

import { useMemo } from 'react'
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

export interface TreeNode {
  id: string
  label: string
  kind: 'question' | 'answer'
  question: string
  answer: string | null
  reach: number
  parent_question: string | null
  parent_answer: string | null
}

const FADE_THRESHOLD = 0.05
const QUESTION_COLOR = '#475569' // slate-600
const ANSWER_COLOR = '#6366f1' // indigo-500
const LINK_COLOR = 'rgba(99, 102, 241, 0.35)'
const FADED_OPACITY = 0.2

function nodeColor(node: TreeNode): string {
  const base = node.kind === 'question' ? QUESTION_COLOR : ANSWER_COLOR
  if (node.reach < FADE_THRESHOLD) {
    return `${base}${Math.round(FADED_OPACITY * 255)
      .toString(16)
      .padStart(2, '0')}`
  }
  return base
}

export function QuestionTree({ nodes }: { nodes: TreeNode[] }) {
  const trace = useMemo(() => {
    if (nodes.length === 0) return null

    // Build node index by id.
    const idxOf: Record<string, number> = {}
    nodes.forEach((n, i) => {
      idxOf[n.id] = i
    })

    const labels = nodes.map((n) =>
      n.kind === 'question' ? n.label : `${n.label} (${(n.reach * 100).toFixed(0)}%)`,
    )
    const colors = nodes.map(nodeColor)

    const sources: number[] = []
    const targets: number[] = []
    const values: number[] = []
    const linkColors: string[] = []

    // Question → its answer-children.
    for (const n of nodes) {
      if (n.kind !== 'answer') continue
      const qIdx = idxOf[`q:${n.question}`]
      if (qIdx === undefined) continue
      sources.push(qIdx)
      targets.push(idxOf[n.id])
      // Plotly Sankey requires strictly positive link values; clamp tiny.
      values.push(Math.max(n.reach, 1e-4))
      linkColors.push(
        n.reach < FADE_THRESHOLD ? 'rgba(99,102,241,0.08)' : LINK_COLOR,
      )
    }

    // Parent answer → child question.
    for (const q of nodes) {
      if (q.kind !== 'question') continue
      if (!q.parent_question || !q.parent_answer) continue
      const fromId = `a:${q.parent_question}_${q.parent_answer}`
      const fromIdx = idxOf[fromId]
      if (fromIdx === undefined) continue
      sources.push(fromIdx)
      targets.push(idxOf[q.id])
      values.push(Math.max(q.reach, 1e-4))
      linkColors.push(
        q.reach < FADE_THRESHOLD ? 'rgba(71,85,105,0.08)' : 'rgba(71,85,105,0.5)',
      )
    }

    return {
      type: 'sankey' as const,
      arrangement: 'snap' as const,
      orientation: 'h' as const,
      node: {
        label: labels,
        color: colors,
        pad: 12,
        thickness: 14,
        line: { color: '#0f172a', width: 0.5 },
        hovertemplate:
          '<b>%{label}</b><br>reach = %{value:.3f}<extra></extra>',
      },
      link: {
        source: sources,
        target: targets,
        value: values,
        color: linkColors,
        hovertemplate:
          '%{source.label} → %{target.label}<br>flow = %{value:.3f}<extra></extra>',
      },
    }
  }, [nodes])

  if (!trace) {
    return <div className="text-sm text-slate-500 italic">No tree data.</div>
  }

  return (
    <Plot
      data={[trace]}
      layout={{
        width: 880,
        height: 560,
        margin: { l: 20, r: 20, t: 10, b: 10 },
        paper_bgcolor: '#0f172a',
        plot_bgcolor: '#0f172a',
        font: { color: '#cbd5e1', size: 11 },
      }}
      config={{
        displaylogo: false,
        modeBarButtonsToRemove: ['toImage'],
        responsive: true,
      }}
      style={{ borderRadius: 8 }}
    />
  )
}

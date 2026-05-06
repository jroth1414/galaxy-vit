export function ModelCard() {
  return (
    <div className="space-y-8 text-slate-300 text-sm leading-relaxed">
      <section>
        <h2 className="text-lg font-medium text-slate-100 mb-2">
          Galaxy-ViT M1 baseline (Zoobot ConvNeXt-nano)
        </h2>
        <p>
          A 10-class morphology classifier finetuned on{' '}
          <a
            className="text-violet-400 hover:text-violet-300"
            href="https://huggingface.co/datasets/matthieulel/galaxy10_decals"
          >
            Galaxy10 DECaLS
          </a>{' '}
          from a galaxy-pretrained{' '}
          <a
            className="text-violet-400 hover:text-violet-300"
            href="https://huggingface.co/mwalmsley/zoobot-encoder-convnext_nano"
          >
            Zoobot ConvNeXt-nano
          </a>{' '}
          encoder. Two-stage finetune (3 epochs head-only with the encoder
          frozen, then full finetune to convergence) with a Cui+19
          class-balanced cross-entropy (β=0.9999) to upweight Galaxy10's
          smallest class.
        </p>
      </section>

      <section className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <Stat value="0.8877" label="val top-1" hint="all 10 classes" />
        <Stat value="0.8738" label="val macro-F1" hint="DEVPLAN T1.5: ≥0.835" />
        <Stat value="+5.40" label="vs ViT-B/16" hint="macro-F1 absolute, %·100" />
        <Stat value="≈110ms" label="p95 CPU latency" hint="per /api/predict" />
      </section>

      <section>
        <h3 className="text-xs uppercase tracking-wider text-slate-500 mb-2">
          Training curves
        </h3>
        <img
          src="/static/curves.png"
          alt="loss + val top-1 + val macro-F1 vs epoch for the Zoobot finetune run"
          className="w-full rounded-md border border-slate-800 bg-slate-900"
        />
        <p className="text-xs text-slate-500 mt-2">
          The flat segment at the start (epochs 1–3) is the head-only stage —
          encoder gradients are masked out, so the head plateaus around
          0.65 macro-F1. At epoch 4 the encoder unfreezes and validation
          metrics jump 17.6 points in one epoch, then climb to the best
          checkpoint at epoch 11 before early-stopping at 16.
        </p>
      </section>

      <section>
        <h3 className="text-xs uppercase tracking-wider text-slate-500 mb-2">
          Architecture
        </h3>
        <ul className="space-y-1 list-disc list-inside">
          <li>
            <strong className="text-slate-200">Encoder</strong>:
            ConvNeXt-nano (~15M params), galaxy-pretrained on Galaxy Zoo
            DECaLS volunteer responses
          </li>
          <li>
            <strong className="text-slate-200">Head</strong>: fresh
            <code className="mx-1 px-1.5 py-0.5 rounded bg-slate-800 text-xs">
              nn.Linear(640, 10)
            </code>
            (10-way softmax)
          </li>
          <li>
            <strong className="text-slate-200">Loss</strong>: weighted
            cross-entropy with Cui+19 effective-number reweighting
            (β=0.9999)
          </li>
          <li>
            <strong className="text-slate-200">Saliency</strong>: GradCAM
            on the final ConvNeXt stage (
            <code className="mx-1 px-1.5 py-0.5 rounded bg-slate-800 text-xs">
              encoder.stages[-1]
            </code>
            ); 7×7 heatmap upsampled to image resolution
          </li>
        </ul>
      </section>

      <section>
        <h3 className="text-xs uppercase tracking-wider text-slate-500 mb-2">
          Class index → name
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-1 text-xs">
          {GALAXY10_LABELS.map((name, i) => (
            <div key={i} className="flex gap-2">
              <span className="text-slate-500 w-4 text-right font-mono">
                {i}
              </span>
              <span className="text-slate-300">{name}</span>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h3 className="text-xs uppercase tracking-wider text-slate-500 mb-2">
          Reproducibility
        </h3>
        <ul className="space-y-1 list-disc list-inside">
          <li>
            Train/val/test split CSV committed at{' '}
            <code className="mx-1 px-1.5 py-0.5 rounded bg-slate-800 text-xs">
              data/splits/galaxy10_split.csv
            </code>{' '}
            (seed=42, 70/15/15 stratified)
          </li>
          <li>
            Per-channel normalisation frozen at{' '}
            <code className="mx-1 px-1.5 py-0.5 rounded bg-slate-800 text-xs">
              configs/normalization.json
            </code>
          </li>
          <li>
            Run config + git SHA + pip freeze written next to the metrics
            JSON for every training run
          </li>
        </ul>
      </section>
    </div>
  )
}

function Stat({
  value,
  label,
  hint,
}: {
  value: string
  label: string
  hint: string
}) {
  return (
    <div className="rounded-md border border-slate-800 bg-slate-900/50 p-3">
      <div className="text-2xl font-mono font-semibold text-slate-100">
        {value}
      </div>
      <div className="text-xs text-slate-400 mt-1">{label}</div>
      <div className="text-[10px] text-slate-500 mt-0.5">{hint}</div>
    </div>
  )
}

const GALAXY10_LABELS = [
  'disturbed',
  'merging',
  'round-smooth',
  'in-between-round-smooth',
  'cigar-shaped-smooth',
  'barred-spiral',
  'unbarred-tight-spiral',
  'unbarred-loose-spiral',
  'edge-on-no-bulge',
  'edge-on-with-bulge',
] as const

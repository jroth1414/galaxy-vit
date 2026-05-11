import { useState } from 'react'
import { Classify } from './Classify'
import { Explorer } from './Explorer'
import { ModelCard } from './ModelCard'
import { Posteriors } from './Posteriors'

type Tab = 'classify' | 'posteriors' | 'explorer' | 'model-card'

export default function App() {
  const [tab, setTab] = useState<Tab>('classify')

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-slate-800 bg-slate-900/50">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold tracking-tight text-slate-50">
              Galaxy-ViT
            </h1>
            <p className="text-xs text-slate-400">
              Galaxy10 DECaLS · Zoobot ConvNeXt-nano finetune
            </p>
          </div>
          <nav className="flex gap-1">
            <TabButton
              active={tab === 'classify'}
              onClick={() => setTab('classify')}
            >
              Classify
            </TabButton>
            <TabButton
              active={tab === 'posteriors'}
              onClick={() => setTab('posteriors')}
            >
              Posteriors
            </TabButton>
            <TabButton
              active={tab === 'explorer'}
              onClick={() => setTab('explorer')}
            >
              Explorer
            </TabButton>
            <TabButton
              active={tab === 'model-card'}
              onClick={() => setTab('model-card')}
            >
              Model Card
            </TabButton>
          </nav>
        </div>
      </header>

      <main className="flex-1 max-w-5xl mx-auto w-full px-6 py-8">
        {tab === 'classify' && <Classify />}
        {tab === 'posteriors' && <Posteriors />}
        {tab === 'explorer' && <Explorer />}
        {tab === 'model-card' && <ModelCard />}
      </main>

      <footer className="border-t border-slate-800 bg-slate-900/50">
        <div className="max-w-5xl mx-auto px-6 py-3 text-xs text-slate-500 flex justify-between">
          <span>
            Trained on Galaxy10 DECaLS (matthieulel/galaxy10_decals) ·{' '}
            <a
              className="text-slate-400 hover:text-slate-200"
              href="https://github.com/jroth1414/galaxy-vit"
            >
              source
            </a>
          </span>
          <span>
            Encoder weights:{' '}
            <a
              className="text-slate-400 hover:text-slate-200"
              href="https://huggingface.co/mwalmsley/zoobot-encoder-convnext_nano"
            >
              Zoobot
            </a>
          </span>
        </div>
      </footer>
    </div>
  )
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`px-4 py-1.5 text-sm rounded-md transition-colors ${
        active
          ? 'bg-slate-800 text-slate-50'
          : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
      }`}
    >
      {children}
    </button>
  )
}

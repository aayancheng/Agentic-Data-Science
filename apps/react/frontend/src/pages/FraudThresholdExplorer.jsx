import React, { useEffect, useMemo, useState } from 'react'
import ThresholdControls from '../components/fraud/ThresholdControls'
import MetricsSummary from '../components/fraud/MetricsSummary'
import PrecisionRecallChart from '../components/fraud/PrecisionRecallChart'
import TransactionTable from '../components/fraud/TransactionTable'
import {
  scoreTransactions,
  computeConfusionCounts,
  computeMetrics,
  summarizeFlagged,
  buildMetricSeries,
} from '../lib/fraudMetrics'

const API_BASE = 'http://localhost:8000'

export default function FraudThresholdExplorer() {
  const [threshold, setThreshold] = useState(0.5)
  const [activePreset, setActivePreset] = useState(0.5)
  const [filter, setFilter] = useState('all')
  const [data, setData] = useState({ transactions: [], metadata: {}, generated_at: null })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const loadDemoData = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE}/api/fraud/demo`)
      if (!response.ok) throw new Error('Failed to load demo payload')
      const payload = await response.json()
      setData(payload)
    } catch (err) {
      setError(err.message || 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadDemoData()
  }, [])

  const scoredRows = useMemo(() => scoreTransactions(data.transactions, threshold), [data.transactions, threshold])
  const counts = useMemo(() => computeConfusionCounts(scoredRows), [scoredRows])
  const metrics = useMemo(() => computeMetrics(counts), [counts])
  const flaggedSummary = useMemo(() => summarizeFlagged(scoredRows), [scoredRows])
  const metricSeries = useMemo(() => buildMetricSeries(data.transactions), [data.transactions])

  return (
    <div>
      <section style={{ marginBottom: '20px' }}>
        <h2>Fraud Threshold Explorer</h2>
        <p style={{ color: '#4b5563', maxWidth: '760px' }}>
          Explore how a calibrated fraud model performs on 50 representative transactions. Adjust the threshold to see how recall, precision, and alert volume move in
          tandem. The sample blends 25 confirmed fraud cases with 25 legitimate ones to mimic a high-risk review queue.
        </p>
      </section>

      {loading && (
        <div style={{ padding: '40px', textAlign: 'center', border: '1px dashed #cbd5f5', borderRadius: '12px' }}>Loading demo transactions…</div>
      )}
      {error && (
        <div style={{ padding: '20px', border: '1px solid #fecaca', background: '#fee2e2', borderRadius: '12px', marginBottom: '20px' }}>
          <strong>Could not load demo payload.</strong>
          <p style={{ marginTop: '8px' }}>{error}</p>
          <button onClick={loadDemoData} style={{ padding: '6px 14px', borderRadius: '6px', border: 'none', background: '#0d9488', color: '#fff' }}>
            Retry
          </button>
        </div>
      )}

      {!loading && !error && (
        <>
          <ThresholdControls
            threshold={threshold}
            onThresholdChange={(value) => {
              setThreshold(value)
              setFilter('all')
            }}
            activePreset={activePreset}
            setActivePreset={setActivePreset}
          />

          <section
            style={{
              border: '1px solid #e5e7eb',
              borderRadius: '12px',
              padding: '16px',
              marginBottom: '20px',
              display: 'flex',
              flexWrap: 'wrap',
              gap: '24px',
            }}
          >
            <div>
              <div style={{ fontSize: '13px', color: '#6b7280' }}>Model version</div>
              <div style={{ fontSize: '20px', fontWeight: 600 }}>{data.metadata?.model_version || 'demo-calibrated-v1'}</div>
            </div>
            <div>
              <div style={{ fontSize: '13px', color: '#6b7280' }}>Sample size</div>
              <div style={{ fontSize: '20px', fontWeight: 600 }}>{data.metadata?.sample_size || 0} transactions</div>
            </div>
            <div>
              <div style={{ fontSize: '13px', color: '#6b7280' }}>Class balance</div>
              <div style={{ fontSize: '20px', fontWeight: 600 }}>
                {data.metadata?.class_balance?.fraud ?? 0} fraud / {data.metadata?.class_balance?.legit ?? 0} legit
              </div>
            </div>
            <div>
              <div style={{ fontSize: '13px', color: '#6b7280' }}>Last refreshed</div>
              <div style={{ fontSize: '20px', fontWeight: 600 }}>{data.generated_at || 'n/a'}</div>
            </div>
          </section>

          <MetricsSummary metrics={metrics} counts={counts} flaggedSummary={flaggedSummary} />
          <PrecisionRecallChart series={metricSeries} currentThreshold={threshold} />
          <TransactionTable rows={scoredRows} filter={filter} setFilter={setFilter} />
        </>
      )}
    </div>
  )
}

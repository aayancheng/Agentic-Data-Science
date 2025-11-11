import React from 'react'
import { formatPercent, formatCurrency } from '../../lib/fraudMetrics'

export default function MetricsSummary({ metrics, counts, flaggedSummary }) {
  const cards = [
    {
      label: 'Flagged Transactions',
      value: `${flaggedSummary.flaggedCount} (${formatPercent(flaggedSummary.flaggedPct)})`,
      sub: 'How many rows exceed the current threshold.',
    },
    {
      label: 'Recall (Fraud Captured)',
      value: formatPercent(metrics.recall),
      sub: 'TP / (TP + FN)',
    },
    {
      label: 'Precision (Alert Quality)',
      value: formatPercent(metrics.precision),
      sub: 'TP / (TP + FP)',
    },
    {
      label: 'False Positive Rate',
      value: formatPercent(metrics.fpr),
      sub: 'FP / (FP + TN)',
    },
    {
      label: 'Avg Ticket (Flagged)',
      value: formatCurrency(flaggedSummary.avgAmount),
      sub: 'Average amount of flagged transactions.',
    },
  ]

  const matrixCells = [
    { label: 'True Positive', value: counts.TP, color: '#10b981' },
    { label: 'False Negative', value: counts.FN, color: '#f97316' },
    { label: 'False Positive', value: counts.FP, color: '#f97316' },
    { label: 'True Negative', value: counts.TN, color: '#6b7280' },
  ]

  return (
    <section style={{ marginBottom: '20px' }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px' }}>
        {cards.map((card) => (
          <div key={card.label} style={{ border: '1px solid #e5e7eb', borderRadius: '12px', padding: '12px' }}>
            <div style={{ fontSize: '13px', color: '#6b7280' }}>{card.label}</div>
            <div style={{ fontSize: '24px', fontWeight: 600, marginTop: '4px' }}>{card.value}</div>
            <div style={{ fontSize: '12px', color: '#9ca3af', marginTop: '6px' }}>{card.sub}</div>
          </div>
        ))}
      </div>
      <div style={{ marginTop: '20px' }}>
        <h3 style={{ marginBottom: '8px' }}>Confusion Matrix (Demo Slice)</h3>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(2, minmax(140px, 1fr))',
            border: '1px solid #e5e7eb',
            borderRadius: '12px',
            overflow: 'hidden',
          }}
        >
          {matrixCells.map((cell) => (
            <div
              key={cell.label}
              style={{
                padding: '16px',
                borderBottom: '1px solid #e5e7eb',
                borderRight: '1px solid #e5e7eb',
                background: `${cell.color}15`,
              }}
            >
              <div style={{ fontSize: '12px', color: '#6b7280' }}>{cell.label}</div>
              <div style={{ fontSize: '26px', fontWeight: 600 }}>{cell.value}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

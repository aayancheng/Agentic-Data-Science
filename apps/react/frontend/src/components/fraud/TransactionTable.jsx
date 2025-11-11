import React from 'react'
import { filterRows, formatPercent, formatCurrency, OUTCOME_LABELS } from '../../lib/fraudMetrics'

const FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'flagged', label: 'Flagged' },
  { key: 'fp', label: 'False Positives' },
  { key: 'missed', label: 'Missed Fraud' },
]

const outcomeStyles = {
  TP: { background: '#d1fae5', color: '#065f46' },
  TN: { background: '#e5e7eb', color: '#374151' },
  FP: { background: '#ffedd5', color: '#9a3412' },
  FN: { background: '#fee2e2', color: '#991b1b' },
}

export default function TransactionTable({ rows, filter, setFilter }) {
  const filtered = filterRows(rows, filter)
  return (
    <section style={{ border: '1px solid #e5e7eb', borderRadius: '12px', padding: '16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <h3 style={{ margin: 0 }}>Transactions ({filtered.length})</h3>
          <p style={{ margin: '4px 0 0', color: '#6b7280' }}>Color indicates outcome at the current threshold.</p>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {FILTERS.map((option) => (
            <button
              key={option.key}
              onClick={() => setFilter(option.key)}
              style={{
                borderRadius: '999px',
                padding: '4px 12px',
                border: '1px solid',
                borderColor: filter === option.key ? '#0f766e' : '#d1d5db',
                background: filter === option.key ? '#ccfbf1' : '#fff',
                cursor: 'pointer',
              }}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>
      <div style={{ overflowX: 'auto', marginTop: '16px' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
          <thead>
            <tr style={{ textAlign: 'left', background: '#f9fafb' }}>
              {['Txn ID', 'Amount', 'Merchant', 'Country', 'Card Network', 'Channel', 'Fraud Prob', 'Model Flag', 'Actual'].map(
                (header) => (
                  <th key={header} style={{ padding: '10px', borderBottom: '1px solid #e5e7eb' }}>
                    {header}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 && (
              <tr>
                <td colSpan={9} style={{ padding: '16px', textAlign: 'center', color: '#9ca3af' }}>
                  Nothing to display for this filter.
                </td>
              </tr>
            )}
            {filtered.map((row) => {
              const style = outcomeStyles[row.outcome] || {}
              return (
                <tr key={row.txn_id} style={{ background: style.background, color: style.color }}>
                  <td style={cellStyle}>{row.txn_id}</td>
                  <td style={cellStyle}>{formatCurrency(row.amount)}</td>
                  <td style={cellStyle}>{row.merchant}</td>
                  <td style={cellStyle}>{row.country}</td>
                  <td style={cellStyle}>{row.card_network}</td>
                  <td style={cellStyle}>{row.channel}</td>
                  <td style={cellStyle}>{formatPercent(row.fraud_prob, 1)}</td>
                  <td style={cellStyle}>{row.model_flag ? 'Flagged' : 'Clear'}</td>
                  <td style={cellStyle}>{row.is_fraud_actual ? 'Fraud' : 'Legit'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <div style={{ marginTop: '12px', fontSize: '13px', color: '#6b7280' }}>
        Legend:&nbsp;
        {Object.entries(outcomeStyles).map(([key, style]) => (
          <span key={key} style={{ marginRight: '12px' }}>
            <span
              style={{
                display: 'inline-block',
                width: '12px',
                height: '12px',
                background: style.background,
                borderRadius: '4px',
                border: '1px solid rgba(0,0,0,0.1)',
                marginRight: '4px',
                verticalAlign: 'middle',
              }}
            />
            {OUTCOME_LABELS[key]}
          </span>
        ))}
      </div>
    </section>
  )
}

const cellStyle = {
  padding: '10px',
  borderBottom: '1px solid #f3f4f6',
}

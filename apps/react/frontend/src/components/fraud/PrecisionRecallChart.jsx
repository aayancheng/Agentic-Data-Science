import React from 'react'
import { formatPercent } from '../../lib/fraudMetrics'

export default function PrecisionRecallChart({ series, currentThreshold }) {
  if (!series || series.length === 0) return null

  const currentPoint = series.reduce(
    (closest, point) =>
      Math.abs(point.threshold - currentThreshold) < Math.abs(closest.threshold - currentThreshold) ? point : closest,
    series[0],
  )

  const width = 480
  const height = 220
  const padding = 30
  const thresholds = series.map((point) => point.threshold)
  const minX = Math.min(...thresholds)
  const maxX = Math.max(...thresholds)

  const scaleX = (value) =>
    padding + ((value - minX) / (maxX - minX || 1)) * (width - padding * 2)
  const scaleY = (value) => height - padding - value * (height - padding * 2)

  const buildPath = (key) =>
    series
      .map((point, idx) => `${idx === 0 ? 'M' : 'L'} ${scaleX(point.threshold)} ${scaleY(point[key])}`)
      .join(' ')

  const precisionPath = buildPath('precision')
  const recallPath = buildPath('recall')
  const currentX = scaleX(currentThreshold)

  return (
    <section style={{ border: '1px solid #e5e7eb', borderRadius: '12px', padding: '16px', marginBottom: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h3 style={{ margin: 0 }}>Precision & Recall vs Threshold</h3>
        <div style={{ display: 'flex', gap: '12px', fontSize: '13px' }}>
          <span style={{ color: '#6366f1' }}>● Precision</span>
          <span style={{ color: '#0ea5e9' }}>● Recall</span>
        </div>
      </div>
      <svg width={width} height={height} role="img" aria-label="Precision and recall chart">
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#9ca3af" strokeWidth="1" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#9ca3af" strokeWidth="1" />
        <path d={precisionPath} fill="none" stroke="#6366f1" strokeWidth="3" strokeLinecap="round" />
        <path d={recallPath} fill="none" stroke="#0ea5e9" strokeWidth="3" strokeLinecap="round" />
        <line
          x1={currentX}
          x2={currentX}
          y1={padding}
          y2={height - padding}
          stroke="#f97316"
          strokeWidth="2"
          strokeDasharray="6 4"
        />
        <text x={currentX + 6} y={padding + 12} fill="#f97316" fontSize="12px">
          {`Threshold ${currentThreshold.toFixed(2)}`}
        </text>
        {series.map((point) => (
          <React.Fragment key={`dots-${point.threshold}`}>
            <circle cx={scaleX(point.threshold)} cy={scaleY(point.precision)} r={4} fill="#6366f1" />
            <circle cx={scaleX(point.threshold)} cy={scaleY(point.recall)} r={4} fill="#0ea5e9" />
          </React.Fragment>
        ))}
        <text x={padding} y={padding - 8} fontSize="11px" fill="#6b7280">
          100%
        </text>
        <text x={padding - 18} y={height - padding + 12} fontSize="11px" fill="#6b7280">
          0%
        </text>
        <text x={width - padding - 20} y={height - padding + 24} fontSize="11px" fill="#6b7280">
          Threshold
        </text>
      </svg>
      <div style={{ marginTop: '12px', fontSize: '13px', color: '#6b7280' }}>
        Precision now: <strong>{formatPercent(currentPoint.precision)}</strong> • Recall now:{' '}
        <strong>{formatPercent(currentPoint.recall)}</strong>
      </div>
    </section>
  )
}

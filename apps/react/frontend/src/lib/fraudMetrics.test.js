import { describe, expect, it } from 'vitest'
import {
  scoreTransactions,
  computeConfusionCounts,
  computeMetrics,
  summarizeFlagged,
  buildMetricSeries,
  filterRows,
} from './fraudMetrics'

const sampleRows = [
  { txn_id: '1', fraud_prob: 0.9, is_fraud_actual: true, amount: 120 },
  { txn_id: '2', fraud_prob: 0.7, is_fraud_actual: false, amount: 60 },
  { txn_id: '3', fraud_prob: 0.2, is_fraud_actual: true, amount: 80 },
  { txn_id: '4', fraud_prob: 0.05, is_fraud_actual: false, amount: 45 },
]

describe('fraudMetrics helpers', () => {
  it('scores transactions and assigns outcomes', () => {
    const scored = scoreTransactions(sampleRows, 0.5)
    expect(scored.map((row) => row.outcome)).toEqual(['TP', 'FP', 'FN', 'TN'])
  })

  it('computes confusion counts + metrics', () => {
    const scored = scoreTransactions(sampleRows, 0.5)
    const counts = computeConfusionCounts(scored)
    expect(counts).toEqual({ TP: 1, FP: 1, FN: 1, TN: 1 })

    const metrics = computeMetrics(counts)
    expect(metrics.precision).toBeCloseTo(0.5)
    expect(metrics.recall).toBeCloseTo(0.5)
    expect(metrics.fpr).toBeCloseTo(0.5)
  })

  it('summarizes flagged rows', () => {
    const scored = scoreTransactions(sampleRows, 0.5)
    const summary = summarizeFlagged(scored)
    expect(summary.flaggedCount).toBe(2)
    expect(summary.flaggedPct).toBeCloseTo(0.5)
    expect(summary.avgAmount).toBeCloseTo((120 + 60) / 2)
  })

  it('builds metric series across thresholds', () => {
    const series = buildMetricSeries(sampleRows, [0.2, 0.8])
    expect(series).toHaveLength(2)
    expect(series[0].threshold).toBe(0.2)
    expect(series[1].threshold).toBe(0.8)
    expect(series[0].recall).toBeGreaterThan(series[1].recall)
  })

  it('filters rows by outcome', () => {
    const scored = scoreTransactions(sampleRows, 0.5)
    expect(filterRows(scored, 'flagged')).toHaveLength(2)
    expect(filterRows(scored, 'fp')).toHaveLength(1)
    expect(filterRows(scored, 'missed')).toHaveLength(1)
    expect(filterRows(scored, 'all')).toHaveLength(4)
  })
})

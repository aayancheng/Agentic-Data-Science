const DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

export const OUTCOME_LABELS = {
  TP: 'True Positive',
  FP: 'False Positive',
  TN: 'True Negative',
  FN: 'False Negative',
}

export function scoreTransactions(transactions = [], threshold = 0.5) {
  const t = Number(threshold) || 0
  return transactions.map((txn) => {
    const prob = Number(txn.fraud_prob ?? 0)
    const modelFlag = prob >= t
    const actual = Boolean(txn.is_fraud_actual)
    let outcome = 'TN'
    if (actual && modelFlag) outcome = 'TP'
    else if (actual && !modelFlag) outcome = 'FN'
    else if (!actual && modelFlag) outcome = 'FP'
    return {
      ...txn,
      model_flag: modelFlag,
      outcome,
    }
  })
}

export function computeConfusionCounts(rows = []) {
  return rows.reduce(
    (acc, row) => {
      acc[row.outcome] = (acc[row.outcome] || 0) + 1
      return acc
    },
    { TP: 0, FP: 0, FN: 0, TN: 0 },
  )
}

export function computeMetrics(counts) {
  const { TP = 0, FP = 0, FN = 0, TN = 0 } = counts
  const precision = TP + FP === 0 ? 0 : TP / (TP + FP)
  const recall = TP + FN === 0 ? 0 : TP / (TP + FN)
  const fpr = FP + TN === 0 ? 0 : FP / (FP + TN)
  return { precision, recall, fpr }
}

export function summarizeFlagged(rows = []) {
  const flagged = rows.filter((row) => row.model_flag)
  const flaggedCount = flagged.length
  const total = rows.length || 1
  const avgAmount =
    flaggedCount === 0
      ? 0
      : flagged.reduce((sum, row) => sum + Number(row.amount || 0), 0) / flaggedCount
  return {
    flaggedCount,
    flaggedPct: flaggedCount / total,
    avgAmount,
  }
}

export function buildMetricSeries(transactions = [], thresholds = DEFAULT_THRESHOLDS) {
  return thresholds.map((th) => {
    const scored = scoreTransactions(transactions, th)
    const counts = computeConfusionCounts(scored)
    const metrics = computeMetrics(counts)
    return {
      threshold: Number(th.toFixed(2)),
      ...metrics,
    }
  })
}

export function filterRows(rows = [], filterKey = 'all') {
  switch (filterKey) {
    case 'flagged':
      return rows.filter((row) => row.model_flag)
    case 'fp':
      return rows.filter((row) => row.outcome === 'FP')
    case 'missed':
      return rows.filter((row) => row.outcome === 'FN')
    default:
      return rows
  }
}

export function formatPercent(value, digits = 1) {
  return `${(Number(value) * 100).toFixed(digits)}%`
}

export function formatCurrency(value) {
  return `$${Number(value || 0).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`
}

export function thresholdStops() {
  return DEFAULT_THRESHOLDS
}

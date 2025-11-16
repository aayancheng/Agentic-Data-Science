import {
  scoreTransactions,
  computeConfusionCounts,
  computeMetrics,
  summarizeFlagged,
  buildMetricSeries,
  filterRows,
  formatPercent,
  formatCurrency,
  OUTCOME_LABELS,
} from './lib/fraudMetrics.js'

const DEFAULT_API_BASE = 'http://localhost:8000'
const FALLBACK_DATA_PATH = './src/data/fraud_threshold_demo.json'
const FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'flagged', label: 'Flagged' },
  { key: 'fp', label: 'False Positives' },
  { key: 'missed', label: 'Missed Fraud' },
]
const outcomeStyles = {
  TP: { background: 'rgba(134, 239, 172, 0.45)', color: '#166534' },
  TN: { background: 'rgba(191, 219, 254, 0.55)', color: '#1e3a8a' },
  FP: { background: 'rgba(254, 215, 170, 0.55)', color: '#9a3412' },
  FN: { background: 'rgba(254, 202, 202, 0.65)', color: '#991b1b' },
}

const state = {
  activeView: 'fraud',
  sidebarCollapsed: false,
  otherOpen: true,
  fraud: {
    threshold: 0.5,
    activePreset: 0.5,
    filter: 'all',
    activePanel: 'stats',
    loading: true,
    error: null,
    notice: null,
    data: { transactions: [], metadata: {}, generated_at: null },
    scoredRows: [],
    counts: { TP: 0, FP: 0, FN: 0, TN: 0 },
    metrics: { precision: 0, recall: 0, fpr: 0 },
    flaggedSummary: { flaggedCount: 0, flaggedPct: 0, avgAmount: 0 },
    metricSeries: [],
  },
}

const sidebar = document.getElementById('sidebar')
const collapseButton = document.getElementById('sidebar-collapse')
const navButtons = Array.from(document.querySelectorAll('.sidebar-link'))
const otherToggle = document.getElementById('sidebar-other-toggle')
const otherIndicator = document.getElementById('sidebar-other-indicator')
const otherLinks = document.getElementById('sidebar-other-links')

const fraudLoading = document.getElementById('fraud-loading')
const fraudError = document.getElementById('fraud-error')
const fraudErrorMessage = document.getElementById('fraud-error-message')
const fraudRetry = document.getElementById('fraud-retry')
const fraudNotice = document.getElementById('fraud-notice')
const thresholdControls = document.getElementById('threshold-controls')
const thresholdValue = document.getElementById('threshold-value')
const thresholdSlider = document.getElementById('threshold-slider')
const thresholdInput = document.getElementById('threshold-input')
const thresholdPresets = Array.from(document.querySelectorAll('#threshold-presets .threshold-preset'))
const fraudTabs = document.getElementById('fraud-tabs')
const fraudTabButtons = Array.from(fraudTabs.querySelectorAll('.fraud-tab'))
const statsPanel = document.getElementById('fraud-stats-panel')
const transactionsPanel = document.getElementById('fraud-transactions-panel')
const fraudOverview = document.getElementById('fraud-overview')
const metricsGrid = document.getElementById('metrics-grid')
const chartSvg = document.getElementById('precision-recall-chart')
const chartInsight = document.getElementById('chart-insight')
const confusionGrid = document.getElementById('confusion-grid')
const transactionsCount = document.getElementById('transactions-count')
const transactionFiltersContainer = document.getElementById('transaction-filters')
const transactionsBody = document.getElementById('transactions-body')
const transactionsLegend = document.getElementById('transactions-legend')

function init() {
  collapseButton.addEventListener('click', () => {
    state.sidebarCollapsed = !state.sidebarCollapsed
    updateSidebar()
  })

  otherToggle.addEventListener('click', () => {
    state.otherOpen = !state.otherOpen
    updateSidebar()
  })

  navButtons.forEach((button) => {
    button.addEventListener('click', () => {
      const target = button.dataset.target
      if (target) {
        state.activeView = target
        updateActiveView()
      }
    })
  })

  fraudRetry.addEventListener('click', () => {
    loadFraudData()
  })

  thresholdSlider.addEventListener('input', (event) => {
    setThreshold(Number(event.target.value), { presetValue: null })
  })

  thresholdInput.addEventListener('change', (event) => {
    const value = Number(event.target.value)
    if (!Number.isNaN(value)) {
      setThreshold(value, { presetValue: null })
    }
  })

  thresholdPresets.forEach((button) => {
    button.addEventListener('click', () => {
      const value = Number(button.dataset.value)
      setThreshold(value, { presetValue: value })
    })
  })

  fraudTabButtons.forEach((button) => {
    button.addEventListener('click', () => {
      const panel = button.dataset.panel
      if (panel) {
        state.fraud.activePanel = panel
        renderFraudTabs()
        renderFraudPanels()
      }
    })
  })

  FILTERS.forEach((option) => {
    const chip = document.createElement('button')
    chip.type = 'button'
    chip.className = 'filter-chip'
    chip.textContent = option.label
    chip.dataset.filter = option.key
    chip.addEventListener('click', () => {
      state.fraud.filter = option.key
      renderTransactionsPanel()
    })
    transactionFiltersContainer.appendChild(chip)
  })

  updateSidebar()
  updateActiveView()
  renderFraudTabs()
  renderThresholdControls()
  renderFraudPanels()
  loadFraudData()
}

function updateSidebar() {
  sidebar.classList.toggle('collapsed', state.sidebarCollapsed)
  collapseButton.textContent = state.sidebarCollapsed ? '›' : '‹'
  collapseButton.setAttribute('aria-label', state.sidebarCollapsed ? 'Expand navigation' : 'Collapse navigation')

  if (state.otherOpen) {
    otherLinks.classList.remove('hidden')
    otherToggle.setAttribute('aria-expanded', 'true')
    otherIndicator.textContent = '−'
  } else {
    otherLinks.classList.add('hidden')
    otherToggle.setAttribute('aria-expanded', 'false')
    otherIndicator.textContent = '+'
  }
}

function updateActiveView() {
  navButtons.forEach((button) => {
    const isActive = button.dataset.target === state.activeView
    button.classList.toggle('active', isActive)
  })

  document.querySelectorAll('[data-view]').forEach((section) => {
    const isVisible = section.dataset.view === state.activeView
    section.classList.toggle('hidden', !isVisible)
  })
}

async function loadFraudData() {
  state.fraud.loading = true
  state.fraud.error = null
  state.fraud.notice = null
  renderFraudStatus()
  try {
    const { payload, source } = await fetchFraudPayload()
    state.fraud.data = payload
    state.fraud.metricSeries = buildMetricSeries(payload.transactions || [])
    state.fraud.activePanel = 'stats'
    if (source === 'local-fallback') {
      state.fraud.notice = 'Showing bundled demo data because the API could not be reached.'
    }
    setThreshold(state.fraud.threshold, { presetValue: state.fraud.activePreset, skipFilterReset: true })
  } catch (error) {
    state.fraud.error = error.message || 'Unknown error'
  } finally {
    state.fraud.loading = false
    renderFraudStatus()
    if (!state.fraud.error) {
      renderThresholdControls()
      renderFraudTabs()
      renderFraudPanels()
    }
  }
}

function normalizeBaseUrl(base) {
  return base.replace(/\/$/, '')
}

async function fetchFraudPayload() {
  const candidates = []
  if (typeof window !== 'undefined') {
    const explicitBase = window.__FRAUD_API_BASE__
    if (typeof explicitBase === 'string' && explicitBase.trim()) {
      candidates.push(`${normalizeBaseUrl(explicitBase.trim())}/api/fraud/demo`)
    }

    if (window.location && /^https?:/.test(window.location.origin)) {
      candidates.push(new URL('/api/fraud/demo', window.location.origin).toString())
    }
  }

  candidates.push(`${normalizeBaseUrl(DEFAULT_API_BASE)}/api/fraud/demo`)

  const attempted = []
  for (const url of [...new Set(candidates)]) {
    try {
      const response = await fetch(url)
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`.trim())
      }
      const payload = await response.json()
      return { payload, source: url }
    } catch (error) {
      attempted.push(`${url} (${error.message})`)
    }
  }

  try {
    const fallbackResponse = await fetch(FALLBACK_DATA_PATH)
    if (!fallbackResponse.ok) {
      throw new Error(`${fallbackResponse.status} ${fallbackResponse.statusText}`.trim())
    }
    const payload = await fallbackResponse.json()
    return { payload, source: 'local-fallback' }
  } catch (fallbackError) {
    const triedMessage = attempted.length ? `Tried: ${attempted.join('; ')}.` : ''
    const finalMessage = fallbackError && fallbackError.message ? ` Last error: ${fallbackError.message}.` : ''
    throw new Error(`Failed to load demo payload. ${triedMessage}${finalMessage}`.trim())
  }
}

function setThreshold(value, options = {}) {
  const { presetValue = null, skipFilterReset = false } = options
  const clamped = Math.min(0.95, Math.max(0.05, Number(value) || 0.5))
  state.fraud.threshold = Number(clamped.toFixed(2))
  state.fraud.activePreset = presetValue
  if (!skipFilterReset) {
    state.fraud.filter = 'all'
  }
  updateFraudDerived()
  renderThresholdControls()
  renderFraudPanels()
}

function updateFraudDerived() {
  const transactions = state.fraud.data.transactions || []
  state.fraud.scoredRows = scoreTransactions(transactions, state.fraud.threshold)
  state.fraud.counts = computeConfusionCounts(state.fraud.scoredRows)
  state.fraud.metrics = computeMetrics(state.fraud.counts)
  state.fraud.flaggedSummary = summarizeFlagged(state.fraud.scoredRows)
  if (!state.fraud.metricSeries.length && transactions.length > 0) {
    state.fraud.metricSeries = buildMetricSeries(transactions)
  }
}

function renderFraudStatus() {
  fraudLoading.classList.toggle('hidden', !state.fraud.loading)
  const hasError = Boolean(state.fraud.error)
  fraudError.classList.toggle('hidden', !hasError)
  if (hasError) {
    fraudErrorMessage.textContent = state.fraud.error
  }

  const showNotice = Boolean(state.fraud.notice) && !state.fraud.loading && !hasError
  fraudNotice.classList.toggle('hidden', !showNotice)
  if (showNotice) {
    fraudNotice.textContent = state.fraud.notice
  } else {
    fraudNotice.textContent = ''
  }

  const hideContent = state.fraud.loading || hasError
  thresholdControls.classList.toggle('hidden', hideContent)
  fraudTabs.classList.toggle('hidden', hideContent)
  statsPanel.classList.toggle('hidden', hideContent || state.fraud.activePanel !== 'stats')
  transactionsPanel.classList.toggle('hidden', hideContent || state.fraud.activePanel !== 'transactions')
}

function renderThresholdControls() {
  thresholdValue.textContent = state.fraud.threshold.toFixed(2)
  thresholdSlider.value = state.fraud.threshold
  thresholdInput.value = state.fraud.threshold.toFixed(2)
  thresholdPresets.forEach((button) => {
    const value = Number(button.dataset.value)
    button.classList.toggle('active', value === state.fraud.activePreset)
  })
}

function renderFraudTabs() {
  fraudTabButtons.forEach((button) => {
    const isActive = button.dataset.panel === state.fraud.activePanel
    button.classList.toggle('active', isActive)
    button.setAttribute('aria-selected', String(isActive))
  })

  statsPanel.classList.toggle('hidden', state.fraud.activePanel !== 'stats')
  transactionsPanel.classList.toggle('hidden', state.fraud.activePanel !== 'transactions')
}

function renderFraudPanels() {
  if (state.fraud.loading || state.fraud.error) return
  renderOverview()
  renderMetricsGrid()
  renderPrecisionRecallChart()
  renderConfusionMatrix()
  renderTransactionsPanel()
}

function renderOverview() {
  const metadata = state.fraud.data.metadata || {}
  const classBalance = metadata.class_balance || {}
  const items = [
    {
      label: 'Model version',
      value: metadata.model_version || 'demo-calibrated-v1',
    },
    {
      label: 'Sample size',
      value: `${metadata.sample_size || 0} transactions`,
    },
    {
      label: 'Class balance',
      value: `${classBalance.fraud ?? 0} fraud / ${classBalance.legit ?? 0} legit`,
    },
    {
      label: 'Last refreshed',
      value: state.fraud.data.generated_at || 'n/a',
    },
  ]

  fraudOverview.innerHTML = items
    .map(
      (item) => `
        <div class="overview-item">
          <div class="overview-label">${item.label}</div>
          <div class="overview-value">${item.value}</div>
        </div>
      `,
    )
    .join('')
}

function renderMetricsGrid() {
  const { metrics, counts, flaggedSummary } = state.fraud
  const cards = [
    {
      label: 'Flagged Transactions',
      value: `${flaggedSummary.flaggedCount} (${formatPercent(flaggedSummary.flaggedPct)})`,
      sub: 'Rows exceeding the current threshold.',
    },
    { label: 'Recall (Fraud Captured)', value: formatPercent(metrics.recall), sub: 'TP / (TP + FN)' },
    { label: 'Precision (Alert Quality)', value: formatPercent(metrics.precision), sub: 'TP / (TP + FP)' },
    { label: 'False Positive Rate', value: formatPercent(metrics.fpr), sub: 'FP / (FP + TN)' },
    {
      label: 'Avg Ticket (Flagged)',
      value: formatCurrency(flaggedSummary.avgAmount),
      sub: 'Average amount of flagged transactions.',
    },
    { label: 'Fraud Confirmed', value: counts.TP + counts.FN, sub: 'Total fraud instances in sample.' },
  ]

  metricsGrid.innerHTML = cards
    .map(
      (card) => `
        <div class="metric-card metrics-surface">
          <div class="label">${card.label}</div>
          <div class="value">${card.value}</div>
          <div class="subtext">${card.sub}</div>
        </div>
      `,
    )
    .join('')
}

function renderPrecisionRecallChart() {
  const series = state.fraud.metricSeries
  if (!series || series.length === 0) {
    chartSvg.innerHTML = ''
    chartInsight.textContent = 'No transactions available to plot.'
    return
  }

  const width = 560
  const height = 260
  const padding = 30
  const thresholds = series.map((point) => point.threshold)
  const minX = Math.min(...thresholds)
  const maxX = Math.max(...thresholds)
  const domain = maxX - minX || 1

  const scaleX = (value) => padding + ((value - minX) / domain) * (width - padding * 2)
  const scaleY = (value) => height - padding - value * (height - padding * 2)

  const buildPath = (key) =>
    series
      .map((point, idx) => `${idx === 0 ? 'M' : 'L'} ${scaleX(point.threshold)} ${scaleY(point[key])}`)
      .join(' ')

  const precisionPath = buildPath('precision')
  const recallPath = buildPath('recall')

  const currentPoint = series.reduce((closest, point) => {
    return Math.abs(point.threshold - state.fraud.threshold) < Math.abs(closest.threshold - state.fraud.threshold)
      ? point
      : closest
  }, series[0])
  const currentX = scaleX(state.fraud.threshold)

  chartSvg.setAttribute('viewBox', `0 0 ${width} ${height}`)
  chartSvg.innerHTML = `
    <defs>
      <linearGradient id="precisionGradient" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#1d4ed8" stop-opacity="0.2" />
        <stop offset="100%" stop-color="#6366f1" stop-opacity="0.2" />
      </linearGradient>
      <linearGradient id="recallGradient" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#0ea5e9" stop-opacity="0.2" />
        <stop offset="100%" stop-color="#38bdf8" stop-opacity="0.2" />
      </linearGradient>
    </defs>
    <rect
      x="${padding}"
      y="${padding}"
      width="${width - padding * 2}"
      height="${height - padding * 2}"
      fill="url(#precisionGradient)"
      opacity="0.2"
      rx="14"
    />
    <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" stroke="#94a3b8" stroke-width="1" />
    <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}" stroke="#94a3b8" stroke-width="1" />
    <path d="${precisionPath}" fill="none" stroke="#1d4ed8" stroke-width="3" stroke-linecap="round" />
    <path d="${recallPath}" fill="none" stroke="#0ea5e9" stroke-width="3" stroke-linecap="round" />
    <line x1="${currentX}" x2="${currentX}" y1="${padding}" y2="${height - padding}" stroke="#f97316" stroke-width="2" stroke-dasharray="6 4" />
    <text x="${currentX + 6}" y="${padding + 12}" fill="#f97316" font-size="12">Threshold ${state.fraud.threshold.toFixed(2)}</text>
    ${series
      .map((point) => {
        const cx = scaleX(point.threshold)
        const precisionCy = scaleY(point.precision)
        const recallCy = scaleY(point.recall)
        return `
          <circle cx="${cx}" cy="${precisionCy}" r="4" fill="#1d4ed8"></circle>
          <circle cx="${cx}" cy="${recallCy}" r="4" fill="#0ea5e9"></circle>
        `
      })
      .join('')}
    <text x="${padding}" y="${padding - 8}" font-size="11" fill="#475569">100%</text>
    <text x="${padding - 18}" y="${height - padding + 12}" font-size="11" fill="#475569">0%</text>
    <text x="${width - padding - 40}" y="${height - padding + 24}" font-size="11" fill="#475569">Threshold</text>
  `

  chartInsight.innerHTML = `Precision now: <strong>${formatPercent(currentPoint.precision)}</strong> • Recall now: <strong>${formatPercent(
    currentPoint.recall,
  )}</strong>`
}

function renderConfusionMatrix() {
  const cells = [
    { label: 'True Positive', value: state.fraud.counts.TP },
    { label: 'False Negative', value: state.fraud.counts.FN },
    { label: 'False Positive', value: state.fraud.counts.FP },
    { label: 'True Negative', value: state.fraud.counts.TN },
  ]

  confusionGrid.innerHTML = cells
    .map(
      (cell) => `
        <div class="matrix-cell">
          <div class="label">${cell.label}</div>
          <div class="value">${cell.value}</div>
        </div>
      `,
    )
    .join('')
}

function renderTransactionsPanel() {
  if (state.fraud.loading || state.fraud.error) return

  const filterButtons = Array.from(transactionFiltersContainer.querySelectorAll('.filter-chip'))
  filterButtons.forEach((button) => {
    button.classList.toggle('active', button.dataset.filter === state.fraud.filter)
  })

  const filtered = filterRows(state.fraud.scoredRows, state.fraud.filter)
  transactionsCount.textContent = filtered.length

  if (filtered.length === 0) {
    transactionsBody.innerHTML = `
      <tr>
        <td colspan="9" style="padding: 18px; text-align: center; color: #94a3b8;">Nothing to display for this filter.</td>
      </tr>
    `
  } else {
    transactionsBody.innerHTML = filtered
      .map((row) => {
        const style = outcomeStyles[row.outcome] || {}
        const background = style.background ? `background: ${style.background};` : ''
        const color = style.color ? `color: ${style.color};` : ''
        return `
          <tr style="${background} ${color}">
            <td>${row.txn_id}</td>
            <td>${formatCurrency(row.amount)}</td>
            <td>${row.merchant}</td>
            <td>${row.country}</td>
            <td>${row.card_network}</td>
            <td>${row.channel}</td>
            <td>${formatPercent(row.fraud_prob, 1)}</td>
            <td>${row.model_flag ? 'Flagged' : 'Clear'}</td>
            <td>${row.is_fraud_actual ? 'Fraud' : 'Legit'}</td>
          </tr>
        `
      })
      .join('')
  }

  transactionsLegend.innerHTML = Object.entries(outcomeStyles)
    .map(
      ([key, style]) => `
        <span class="legend-item">
          <span class="legend-swatch" style="background: ${style.background};"></span>
          ${OUTCOME_LABELS[key]}
        </span>
      `,
    )
    .join('')
}

init()

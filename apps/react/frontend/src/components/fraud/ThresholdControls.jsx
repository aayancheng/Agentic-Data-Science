import React from 'react'

const presets = [
  { label: 'High Recall (0.20)', value: 0.2 },
  { label: 'Balanced (0.50)', value: 0.5 },
  { label: 'High Precision (0.80)', value: 0.8 },
]

export default function ThresholdControls({ threshold, onThresholdChange, activePreset, setActivePreset }) {
  const handleSlider = (event) => {
    const value = Number(event.target.value)
    setActivePreset(null)
    onThresholdChange(value)
  }

  const handleInput = (event) => {
    const value = Number(event.target.value)
    if (Number.isNaN(value)) return
    const clamped = Math.min(0.95, Math.max(0.05, value))
    setActivePreset(null)
    onThresholdChange(Number(clamped.toFixed(2)))
  }

  return (
    <section style={{ background: '#f4f4f7', padding: '16px', borderRadius: '12px', marginBottom: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <h3 style={{ margin: 0 }}>Decision Threshold</h3>
          <p style={{ margin: '4px 0 0', color: '#555' }}>Drag the slider or pick a preset to balance recall vs precision.</p>
        </div>
        <div style={{ fontSize: '20px', fontWeight: 600 }}>Current: {threshold.toFixed(2)}</div>
      </div>
      <input
        type="range"
        min="0.05"
        max="0.95"
        step="0.01"
        value={threshold}
        onChange={handleSlider}
        style={{ width: '100%', margin: '16px 0' }}
      />
      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
        {presets.map((preset) => (
          <button
            key={preset.value}
            onClick={() => {
              setActivePreset(preset.value)
              onThresholdChange(preset.value)
            }}
            style={{
              borderRadius: '999px',
              padding: '6px 14px',
              border: '1px solid',
              borderColor: activePreset === preset.value ? '#0d9488' : '#d1d5db',
              background: activePreset === preset.value ? '#d1fae5' : '#fff',
              cursor: 'pointer',
            }}
          >
            {preset.label}
          </button>
        ))}
        <label style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px' }}>
          Manual:
          <input
            type="number"
            min="0.05"
            max="0.95"
            step="0.01"
            value={threshold}
            onChange={handleInput}
            style={{ width: '80px', padding: '4px 6px' }}
          />
        </label>
      </div>
    </section>
  )
}

import React from 'react'
export default function Nav({tab, setTab}){
  const tabs = [['fraud','Fraud'], ['taxi','Taxi'], ['sentiment','Sentiment']]
  return (
    <div style={{display:'flex', gap:'8px', margin:'12px 0'}}>
      {tabs.map(([k,label]) => (
        <button key={k} onClick={() => setTab(k)} style={{padding:'8px 12px', borderRadius: 8, border: '1px solid #ddd', background: tab===k? '#f2f2f2':'#fff'}}>
          {label}
        </button>
      ))}
    </div>
  )
}

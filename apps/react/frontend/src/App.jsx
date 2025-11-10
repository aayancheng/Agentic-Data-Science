import React, { useState } from 'react'
import Nav from './components/Nav'
import Fraud from './pages/Fraud'
import Taxi from './pages/Taxi'
import Sentiment from './pages/Sentiment'

export default function App(){
  const [tab, setTab] = useState('fraud')
  return (
    <div style={{fontFamily:'ui-sans-serif', padding:'16px', maxWidth: '1000px', margin: '0 auto'}}>
      <h1>Agentic Data Science — Demo</h1>
      <Nav tab={tab} setTab={setTab} />
      {tab === 'fraud' && <Fraud/>}
      {tab === 'taxi' && <Taxi/>}
      {tab === 'sentiment' && <Sentiment/>}
      <footer style={{marginTop:'24px', opacity:0.6}}>Backend: http://localhost:8000 • Frontend Dev: http://localhost:5173</footer>
    </div>
  )
}

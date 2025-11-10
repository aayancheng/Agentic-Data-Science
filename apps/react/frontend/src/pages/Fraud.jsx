import React, { useState } from 'react'
export default function Fraud(){
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const submit = async () => {
    if(!file) return
    const form = new FormData()
    form.append('file', file)
    const r = await fetch('http://localhost:8000/api/fraud/score', {method:'POST', body: form})
    const data = await r.json()
    setResult(data)
  }
  return (<div>
    <h2>Fraud scoring</h2>
    <input type="file" onChange={e=>setFile(e.target.files[0])}/>
    <button onClick={submit} style={{marginLeft:8}}>Score</button>
    {result && <pre style={{background:'#f9f9f9', padding:12, marginTop:12}}>{JSON.stringify(result, null, 2)}</pre>}
  </div>)
}

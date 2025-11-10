import React, { useState } from 'react'
export default function Sentiment(){
  const [text, setText] = useState('I loved this movie!')
  const [res, setRes] = useState(null)
  const submit = async () => {
    const r = await fetch('http://localhost:8000/api/sentiment/predict', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({text})
    })
    setRes(await r.json())
  }
  return (<div>
    <h2>IMDb — sentiment</h2>
    <textarea value={text} onChange={e=>setText(e.target.value)} rows={5} style={{width:'100%'}}/>
    <div><button onClick={submit} style={{marginTop:8}}>Predict</button></div>
    {res && <pre style={{background:'#f9f9f9', padding:12, marginTop:12}}>{JSON.stringify(res, null, 2)}</pre>}
  </div>)
}

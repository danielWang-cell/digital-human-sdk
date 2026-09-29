import React, {useState} from 'react';
import {createRoot} from 'react-dom/client';
import {AreaChart, Area, ResponsiveContainer, Tooltip, XAxis} from 'recharts';
import './style.css';

const latency=[{t:'0s',v:18.4},{t:'5s',v:19.1},{t:'10s',v:18.8},{t:'15s',v:20.2},{t:'20s',v:19.4},{t:'25s',v:18.9}];
const stages=['Audio','Mel','Face Align','Inference','Blend','Output'];
function App(){
  const [active,setActive]=useState(3); const [playing,setPlaying]=useState(false);
  return <main>
    <nav><div className="brand"><span className="brand-dot"/> Digital Human Lab</div><div className="navlinks"><a href="#demo">演示</a><a href="#pipeline">系统</a><a href="#metrics">性能</a><button className="nav-button">查看 GitHub ↗</button></div></nav>
    <header><div><span className="eyebrow">A SMALL SYSTEM FOR HUMAN EXPRESSION</span><h1>让一张照片<br/><span>听懂一段声音。</span></h1><p className="sub">从音频到表情，把数字人的每一步都变得清晰、可观察。</p><div className="hero-actions"><button className="primary" onClick={()=>document.getElementById('demo').scrollIntoView({behavior:'smooth'})}>观看演示 <b>↓</b></button><button className="text-button">探索系统 <b>→</b></button></div></div><div className="hero-orb"><div className="orb-ring ring-one"/><div className="orb-ring ring-two"/><div className="orb-core">DH<br/><small>LAB</small></div></div></header>
    <section id="demo" className="hero grid2"><div className="video card"><div className="video-top"><span>DEMO / SAMPLE 01</span><span>00:12 / 00:30</span></div><div className="portrait"><div className="scan"/><div className="face-mark">PRESET OUTPUT / 25 FPS</div></div><div className="controls"><button onClick={()=>setPlaying(!playing)}>{playing?'暂停':'播放'}样本</button><input type="range" defaultValue="38"/><span>WAV2LIP</span></div></div>
      <div className="card status"><div className="section-title">CURRENT RUN <span>LIVE PREVIEW</span></div><div className="status-row"><span>MODEL</span><b>Wav2Lip</b></div><div className="status-row"><span>BACKEND</span><b>ncnn / CPU</b></div><div className="status-row"><span>QUEUE</span><b className="green">04 / 50</b></div><div className="status-row"><span>OUTPUT</span><b className="green">SYNCED</b></div><div className="metric-big">18.80 <small>FPS</small></div><div className="muted">steady over the last 30 seconds</div></div></section>
    <section id="pipeline" className="card pipeline"><div className="section-title">HOW IT WORKS <span>点击节点查看耗时</span></div><div className="nodes">{stages.map((s,i)=><React.Fragment key={s}><button className={i===active?'node active':'node'} onClick={()=>setActive(i)}><em>0{i+1}</em>{s}<small>{[4.2,7.8,2.1,48.6,3.7,8.4][i]} ms</small></button>{i<stages.length-1&&<div className={i<active?'line lit':'line'}/>}</React.Fragment>)}</div><div className="trace-note">{stages[active]} stage selected · frame context #0184 · PTS 7360 ms</div></section>
    <section id="metrics" className="grid2 lower"><div className="card chart"><div className="section-title">THROUGHPUT / FPS <span>LAST 30 SECONDS</span></div><ResponsiveContainer width="100%" height={180}><AreaChart data={latency}><XAxis dataKey="t" hide/><Tooltip contentStyle={{background:'#ffffff',border:'1px solid #d7e1ec',color:'#142238'}}/><Area type="monotone" dataKey="v" stroke="#1f6feb" fill="#6ea8fe" fillOpacity=".12"/></AreaChart></ResponsiveContainer></div><div className="card logs"><div className="section-title">ENGINE LOG <span>LIVE</span></div><p><b>14:32:08</b> static face cache ready</p><p><b>14:32:09</b> PTS drift corrected <mark>+2 ms</mark></p><p><b>14:32:10</b> render queue stable</p><p><b>14:32:11</b> output frame committed</p></div></section>
    <footer><span>Digital Human Lab</span><span>PERFORMANCE · PIPELINE · ENGINEERING LOG</span></footer>
  </main>
}
createRoot(document.getElementById('root')).render(<App/>);

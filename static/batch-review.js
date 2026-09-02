const root=document.getElementById('root');
const meta=document.getElementById('meta');
const runId=new URLSearchParams(location.search).get('run');

function formatSeconds(ms){return ms ? `${Math.round(ms/1000)} 秒` : '—';}
function escapeHtml(value){const node=document.createElement('span');node.textContent=value;return node.innerHTML;}
async function load(){
  if(!runId){root.textContent='缺少批次编号。';return;}
  const response=await fetch(`/api/batch-runs/${encodeURIComponent(runId)}/items`);
  if(!response.ok){root.textContent='找不到此批次。';return;}
  const data=await response.json(),run=data.run||{};
  meta.innerHTML=[`批次 ${run.run_id||runId}`,`图片 ${run.completed_images||0}/${run.samples||0}`,`确认头部 ${run.confirmed_people||0}`,`模型 ${run.provider||'—'}`].map(value=>`<span>${escapeHtml(value)}</span>`).join('');
  if(!data.items.length){root.textContent='批次尚无可检查的完成结果。';return;}
  root.className='run-grid';
  root.innerHTML=data.items.map(item=>`<article class="run-card"><h2>样本 ${escapeHtml(item.sample_id)}</h2><p class="result-meta">确认 ${item.confirmed_people} 人头 · 动漫化 ${item.anime_people} · 安全回退 ${item.safe_people} · ${formatSeconds(item.elapsed_ms)}</p><div class="pair"><figure><img loading="lazy" src="${item.original_url}" alt="样本 ${item.sample_id} 原图" onclick="window.open(this.src,'_blank')"><figcaption>原图</figcaption></figure><figure><img loading="lazy" src="${item.result_url}" alt="样本 ${item.sample_id} 结果" onclick="window.open(this.src,'_blank')"><figcaption>结果（点击可放大）</figcaption></figure></div></article>`).join('');
}
load();

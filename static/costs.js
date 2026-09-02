const money = (value, currency = '¥') => value ? `${currency}${Number(value).toFixed(2)}` : `${currency}0.00`;
const seconds = value => value ? `${(value / 1000).toFixed(1)} 秒` : '—';
async function loadCosts() {
  const response = await fetch('/api/costs/summary');
  if (!response.ok) throw new Error('无法读取成本账本');
  const data = await response.json();
  document.getElementById('costNote').textContent = data.note;
  const publishable = data.reference_cny_per_publishable_photo === null ? '尚未标记' : money(data.reference_cny_per_publishable_photo);
  document.getElementById('costCards').innerHTML = [
    ['已处理照片', data.photos_processed], ['成功 AI 生成', data.generated_since_enabled], ['安全回退', data.safe_fallbacks], ['重试次数', data.retries], ['方舟套餐额度', data.quota_units], ['参考成本（人民币）', money(data.reference_cny)], ['fal 参考成本（美元）', money(data.reference_usd, '$')], ['每张可发布照片参考成本', publishable],
  ].map(([label, value]) => `<article class="cost-card"><span>${label}</span><strong>${value}</strong></article>`).join('');
  document.getElementById('costRows').innerHTML = data.providers.length ? data.providers.map(item => `<tr><td><strong>${item.provider}</strong><small>${item.model}</small></td><td>${item.generated}</td><td>${item.fallbacks}</td><td>${seconds(item.average_elapsed_ms)}</td><td>${money(item.reference_cny)}${item.reference_usd ? ` / $${item.reference_usd.toFixed(3)}` : ''}${item.quota_units ? ` / ${item.quota_units} 额度` : ''}</td></tr>`).join('') : '<tr><td colspan="5">还没有生成记录。完成下一张真实测试图后，这里会自动出现。</td></tr>';
}
loadCosts().catch(error => { document.getElementById('costNote').textContent = error.message; });

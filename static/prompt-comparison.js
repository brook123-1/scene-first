const root = document.getElementById('reviewRoot');

async function load() {
  const response = await fetch('/api/prompt-comparison/items');
  const data = await response.json();
  if (!data.items.length) { root.textContent = data.message || '尚无结果。'; return; }
  root.className = '';
  root.innerHTML = data.items.map((item, index) => `
    <article class="review-card" data-id="${item.id}">
      <p class="eyebrow">同图候选 ${item.blind_label} · ${index + 1}/${data.items.length}</p>
      <div class="review-images">
        <figure><img src="${item.original_url}" alt="原图"><figcaption>原图</figcaption></figure>
        <figure><img src="${item.result_url}" alt="候选结果"><figcaption>候选结果</figcaption></figure>
      </div>
      <div class="rating">
        <label>你敢直接发布吗？<select data-key="publishable"><option value="">请选择</option><option value="true">可以</option><option value="false">不可以</option></select></label>
        <label>自然感 1–5<input data-key="naturalness" type="range" min="1" max="5" value="3"></label>
        <label>隐私感 1–5<input data-key="privacy" type="range" min="1" max="5" value="3"></label>
      </div>
      <label style="display:block;margin-top:12px;font-size:12px">原因或违和点<textarea data-key="notes"></textarea></label>
      <button class="primary" style="margin-top:14px" onclick="save(this)">保存这项评分 <span>→</span></button>
    </article>`).join('');
}

async function save(button) {
  const card = button.closest('article');
  const read = key => card.querySelector(`[data-key=${key}]`).value;
  if (!read('publishable')) { button.textContent = '请先选择是否敢发布'; return; }
  const response = await fetch('/api/prompt-comparison/rate', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({item_id: card.dataset.id, publishable: read('publishable') === 'true', naturalness: Number(read('naturalness')), privacy: Number(read('privacy')), notes: read('notes')})
  });
  button.textContent = response.ok ? '已保存 ✓' : '保存失败'; button.disabled = response.ok;
}

load();

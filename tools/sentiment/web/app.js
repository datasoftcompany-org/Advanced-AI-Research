const form = document.querySelector('#form');
const inputs = document.querySelector('#inputs');
const status = document.querySelector('#status');
const error = document.querySelector('#error');
const results = document.querySelector('#results');
const progress = document.querySelector('#progress');
let csv = '';
const labels = {positive: 'إيجابي', negative: 'سلبي', ok: 'مكتمل', empty: 'فارغ', invalid: 'رد غير صالح', failure: 'فشل الطلب'};
document.querySelectorAll('[name=mode]').forEach(input => input.addEventListener('change', () => {
  const mode = form.elements.mode.value;
  document.querySelector('#text-panel').hidden = mode !== 'text';
  document.querySelector('#csv-panel').hidden = mode !== 'csv';
}));
function showResult(event) {
  const body = document.querySelector('#rows');
  body.replaceChildren();
  let succeeded = 0;
  for (const row of event.rows) {
    const tr = document.createElement('tr');
    for (const value of [row.row, row.text, labels[row.sentiment] || '—', labels[row.status] || row.status]) {
      const td = document.createElement('td');
      td.textContent = value;
      td.dir = 'auto';
      tr.append(td);
    }
    body.append(tr);
    succeeded += row.status === 'ok';
  }
  csv = event.csv;
  document.querySelector('#summary').textContent = `${succeeded} تصنيف مكتمل من ${event.rows.length} صفًا. راجع حالة كل صف.`;
  results.hidden = false;
  status.textContent = 'اكتمل التشغيل. النتائج جاهزة للتنزيل.';
}
form.addEventListener('submit', async event => {
  event.preventDefault();
  error.hidden = true;
  results.hidden = true;
  csv = '';
  progress.value = 0;
  status.textContent = 'جارٍ تجهيز الطلب…';
  const data = {mode: form.elements.mode.value, model: document.querySelector('#model').value};
  inputs.disabled = true;
  let done = false;
  try {
    if (data.mode === 'text') {
      data.text = document.querySelector('#text').value;
      if (!data.text.trim()) throw new Error('أدخل مراجعة بالإنجليزية.');
    } else {
      const file = document.querySelector('#file').files[0];
      if (!file) throw new Error('اختر ملف CSV.');
      if (file.size > 10 * 1024 * 1024) throw new Error('الملف يتجاوز 10 MiB.');
      data.csv = await file.text();
      data.column = document.querySelector('#column').value;
    }
    const response = await fetch('/api/classify', {
      method: 'POST', headers: {'Content-Type': 'application/json', 'X-Session-Token': document.querySelector('[name=session-token]').content},
      body: JSON.stringify(data)
    });
    if (!response.ok) throw new Error(await response.text());
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = '';
    while (true) {
      const chunk = await reader.read();
      pending += decoder.decode(chunk.value || new Uint8Array(), {stream: !chunk.done});
      const lines = pending.split('\n');
      pending = lines.pop();
      for (const line of lines) {
        if (!line) continue;
        const update = JSON.parse(line);
        if (update.type === 'error') throw new Error(update.message);
        if (update.type === 'progress') {
          progress.max = update.total;
          progress.value = update.completed;
          status.textContent = `جارٍ التصنيف: ${update.completed} من ${update.total}`;
        }
        if (update.type === 'done') {showResult(update); done = true;}
      }
      if (chunk.done) break;
    }
    if (!done) throw new Error('انقطع الاتصال قبل اكتمال النتائج. أعد المحاولة.');
  } catch (cause) {
    results.hidden = true;
    csv = '';
    status.textContent = 'تعذّر إكمال التشغيل.';
    error.textContent = cause.message;
    error.hidden = false;
  } finally {
    inputs.disabled = false;
  }
});
document.querySelector('#download').addEventListener('click', () => {
  if (!csv) return;
  const url = URL.createObjectURL(new Blob(['\ufeff', csv], {type: 'text/csv;charset=utf-8'}));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'sentiment-results.csv';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

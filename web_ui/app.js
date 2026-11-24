'use strict';

const byId = id => document.getElementById(id);
const canvas = byId('canvas');
const draw = canvas.getContext('2d');
const tabs = [...document.querySelectorAll('[data-mode]')];
const examples = [...document.querySelectorAll('[data-example]')];
const exampleNames = {front: '正面楼梯', angled: '斜视楼梯', dark: '暗光楼梯'};
let mode = 'edges';
let result;
let busy = false;
let name = 'stairs';
let point;

function status(message, error = false) {
  byId('status').textContent = message;
  byId('status').classList.toggle('error', error);
}

function fit() {
  if (!result) return;
  const stage = byId('stage');
  const scale = Math.min(stage.clientWidth / result.width, stage.clientHeight / result.height);
  canvas.style.width = `${result.width * scale}px`;
  canvas.style.height = `${result.height * scale}px`;
}

function render() {
  if (!result) return;
  const {width, height, rgb, heat, groups, ranges, location} = result;
  canvas.width = width;
  canvas.height = height;
  draw.drawImage(mode === 'heat' ? heat : rgb, 0, 0, width, height);
  if (mode === 'edges') {
    const size = Math.max(2, width / 350);
    groups.forEach((lines, index) => {
      draw.lineWidth = size;
      draw.lineCap = 'round';
      draw.strokeStyle = '#ffdb46';
      let widest = -1;
      let center;
      for (const [[x1, y1], [x2, y2]] of lines) {
        draw.beginPath();
        draw.moveTo(x1, y1);
        draw.lineTo(x2, y2);
        draw.stroke();
        if (Math.hypot(x2 - x1, y2 - y1) > widest) {
          widest = Math.hypot(x2 - x1, y2 - y1);
          center = [(x1 + x2) / 2, (y1 + y2) / 2];
        }
      }
      if (!center) return;
      const radius = Math.max(10, width / 85);
      const x = Math.min(width - radius, Math.max(radius, center[0]));
      const y = Math.min(height - radius, Math.max(radius, center[1]));
      draw.fillStyle = '#132747';
      draw.beginPath();
      draw.arc(x, y, radius, 0, 2 * Math.PI);
      draw.fill();
      draw.fillStyle = '#fff';
      draw.font = `bold ${Math.round(radius * 1.12)}px system-ui`;
      draw.textAlign = 'center';
      draw.textBaseline = 'middle';
      draw.fillText(index + 1, x, y);
    });
  }
  if (mode === 'heat' && point) {
    draw.lineWidth = Math.max(2, width / 360);
    draw.strokeStyle = '#fff';
    draw.beginPath();
    draw.arc(point.x, point.y, Math.max(7, width / 120), 0, 2 * Math.PI);
    draw.stroke();
  }
  byId('stage').classList.toggle('measure', mode === 'heat');
  byId('caption').textContent = mode === 'heat' ? '暖色较近 · 冷色较远 · 点击读取相对深度' : `${location} · ${width} × ${height}`;
  byId('coordinate').textContent = mode === 'heat' && point ? `(${point.x}, ${point.y}) · ${point.value.toFixed(3)}` : '';
  fit();
}

async function loadImage(data, type) {
  const image = new Image();
  image.src = `data:image/${type};base64,${data}`;
  await image.decode();
  return image;
}

function readNpy(base64) {
  const bytes = Uint8Array.from(atob(base64), char => char.charCodeAt(0));
  const header = new DataView(bytes.buffer);
  if (bytes[6] !== 1) throw new Error('深度文件版本不受支持。');
  const start = 10 + header.getUint16(8, true);
  return {bytes, values: new Float32Array(bytes.buffer, start)};
}

async function analyze(getImage, location, filename) {
  if (busy) return;
  busy = true;
  result = undefined;
  point = undefined;
  byId('placeholder').hidden = false;
  byId('placeholder').textContent = 'GPU 正在分析图片…';
  byId('caption').textContent = location;
  byId('coordinate').textContent = '';
  byId('count').textContent = '—';
  canvas.hidden = true;
  byId('save-image').disabled = byId('save-depth').disabled = true;
  examples.forEach(button => { button.disabled = true; });
  status('正在识别阶缘并估计深度');
  try {
    const image = await getImage();
    const response = await fetch('/api/analyze', {method: 'POST', body: image,
      headers: {'Content-Type': image.type || 'application/octet-stream'}});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '推理请求失败。');
    const [rgb, heat] = await Promise.all([loadImage(payload.rgb, 'jpeg'), loadImage(payload.heat, 'png')]);
    const depth = readNpy(payload.depth_npy);
    if (depth.values.length !== payload.width * payload.height) throw new Error('深度数组与图像尺寸不一致。');
    result = {...payload, rgb, heat, depth, location};
    name = filename;
    byId('count').textContent = payload.count;
    byId('placeholder').hidden = true;
    canvas.hidden = false;
    byId('save-image').disabled = byId('save-depth').disabled = false;
    status(payload.count ? `分析完成 · GPU ${payload.seconds.toFixed(2)} 秒` : '分析完成 · 未检出可见台阶');
    render();
  } catch (error) {
    byId('placeholder').textContent = '请选择另一张图片或重试';
    status(error.message || '分析失败，请重试。', true);
  } finally {
    busy = false;
    examples.forEach(button => { button.disabled = false; });
  }
}

function upload(file) {
  if (!file || busy) return;
  if (!file.size || file.size > 12 * 1024 * 1024) {
    status('请选择不超过 12 MB 的图片。', true);
    return;
  }
  analyze(() => file, file.name, file.name.replace(/\.[^.]+$/, ''));
}

function example(id) {
  analyze(async () => {
    const response = await fetch(`/api/example/${id}`);
    if (!response.ok) throw new Error('无法加载示例图片。');
    return response.blob();
  }, exampleNames[id], `stairs-${id}`);
}

tabs.forEach(button => button.addEventListener('click', () => {
  mode = button.dataset.mode;
  tabs.forEach(tab => tab.setAttribute('aria-pressed', String(tab === button)));
  render();
}));
examples.forEach(button => button.addEventListener('click', () => example(button.dataset.example)));
byId('file').addEventListener('change', event => {
  upload(event.target.files[0]);
  event.target.value = '';
});
const drop = byId('drop');
['dragover', 'dragleave', 'drop'].forEach(type => drop.addEventListener(type, event => {
  event.preventDefault();
  drop.classList.toggle('drag', type === 'dragover');
  if (type === 'drop') upload(event.dataTransfer.files[0]);
}));
canvas.addEventListener('click', event => {
  if (mode !== 'heat' || !result) return;
  const rect = canvas.getBoundingClientRect();
  const x = Math.min(result.width - 1, Math.max(0, Math.floor((event.clientX - rect.left) * result.width / rect.width)));
  const y = Math.min(result.height - 1, Math.max(0, Math.floor((event.clientY - rect.top) * result.height / rect.height)));
  point = {x, y, value: result.depth.values[y * result.width + x]};
  render();
});

function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

byId('save-image').addEventListener('click', () => canvas.toBlob(blob => {
  if (blob) download(blob, `${name}-${mode}.png`);
}));
byId('save-depth').addEventListener('click', () => {
  download(new Blob([result.depth.bytes], {type: 'application/octet-stream'}), `${name}-relative-depth.npy`);
});
new ResizeObserver(fit).observe(byId('stage'));
fetch('/api/health').then(response => {
  if (!response.ok) throw new Error('Unavailable');
  byId('health').textContent = 'GPU 已就绪';
  byId('health').classList.add('ready');
  if (!busy && !result) example('front');
}).catch(() => { byId('health').textContent = '连接失败'; });

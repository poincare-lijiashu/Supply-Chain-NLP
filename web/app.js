/* 工作台逻辑：健康检查 + 批量分类渲染 */
const $ = (sel) => document.querySelector(sel);
const input = $("#input"), runBtn = $("#run"), resultsEl = $("#results");

async function checkHealth() {
  try {
    const r = await fetch("/health");
    const j = await r.json();
    $("#health-dot").classList.add("ok");
    $("#model-badge").textContent = `模型：${j.model} · ${j.classes.length} 类目`;
  } catch (e) {
    $("#health-dot").classList.add("down");
    $("#model-badge").textContent = "服务不可用";
  }
}

function render(items) {
  resultsEl.classList.remove("empty");
  resultsEl.innerHTML = "";
  for (const it of items) {
    const pct = (it.prob * 100).toFixed(1);
    const topk = Object.entries(it.probs || {})
      .sort((a, b) => b[1] - a[1]).slice(0, 4)
      .map(([k, v]) => `<div><span>${k}</span><span>${(v * 100).toFixed(1)}%</span></div>`)
      .join("");
    const el = document.createElement("div");
    el.className = "item";
    el.innerHTML = `
      <div class="item-head">
        <span class="item-text"></span>
        <span class="item-class">${it.class_name} · ${pct}%</span>
      </div>
      <div class="bar"><i class="${it.prob < 0.6 ? "low" : ""}"></i></div>
      <div class="prob-line">置信度 ${pct}%（阈值 60%，低于转人工）</div>
      ${it.needs_human_review ? `<span class="review">⚠ ${it.review_reason}</span>` : ""}
      <div class="topk">${topk}</div>`;
    el.querySelector(".item-text").textContent = it.text;
    resultsEl.appendChild(el);
    requestAnimationFrame(() => {
      el.querySelector(".bar > i").style.width = `${Math.max(it.prob * 100, 2)}%`;
    });
  }
}

async function run() {
  const texts = input.value.split("\n").map(s => s.trim()).filter(Boolean);
  if (!texts.length) { $("#hint").textContent = "请先输入至少一条描述"; return; }
  runBtn.disabled = true; runBtn.textContent = "分类中…"; $("#hint").textContent = "";
  try {
    const r = await fetch("/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts }),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    render(await r.json());
  } catch (e) {
    $("#hint").textContent = `调用失败：${e.message}`;
  } finally {
    runBtn.disabled = false; runBtn.textContent = "开始分类";
  }
}

$("#run").addEventListener("click", run);
$("#clear").addEventListener("click", () => {
  input.value = "";
  resultsEl.classList.add("empty");
  resultsEl.innerHTML = document.querySelector("#results").dataset.emptyHtml
    || '<div class="empty-guide"><p class="empty-title">三步完成寄件初审</p><ol><li>左侧输入托寄物描述（可批量多行）</li><li>点击「开始分类」，模型实时给出 15 类置信度</li><li>低置信度与违禁品命中自动标记 → 转人工复核</li></ol></div>';
});
document.querySelectorAll(".chip").forEach(c =>
  c.addEventListener("click", () => { input.value += (input.value ? "\n" : "") + c.dataset.text; }));
input.addEventListener("keydown", e => { if (e.key === "Enter" && e.ctrlKey) run(); });
checkHealth();

// URL 带 #demo 时自动填充示例并运行（演示/冒烟入口）
if (location.hash === "#demo") {
  input.value = ["寄两份合同文件，明天必须到", "一箱车厘子", "给小孩寄的奶粉",
    "充电宝2个", "陶瓷碗4个，轻拿轻放", "一箱东西"].join("\n");
  run();
}

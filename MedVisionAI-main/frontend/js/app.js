// app.js — shared helpers used across all pages
const API = ""; // same-origin (FastAPI serves both API and frontend)

function toast(msg) {
  let t = document.getElementById("toast");
  if (!t) {
    t = document.createElement("div");
    t.id = "toast";
    t.className = "toast";
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 2500);
}

async function apiGet(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function apiPostJSON(path, body) {
  const res = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function apiPostForm(path, formData) {
  const res = await fetch(API + path, { method: "POST", body: formData });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ---- connectivity indicator (this app works fully offline; this
// only reflects whether the browser currently has a network path,
// used for the optional "sync" demo) ----
function updateConnectivity() {
  const el = document.getElementById("connStatus");
  if (!el) return;
  if (navigator.onLine) {
    el.innerHTML = '<span class="dot green"></span> Online — sync available';
  } else {
    el.innerHTML = '<span class="dot amber"></span> Offline — AI screening still fully available';
  }
}
window.addEventListener("online", updateConnectivity);
window.addEventListener("offline", updateConnectivity);
document.addEventListener("DOMContentLoaded", updateConnectivity);

// ---- simple i18n ----
const I18N = {
  en: {
    tagline: "Early Eye Screening. Anywhere. Offline.",
    start: "Start Screening", doctor: "Doctor Login", staff: "Screening Staff Login",
    history: "Patient History",
  },
  ta: {
    tagline: "ஆரம்ப கண் பரிசோதனை. எங்கும். இணையம் இல்லாமலும்.",
    start: "பரிசோதனையைத் தொடங்கு", doctor: "மருத்துவர் உள்நுழைவு", staff: "பணியாளர் உள்நுழைவு",
    history: "நோயாளர் வரலாறு",
  },
  hi: {
    tagline: "प्रारंभिक नेत्र जांच। कहीं भी। बिना इंटरनेट।",
    start: "स्क्रीनिंग शुरू करें", doctor: "डॉक्टर लॉगिन", staff: "स्टाफ लॉगिन",
    history: "मरीज़ का इतिहास",
  },
};

function applyLang(lang) {
  localStorage.setItem("rr_lang", lang);
  const dict = I18N[lang] || I18N.en;
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.getAttribute("data-i18n");
    if (dict[key]) el.textContent = dict[key];
  });
}

document.addEventListener("DOMContentLoaded", () => {
  const saved = localStorage.getItem("rr_lang") || "en";
  const sel = document.getElementById("langSelect");
  if (sel) {
    sel.value = saved;
    sel.addEventListener("change", (e) => applyLang(e.target.value));
  }
  applyLang(saved);
});

function riskBadgeClass(risk) {
  if (!risk) return "recapture";
  if (risk === "urgent") return "urgent";
  if (risk === "priority") return "priority";
  if (risk === "routine") return "routine";
  return "recapture";
}

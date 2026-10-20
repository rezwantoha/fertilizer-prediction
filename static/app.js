"use strict";

const $ = (selector, parent = document) => parent.querySelector(selector);
const $$ = (selector, parent = document) => [...parent.querySelectorAll(selector)];
const translations = JSON.parse($("#app-copy").dataset.translations);
let language = document.documentElement.lang;
let displayUnit = "kg";
let latestPrediction = null;
let latestCalibration = null;
const formErrors = new Map();

function t(key, params = {}) {
  return (translations[language][key] || translations[language].unavailable)
    .replace(/\{(\w+)\}/g, (_, name) => String(params[name] ?? ""));
}
function format(value, digits = 2) {
  return new Intl.NumberFormat(language === "bn" ? "bn-BD" : "en-US",
    {maximumFractionDigits: digits}).format(Number(value));
}
function normalizeDigits(value) {
  return String(value).replace(/[০-৯]/g, digit => "০১২৩৪৫৬৭৮৯".indexOf(digit));
}
function inputDigits(value) {
  const ascii = normalizeDigits(value);
  return language === "bn" ? ascii.replace(/[0-9]/g, digit => "০১২৩৪৫৬৭৮৯"[Number(digit)]) : ascii;
}
function amountText(value) {
  const minimumDigits = displayUnit === "kg" ? 3 : 1;
  const digits = value > 0 && value < 1 ? Math.min(10, Math.max(minimumDigits, Math.ceil(-Math.log10(value)) + 1)) : minimumDigits;
  return format(value, digits);
}
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function setLanguage(nextLanguage) {
  if (!translations[nextLanguage]) return;
  language = nextLanguage;
  document.documentElement.lang = language;
  document.title = t("page_title");
  $('meta[name="description"]').content = t("description");
  $$("[data-i18n]").forEach(node => { node.textContent = t(node.dataset.i18n); });
  $$("[data-i18n-placeholder]").forEach(node => { node.placeholder = t(node.dataset.i18nPlaceholder); });
  $$("[data-i18n-aria]").forEach(node => { node.setAttribute("aria-label", t(node.dataset.i18nAria)); });
  $$("[data-crop]").forEach(node => { node.textContent = translations[language].crops[node.dataset.crop]; });
  $$("[data-number]").forEach(node => { node.textContent = format(node.dataset.number, 0); });
  $$("[data-numeric]").forEach(node => { node.value = inputDigits(node.value); });
  $$("[data-language]").forEach(link => {
    if (link.dataset.language === language) link.setAttribute("aria-current", "true");
    else link.removeAttribute("aria-current");
  });
  $$("[data-submit-label]").forEach(button => {
    button.textContent = t(button.dataset.busy === "true" ? "calculating" : button.dataset.submitLabel);
  });
  for (const [form, error] of formErrors) renderError(form, error);
  if (latestPrediction) renderPrediction();
  if (latestCalibration) renderCalibration();
  // Store only a language preference, never the user's soil/plot inputs.
  document.cookie = `calculator_language=${language};path=/;max-age=31536000;SameSite=Lax`;
  const url = new URL(window.location.href);
  url.searchParams.set("lang", language);
  history.replaceState(null, "", url);
}
$$("[data-language]").forEach(link => link.addEventListener("click", event => {
  event.preventDefault();
  setLanguage(link.dataset.language);
}));

function errorNode(form) {
  return form.id === "prediction-form" ? $("#prediction-error") : $("#calibration-error");
}
function clearError(form) {
  errorNode(form).hidden = true;
  formErrors.delete(form);
  $$("[aria-invalid]", form).forEach(field => field.removeAttribute("aria-invalid"));
}
function renderError(form, error) {
  const field = error.field && form.elements.namedItem(error.field);
  const fieldKey = field?.dataset.label || error.field || "";
  const params = {...error.params, field: t(fieldKey)};
  for (const key of ["min", "max"]) {
    if (params[key] !== undefined) params[key] = format(params[key], 8);
  }
  const node = errorNode(form);
  node.textContent = t(error.code, params);
  node.hidden = false;
  if (field) field.setAttribute("aria-invalid", "true");
}
function showError(form, error, focus = true) {
  formErrors.set(form, error);
  renderError(form, error);
  if (focus && error.field) form.elements.namedItem(error.field)?.focus();
}
function payloadFor(form) {
  const payload = Object.fromEntries(new FormData(form));
  if (form.id === "prediction-form" && !Object.hasOwn(translations.en.crops, payload.crop ?? "")) {
    throw {code: "invalid_crop", field: "crop"};
  }
  for (const field of $$("[data-numeric]", form)) {
    const text = normalizeDigits(field.value).trim();
    if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(text) || !Number.isFinite(Number(text))) {
      throw {code: "invalid_number", field: field.name};
    }
    const value = Number(text);
    const min = Number(field.dataset.min), max = Number(field.dataset.max);
    if (value < min || value > max) {
      throw {code: "invalid_range", field: field.name, params: {min, max}};
    }
    payload[field.name] = value;
  }
  return payload;
}

async function submitForm(form, endpoint, output, onSuccess) {
  const submit = $("[data-submit-label]", form);
  if (submit.disabled) return;
  clearError(form);
  let payload;
  try { payload = payloadFor(form); }
  catch (error) { showError(form, error); return; }
  const controls = $$("input, select, button", form).map(control => ({control, disabled: control.disabled}));
  controls.forEach(({control}) => { control.disabled = true; });
  submit.dataset.busy = "true";
  submit.textContent = t("calculating");
  output.setAttribute("aria-busy", "true");
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20000);
  let failure;
  try {
    const response = await fetch(endpoint, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload), signal: controller.signal
    });
    if (!response.headers.get("content-type")?.includes("application/json")) throw {code: "unavailable"};
    const data = await response.json();
    if (!response.ok) throw {code: data.code || "unavailable", field: data.field, params: data.params};
    onSuccess(data);
    output.querySelector("h2")?.focus({preventScroll: true});
    if (window.matchMedia("(max-width: 800px)").matches) {
      output.scrollIntoView({block: "start", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth"});
    }
  } catch (error) {
    failure = error.name === "AbortError" ? {code: "timeout"} :
      error instanceof TypeError ? {code: "connection_error"} :
      error.code ? error : {code: "unavailable"};
  } finally {
    clearTimeout(timeout);
    controls.forEach(({control, disabled}) => { control.disabled = disabled; });
    delete submit.dataset.busy;
    submit.textContent = t(submit.dataset.submitLabel);
    output.setAttribute("aria-busy", "false");
    if (failure) showError(form, failure);
  }
}

function renderPrediction() {
  const data = latestPrediction;
  $("#empty-result").hidden = true;
  $("#prediction-result").hidden = false;
  const {area, area_unit, nk, p} = data.inputs;
  let areaUnit = t(area_unit);
  if (language === "en" && area !== 1) areaUnit += "s";
  $("#result-context").textContent = t("plot_summary", {
    crop: translations[language].crops[data.crop], area: format(area, 6), unit: areaUnit
  });
  $("#result-inputs").textContent = t("inputs_summary", {nk: format(nk, 8), p: format(p, 8)});
  $("#result-warning").hidden = data.on_training_diagonal;
  $("#result-warning").textContent = t("off_diagonal");
  const container = $("#fertilizer-results");
  container.replaceChildren();
  data.fertilizers.forEach(fertilizer => {
    const row = element("article", "fertilizer-result");
    row.dataset.kind = fertilizer.id;
    const heading = element("h3");
    const symbol = element("span", "", {urea: "N", mop: "K", tsp: "P"}[fertilizer.id]);
    symbol.setAttribute("aria-hidden", "true");
    heading.append(symbol, document.createTextNode(t(fertilizer.id)));
    const amount = element("p", "fertilizer-amount",
      amountText(displayUnit === "g" ? fertilizer.total_g : fertilizer.total_kg));
    amount.append(element("span", "", t(displayUnit)));
    row.append(heading, amount, element("p", "fertilizer-rate",
      t("rate", {amount: format(fertilizer.rate_g_per_decimal, 1)})));
    container.append(row);
  });
}
function clearPrediction() {
  latestPrediction = null;
  $("#prediction-result").hidden = true;
  $("#empty-result").hidden = false;
}

const predictionForm = $("#prediction-form");
predictionForm.addEventListener("submit", event => {
  event.preventDefault();
  submitForm(predictionForm, "/api/predict", $(".output-panel"), data => {
    latestPrediction = data;
    renderPrediction();
  });
});
for (const eventName of ["input", "change", "reset"]) {
  predictionForm.addEventListener(eventName, () => { clearPrediction(); clearError(predictionForm); });
}
$("#sample-button").addEventListener("click", () => {
  if ($("#crop").disabled) return;
  $("#crop").value = "Boro Rice(BRI 29)";
  $("#nk").value = inputDigits("0.12");
  $("#p").value = inputDigits("12");
  $("#area").value = inputDigits("10");
  $("#area-unit").value = "decimal";
  clearPrediction();
  predictionForm.requestSubmit();
});
$$("[data-mass]").forEach(button => button.addEventListener("click", () => {
  displayUnit = button.dataset.mass;
  $$("[data-mass]").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
  if (latestPrediction) renderPrediction();
}));
$("#print-result").addEventListener("click", () => window.print());
$("#download-result").addEventListener("click", () => {
  if (!latestPrediction) return;
  const data = latestPrediction;
  const csvNumber = value => format(value, 8);
  const rows = [
    [t("csv_title")], [t("crop"), translations[language].crops[data.crop]],
    [t("nk"), csvNumber(data.inputs.nk)], [t("p"), csvNumber(data.inputs.p)],
    [t("area"), csvNumber(data.inputs.area), t(data.inputs.area_unit)],
    [t("csv_fertilizer"), t("csv_rate"), t("csv_total_g"), t("csv_total_kg")],
    ...data.fertilizers.map(item => [t(item.id), csvNumber(item.rate_g_per_decimal), csvNumber(item.total_g), csvNumber(item.total_kg)]),
    [t("csv_note"), t("result_note")],
    ...(!data.on_training_diagonal ? [[t("csv_note"), t("off_diagonal")]] : [])
  ];
  const csv = "\ufeff" + rows.map(row => row.map(value => `"${String(value).replaceAll('"', '""')}"`).join(",")).join("\r\n");
  const url = URL.createObjectURL(new Blob([csv], {type: "text/csv;charset=utf-8;"}));
  const link = element("a");
  link.href = url;
  link.download = language === "bn" ? "সারের-হিসাব.csv" : "fertilizer-estimate.csv";
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

function renderCalibration() {
  $("#calibration-result").hidden = false;
  const container = $("#calibration-readings");
  container.replaceChildren();
  latestCalibration.readings.forEach(reading => {
    const item = element("article", "calibration-reading");
    const header = element("header");
    const amount = element("strong", "", format(reading.corrected, reading.id === "n" ? 5 : reading.id === "k" ? 4 : 2));
    const unit = reading.id === "p" && language === "bn" ? "মি.গ্রা./কেজি" : reading.unit;
    amount.append(element("span", "", ` ${unit}`));
    header.append(element("h3", "", t(reading.id === "p" ? "sensor_p_name" : reading.id)), amount);
    item.append(header,
      element("p", "equation", `${format(reading.slope, 5)} × ${format(reading.raw, 5)} + ${format(reading.intercept, 5)}`),
      element("p", "quality", t(`${reading.id}_quality`)),
      element("p", "", t(`${reading.id}_note`)));
    container.append(item);
  });
}
const calibrationForm = $("#calibration-form");
calibrationForm.addEventListener("submit", event => {
  event.preventDefault();
  submitForm(calibrationForm, "/api/calibrate", $("#calibration-result"), data => {
    latestCalibration = data;
    renderCalibration();
  });
});
calibrationForm.addEventListener("input", () => {
  latestCalibration = null;
  $("#calibration-result").hidden = true;
  clearError(calibrationForm);
});

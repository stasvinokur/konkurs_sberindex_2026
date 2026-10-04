// Сценарий лендинга: оглавление, выбор месяца на графиках, выбор ряда, кнопки копирования.
// Страница читается и без него: все числа и выводы стоят в разметке, сценарий добавляет выбор.
(() => {
  'use strict';
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const MONTHS = ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь',
    'октябрь', 'ноябрь', 'декабрь'];
  const SHORT = ['янв', 'фев', 'мар', 'апр', 'май', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек'];
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const known = value => typeof value === 'number' && Number.isFinite(value);
  // Число как в тексте страницы: пробел между тысячами, запятая в дроби, минус — «−».
  const fmt = (value, digits = 0) => {
    if (!known(value)) return '—';
    const [whole, fraction] = Math.abs(value).toFixed(digits).split('.');
    const text = whole.replace(/\B(?=(\d{3})+(?!\d))/g, ' ') + (fraction ? ',' + fraction : '');
    return (value < 0 && /[1-9]/.test(text) ? '−' : '') + text;
  };
  const monthName = month => {
    const name = MONTHS[Number(month.slice(5, 7)) - 1];
    return name[0].toUpperCase() + name.slice(1) + ' ' + month.slice(0, 4);
  };
  const coordinate = value => String(Math.round(value * 10) / 10);

  // ── Оглавление: вкладки, сворачивание на узком экране, подсветка текущего раздела ──────────
  const side = $('.side');
  if (side) {
    const narrow = matchMedia('(max-width: 900px)');
    const toggle = $('.toc-toggle', side);
    const collapse = on => {
      side.toggleAttribute('data-collapsed', on);
      if (toggle) toggle.setAttribute('aria-expanded', String(!on));
    };
    collapse(true);
    if (toggle) toggle.addEventListener('click', () => collapse(!side.hasAttribute('data-collapsed')));
    const tabs = $$('[data-tab]', side);
    tabs.forEach(tab => tab.addEventListener('click', () => {
      tabs.forEach(other => other.setAttribute('aria-pressed', String(other === tab)));
      $$('[data-pane]', side).forEach(pane => { pane.hidden = pane.dataset.pane !== tab.dataset.tab; });
    }));
    side.addEventListener('click', event => {
      if (event.target.closest('a') && narrow.matches) collapse(true);
    });
    const links = $$('[data-pane="sections"] a[href^="#"]', side);
    const targets = links.map(link => document.getElementById(link.getAttribute('href').slice(1)));
    let pending = false;
    const mark = () => {
      pending = false;
      let current = 0;
      targets.forEach((target, i) => {
        if (target && target.getBoundingClientRect().top <= 140) current = i;
      });
      links.forEach((link, i) => {
        if (i === current) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    };
    addEventListener('scroll', () => {
      if (!pending) { pending = true; requestAnimationFrame(mark); }
    }, { passive: true });
    mark();
  }

  // ── Кнопки «скопировать команду» ───────────────────────────────────────────────────────────
  $$('[data-copy]').forEach(button => button.addEventListener('click', async () => {
    const code = $('code', button.closest('.cmd'));
    try {
      await navigator.clipboard.writeText(code.textContent);
    } catch {
      // Буфер обмена недоступен: выделяем команду, чтобы её можно было скопировать вручную.
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      return;
    }
    const label = button.getAttribute('aria-label');
    button.setAttribute('data-done', '');
    button.setAttribute('aria-label', 'Скопировано');
    setTimeout(() => {
      button.removeAttribute('data-done');
      button.setAttribute('aria-label', label);
    }, 1600);
  }));

  // ── Выбор месяца на графике: указателем, касанием и стрелками ──────────────────────────────
  // Сетка месяцев записана в атрибутах рисунка; `show` обновляет панель и возвращает подпись.
  const picker = (svg, count, show) => {
    const hit = $('[data-hit]', svg);
    const pick = $('[data-pick]', svg);
    const x0 = Number(svg.dataset.x0);
    const step = Number(svg.dataset.step);
    const width = svg.viewBox.baseVal.width;
    let index = Number(hit.getAttribute('aria-valuenow')) || 0;
    const select = next => {
      index = clamp(next, 0, count - 1);
      pick.setAttribute('x', coordinate(x0 + index * step - step / 2));
      hit.setAttribute('aria-valuenow', String(index));
      hit.setAttribute('aria-valuetext', show(index));
    };
    const under = event => {
      const box = svg.getBoundingClientRect();
      return Math.round(((event.clientX - box.left) / box.width * width - x0) / step);
    };
    hit.addEventListener('pointermove', event => select(under(event)));
    hit.addEventListener('pointerdown', event => select(under(event)));
    hit.addEventListener('keydown', event => {
      const move = { ArrowLeft: -1, ArrowRight: 1, ArrowDown: -1, ArrowUp: 1 }[event.key];
      if (move) select(index + move);
      else if (event.key === 'Home') select(0);
      else if (event.key === 'End') select(count - 1);
      else return;
      event.preventDefault();
    });
    select(index);
  };

  // ── Пример работы детектора: значения выбранного месяца ────────────────────────────────────
  $$('[data-demo]').forEach(root => {
    const data = JSON.parse($('script[type="application/json"]', root).textContent);
    const field = name => $(`[data-f="${name}"]`, root);
    picker($('svg.ch', root), data.months.length, i => {
      field('month').textContent = monthName(data.months[i]);
      field('y').textContent = fmt(data.y[i]) + ' ₽';
      field('forecast').textContent = known(data.forecast[i]) ? fmt(data.forecast[i]) + ' ₽' : '—';
      field('z').textContent = fmt(data.z[i], 1);
      field('statistic').textContent = fmt(data.statistic[i], 1);
      field('alarm').hidden = !data.alarms.includes(i);
      return monthName(data.months[i]);
    });
  });

  // ── Выбор ряда: регион и категория, график рисуется здесь же ───────────────────────────────
  const explorer = $('#explorer');
  if (explorer) {
    const data = JSON.parse($('script[type="application/json"]', explorer).textContent);
    const region = $('#ex-region');
    const category = $('#ex-category');
    const plot = $('#ex-plot');
    const field = name => $(`[data-f="${name}"]`, explorer);
    const [W, H, LEFT, TOP, INSET] = [760, 340, 54, 14, 20];
    const RIGHT = W - 12;
    const BOTTOM = H - 30;

    const options = (select, values) => {
      const keep = select.value;
      select.replaceChildren(...values.map(value => new Option(value, value)));
      if (values.includes(keep)) select.value = keep;
    };
    const categories = name => [...new Set(data.series.filter(s => s.region === name).map(s => s.category))]
      .sort((a, b) => a.localeCompare(b, 'ru'));
    // Деления с круглым шагом — то же правило, что у графиков, собранных при сборке страницы.
    const ticks = (low, high, count) => {
      if (!(high > low)) { const pad = Math.abs(low) * 0.1 || 1; low -= pad; high += pad; }
      const raw = (high - low) / count;
      const power = 10 ** Math.floor(Math.log10(raw));
      const error = raw / power;
      const step = power * (error >= Math.sqrt(50) ? 10 : error >= Math.sqrt(10) ? 5 : error >= Math.SQRT2 ? 2 : 1);
      const out = [];
      for (let i = Math.floor(low / step + 1e-9); i <= Math.ceil(high / step - 1e-9); i++) out.push(i * step);
      return out;
    };
    const path = (xs, ys) => {
      let out = '';
      let drawing = false;
      ys.forEach((y, i) => {
        if (y === null) { drawing = false; return; }
        out += (drawing ? 'L' : 'M') + coordinate(xs[i]) + ' ' + coordinate(y);
        drawing = true;
      });
      return out;
    };

    const draw = () => {
      const item = data.series.find(s => s.region === region.value && s.category === category.value);
      if (!item) return;
      const count = item.ds.length;
      const values = [...item.y, ...item.y_hat, ...item.q_lo, ...item.q_hi].filter(known);
      const marks = ticks(Math.min(...values), Math.max(...values), 6);
      const [low, high] = [marks[0], marks[marks.length - 1]];
      const thousands = Math.max(...values) >= 10000;
      const unit = thousands ? 'тыс. ₽' : '₽';
      const y = value => BOTTOM - (value - low) / (high - low) * (BOTTOM - TOP);
      const step = (RIGHT - LEFT - 2 * INSET) / Math.max(count - 1, 1);
      const xs = item.ds.map((_, i) => LEFT + INSET + i * step);
      const ys = key => item[key].map(value => (known(value) ? y(value) : null));
      const alarms = item.alarms.map(month => item.ds.indexOf(month)).filter(i => i >= 0);
      const whole = marks.every(mark => Number.isInteger(thousands ? mark / 1000 : mark));
      const parts = [];
      marks.forEach(mark => {
        const at = coordinate(y(mark));
        parts.push(`<line class="grid" x1="${LEFT}" x2="${RIGHT}" y1="${at}" y2="${at}"/>`);
        parts.push(`<text class="s11" x="${LEFT - 8}" y="${coordinate(y(mark) + 4)}" text-anchor="end">` +
          `${fmt(thousands ? mark / 1000 : mark, whole ? 0 : 1)}</text>`);
      });
      parts.push(`<line class="axis" x1="${LEFT}" x2="${RIGHT}" y1="${BOTTOM}" y2="${BOTTOM}"/>`);
      item.ds.forEach((month, i) => {
        const label = SHORT[Number(month.slice(5, 7)) - 1] + (i === 0 ? ' ' + month.slice(0, 4) : '');
        parts.push(`<text class="s11" x="${coordinate(xs[i])}" y="${BOTTOM + 18}" text-anchor="middle">${label}</text>`);
      });
      const fact = ys('y');
      const [lower, upper] = [ys('q_lo'), ys('q_hi')];
      (item.band || []).forEach(([start, stop]) => {
        if (stop - start < 2) return;
        const top = [];
        const bottom = [];
        for (let i = start; i < stop; i++) {
          top.push(coordinate(xs[i]) + ' ' + coordinate(upper[i]));
          bottom.unshift(coordinate(xs[i]) + ' ' + coordinate(lower[i]));
        }
        parts.push(`<path class="band" d="M${top.concat(bottom).join('L')}Z"/>`);
      });
      const selected = alarms.length ? alarms[0] : count - 1;
      parts.push(`<rect class="pick" data-pick x="0" y="${TOP}" width="${coordinate(step)}" height="${BOTTOM - TOP}"/>`);
      alarms.forEach(i => parts.push(`<line class="drop" x1="${coordinate(xs[i])}" x2="${coordinate(xs[i])}" ` +
        `y1="${coordinate(fact[i] ?? TOP)}" y2="${BOTTOM}"/>`));
      parts.push(`<path class="ln fc" d="${path(xs, ys('y_hat'))}"/>`);
      parts.push(`<path class="ln fact" d="${path(xs, fact)}"/>`);
      fact.forEach((value, i) => {
        if (value !== null) parts.push(`<circle class="pt" cx="${coordinate(xs[i])}" cy="${coordinate(value)}" r="3.5"/>`);
      });
      alarms.forEach(i => {
        if (fact[i] !== null) parts.push(`<circle class="ring" cx="${coordinate(xs[i])}" cy="${coordinate(fact[i])}" r="7.5"/>`);
        parts.push(`<path class="tri" d="M${coordinate(xs[i])} ${BOTTOM + 1}l-5 8h10z"/>`);
      });
      parts.push(`<rect class="hit" data-hit tabindex="0" role="slider" aria-label="Месяц ряда" aria-valuemin="0" ` +
        `aria-valuemax="${count - 1}" aria-valuenow="${selected}" x="${LEFT}" y="${TOP}" width="${RIGHT - LEFT}" height="${BOTTOM - TOP}"/>`);
      const name = item.mo || item.region;
      const place = name[0].toUpperCase() + name.slice(1);
      const title = `${place} · ${item.category.toLowerCase()}, ${unit}`;
      plot.innerHTML = `<svg class="ch" viewBox="0 0 ${W} ${H}" role="group" aria-label="${title.replace(/"/g, '&quot;')}: ` +
        `факт, прогноз и интервал по месяцам" data-x0="${coordinate(xs[0])}" data-step="${step.toFixed(3)}">${parts.join('')}</svg>`;
      field('title').textContent = title;
      field('place').textContent = place;
      field('alarms').textContent = String(alarms.length);
      picker($('svg', plot), count, i => {
        field('month').textContent = monthName(item.ds[i]);
        field('y').textContent = fmt(item.y[i]) + ' ₽';
        field('forecast').textContent = fmt(item.y_hat[i]) + ' ₽';
        field('interval').textContent = known(item.q_lo[i]) && known(item.q_hi[i])
          ? `${fmt(item.q_lo[i])}–${fmt(item.q_hi[i])} ₽` : '—';
        field('alarm').hidden = !alarms.includes(i);
        return monthName(item.ds[i]);
      });
    };
    region.addEventListener('change', () => { options(category, categories(region.value)); draw(); });
    category.addEventListener('change', draw);
    options(region, data.regions);
    options(category, categories(region.value));
    draw();
  }
})();

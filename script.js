(function () {
  'use strict';

  // ---------- Data ----------
  // rows: [name, value, ours?, reference?]
  var CHARTS = [
    { title: 'nuPlan closed-loop', metric: 'mean CLS ↑', min: 70, rows: [
      ['Log-Replay', 83.1, 0, 1], ['PDM-Closed', 84.5], ['Flow Planner', 82.2], ['Pi-DiMT', 83.4], ['Plan-R1', 85.4], ['PlannerRFT', 83.4], ['DriveRL', 93.0, 1], ['DriveRL-TTS', 93.6, 1] ] },
    { title: 'NAVSIMv1 navtest', metric: 'PDMS ↑', min: 80, rows: [
      ['Human Driver', 94.8, 0, 1], ['UniAD', 83.4], ['DriveVLA-W0', 90.2], ['iPad', 91.7], ['DriveFine', 91.8], ['DrivoR-Scale', 94.6], ['DriveZero', 94.8, 1], ['DriveZero-Scale', 95.3, 1] ] },
    { title: 'NAVSIMv2 navhard', metric: 'EPDMS ↑', min: 0, rows: [
      ['RAP', 39.6], ['ZTRS', 48.1], ['GigaPixel', 50.1], ['GuideFlow', 51.5], ['SimScale', 53.2], ['DrivoR-Scale', 54.6], ['DriveZero', 51.5, 1], ['DriveZero-Scale', 57.1, 1] ] },
    { title: 'HUGSIM closed-loop', metric: 'HD-Score ↑', min: 0, rows: [
      ['VAD', 13.4], ['LTF', 23.7], ['UniAD', 32.7], ['DrivoR-Scale', 38.1], ['GigaPixel', 38.5], ['DriveZero', 39.4, 1], ['DriveZero-Scale', 46.6, 1] ] }
  ];

  var HUGSIM = {
    hard: [
      { folder: 'nuscenes__scene-0064_hard_00', dataset: 'nuScenes', scene: 'Scene 0064', drivor: [8.9, 37.8, 23.6], dermes: [98.1, 100.0, 98.1] },
      { folder: 'nuscenes__scene-0071_hard_00', dataset: 'nuScenes', scene: 'Scene 0071', drivor: [45.6, 74.2, 61.4], dermes: [98.2, 100.0, 98.2] },
      { folder: 'nuscenes__scene-0166_hard_00', dataset: 'nuScenes', scene: 'Scene 0166', drivor: [1.7, 17.0, 9.9], dermes: [92.6, 100.0, 92.6] },
      { folder: 'waymo__166085257829_0_200_hard_00', dataset: 'Waymo', scene: 'Scene 166085257829', drivor: [6.2, 11.7, 52.8], dermes: [98.0, 100.0, 98.0] },
      { folder: 'waymo__398895700423_0_200_hard_00', dataset: 'Waymo', scene: 'Scene 398895700423', drivor: [16.4, 58.4, 28.1], dermes: [98.8, 100.0, 98.8] },
      { folder: 'waymo__881121006469_0_200_hard_00', dataset: 'Waymo', scene: 'Scene 881121006469', drivor: [50.7, 56.0, 90.5], dermes: [100.0, 100.0, 100.0] }
    ],
    medium: [
      { folder: 'kitti360__0000_4480_4680_medium_02', dataset: 'KITTI-360', scene: 'Sequence 0000 / 4480', drivor: [12.4, 24.6, 50.2], dermes: [73.1, 100.0, 73.1] },
      { folder: 'kitti360__0000_5980_6180_medium_01', dataset: 'KITTI-360', scene: 'Sequence 0000 / 5980', drivor: [8.9, 23.1, 38.4], dermes: [97.0, 100.0, 97.0] },
      { folder: 'nuscenes__scene-0064_medium_02', dataset: 'nuScenes', scene: 'Scene 0064', drivor: [31.8, 90.7, 35.1], dermes: [94.4, 100.0, 94.4] },
      { folder: 'pandaset__028_medium_02', dataset: 'PandaSet', scene: 'Scene 028', drivor: [44.7, 62.6, 71.4], dermes: [100.0, 100.0, 100.0] },
      { folder: 'waymo__131421903137_0_200_medium_02', dataset: 'Waymo', scene: 'Scene 131421903137', drivor: [14.9, 28.1, 53.1], dermes: [100.0, 100.0, 100.0] },
      { folder: 'waymo__150623512729_0_200_medium_00', dataset: 'Waymo', scene: 'Scene 150623512729', drivor: [30.8, 60.8, 50.7], dermes: [100.0, 100.0, 100.0] }
    ]
  };

  // Cases shown in the test-time search selector (files in assets/tts/)
  var TTS_CASES = ['01', '02', '04', '08'];

  var $ = function (id) { return document.getElementById(id); };
  var pad = function (n) { return String(n).padStart(2, '0'); };
  var f1 = function (v) { return v.toFixed(1); };

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  // ---------- Result charts ----------
  function renderCharts() {
    var host = $('charts');
    CHARTS.forEach(function (c) {
      var max = Math.max.apply(null, c.rows.map(function (r) { return r[1]; }));
      var box = el('div');
      var head = el('div', 'chart-head');
      head.appendChild(el('strong', null, c.title));
      head.appendChild(el('span', null, c.metric));
      box.appendChild(head);
      var rows = el('div', 'chart-rows');
      c.rows.forEach(function (r) {
        var name = r[0], v = r[1], ours = r[2], ref = r[3];
        var row = el('div', 'chart-row' + (ours ? ' ours' : ref ? ' ref' : ''));
        row.appendChild(el('span', 'name', name));
        var track = el('div', 'track');
        var bar = el('div', 'bar');
        bar.style.width = Math.max(2, ((v - c.min) / (max - c.min)) * 100).toFixed(1) + '%';
        track.appendChild(bar);
        row.appendChild(track);
        row.appendChild(el('span', 'value', f1(v)));
        rows.appendChild(row);
      });
      box.appendChild(rows);
      host.appendChild(box);
    });
  }

  // ---------- Tab helper ----------
  function makeTabs(host, items, cls, onSelect) {
    host.innerHTML = '';
    var buttons = items.map(function (label, i) {
      var b = el('button', 'tab ' + cls, label);
      b.type = 'button';
      b.addEventListener('click', function () { onSelect(i); });
      host.appendChild(b);
      return b;
    });
    return function setActive(i) {
      buttons.forEach(function (b, j) { b.classList.toggle('active', j === i); });
    };
  }

  // ---------- Test-time search selector ----------
  function initTTS() {
    var img = $('tts-img'), label = $('tts-label');
    var setActive = makeTabs($('tts-tabs'), TTS_CASES.map(function (_, i) { return String(i + 1); }), 'round', select);
    function select(i) {
      img.src = 'assets/tts/' + TTS_CASES[i] + '.gif';
      label.textContent = pad(i + 1);
      setActive(i);
    }
    select(0);
  }

  // ---------- HUGSIM comparison ----------
  function initHugsim() {
    var diff = 'hard', scene = 0;
    var setDiffActive, setSceneActive;

    function render() {
      var item = HUGSIM[diff][scene];
      var base = 'assets/hugsim/' + diff + '/' + item.folder + '/';
      $('hs-label').textContent = item.dataset + ' · ' + item.scene;
      var vd = $('hs-drivor'), vz = $('hs-dermes');
      vd.src = base + 'drivor.mp4';
      vz.src = base + 'dermes.mp4';
      vd.load(); vz.load();
      for (var k = 0; k < 3; k++) {
        $('hs-d' + k).textContent = f1(item.drivor[k]);
        $('hs-z' + k).textContent = f1(item.dermes[k]);
      }
      setDiffActive(diff === 'hard' ? 0 : 1);
      setSceneActive(scene);
    }

    setDiffActive = makeTabs($('diff-tabs'), ['Hard', 'Medium'], 'wide', function (i) {
      diff = i === 0 ? 'hard' : 'medium';
      scene = 0;
      render();
    });
    setSceneActive = makeTabs($('scene-tabs'), HUGSIM.hard.map(function (_, i) { return String(i + 1); }), 'round', function (i) {
      scene = i;
      render();
    });
    render();
  }

  // ---------- BibTeX copy ----------
  function initBib() {
    var btn = $('copy-bib'), text = $('bib-text').textContent;
    btn.addEventListener('click', function () {
      var done = function () {
        btn.textContent = 'Copied';
        setTimeout(function () { btn.textContent = 'Copy'; }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done);
      } else {
        var ta = document.createElement('textarea');
        ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand('copy'); } catch (e) {}
        document.body.removeChild(ta); done();
      }
    });
  }

  renderCharts();
  initTTS();
  initHugsim();
  initBib();
})();

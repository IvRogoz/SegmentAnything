(() => {
  const $ = id => document.getElementById(id);
  const file = $('fileInput'), drop = $('dropZone'), refs = new Map();
  let source = null, activeFile = null, imageAspect = 1, referenceNumber = 0;
  const scene = new THREE.Scene();
  const viewHeight = 3.2;
  const camera = new THREE.OrthographicCamera(-viewHeight * innerWidth / innerHeight / 2, viewHeight * innerWidth / innerHeight / 2, viewHeight / 2, -viewHeight / 2, .1, 100);
  const renderer = new THREE.WebGLRenderer({ canvas: $('canvas'), antialias: true });
  const world = new THREE.Group(); scene.add(world);
  camera.position.z = 4; renderer.setSize(innerWidth, innerHeight);
  scene.add(new THREE.AmbientLight(0xffffff, 1));
  const orbit = new THREE.OrbitControls(camera, renderer.domElement);
  const texture = src => new Promise((resolve, reject) => {
    new THREE.TextureLoader().load(src, t => resolve(t), undefined, reject);
  });
  const img = src => new Promise((resolve, reject) => {
    const image = new Image(); image.onload = () => resolve(image); image.onerror = reject; image.src = src;
  });
  function render() { requestAnimationFrame(render); orbit.update(); renderer.render(scene, camera); }
  render();
  function clearWorld() { while (world.children.length) world.remove(world.children[0]); refs.forEach(r => scene.remove(r.mesh)); refs.clear(); $('refLayerContainer').innerHTML = ''; referenceNumber = 0; }
  function plane(width, height, map, z = 0) {
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, height), new THREE.MeshBasicMaterial({ map, transparent: true }));
    mesh.position.z = z; return mesh;
  }
  async function depthMesh(imageUrl, depthUrl, width, height) {
    const [color, depth] = await Promise.all([texture(imageUrl), img(depthUrl)]);
    const divisions = 180, canvas = document.createElement('canvas');
    canvas.width = canvas.height = divisions + 1;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(depth, 0, 0, canvas.width, canvas.height);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    const geometry = new THREE.PlaneGeometry(width, height, divisions, divisions);
    const positions = geometry.attributes.position;
    for (let y = 0; y <= divisions; y++) for (let x = 0; x <= divisions; x++) {
      const vertex = y * (divisions + 1) + x, sample = pixels[vertex * 4] / 255;
      positions.setZ(vertex, -(1 - sample) * 1.8);
    }
    positions.needsUpdate = true; geometry.computeVertexNormals();
    return new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ map: color, side: THREE.DoubleSide }));
  }
  async function draw(data) {
    clearWorld(); source = data; imageAspect = data.shape[1] / data.shape[0];
    $('executionProvider').textContent = data.mode === 'depth' ? 'ZipDepth' : 'SAM Segmentation';
    $('meshResolution').textContent = `${data.shape[1]} × ${data.shape[0]}`;
    $('rangeX').textContent = `-${(imageAspect * 1.35).toFixed(2)} … ${(imageAspect * 1.35).toFixed(2)}`;
    $('rangeY').textContent = '-1.35 … 1.35';
    $('rangeZ').textContent = data.mode === 'depth' ? 'relative depth' : `${data.masks.length} layers`;
    $('showSamLayers').closest('label').style.display = data.mode === 'depth' ? 'none' : '';
    const width = 2.7, height = width / imageAspect;
    if (data.mode === 'depth') {
      world.add(await depthMesh(data.image, data.depth, width, height));
      return;
    }
    const background = plane(width, height, await texture(data.background), -1); background.userData.kind = 'background'; world.add(background);
    for (let i = 0; i < data.masks.length; i++) { const layer = plane(width, height, await texture(data.masks[i].object), i * .08); layer.userData.kind = 'sam'; world.add(layer); }
  }
  async function addReference(fileToAdd) {
    if (!fileToAdd || !source) return;
    const dataUrl = await new Promise((resolve, reject) => { const r = new FileReader(); r.onload = () => resolve(r.result); r.onerror = reject; r.readAsDataURL(fileToAdd); });
    const tex = await texture(dataUrl), width = 1.25, referenceAspect = tex.image.width / tex.image.height, height = width / referenceAspect;
    const mesh = plane(width, height, tex, .3); mesh.visible = $('showReferences').checked; scene.add(mesh);
    const id = ++referenceNumber, state = { mesh, x: 0, y: 0, z: .3, scale: 1, rx: 0, ry: 0, rz: 0 };
    refs.set(id, state);
    const card = document.createElement('div'); card.className = 'ref-layer-card';
    card.innerHTML = `<div class="ref-layer-header"><span>Layer ${id} <small>${fileToAdd.name}</small></span><button class="remove-layer-btn" type="button" title="Remove layer">×</button></div>
      <label class="chk layer-toggle"><input type="checkbox" class="layer-visible" checked> Show layer</label>
      <div class="layer-control"><label>Position X</label><input type="range" class="layer-x" min="-3" max="3" step=".01" value="0"><output class="layer-x-val">0.00</output></div>
      <div class="layer-control"><label>Position Y</label><input type="range" class="layer-y" min="-3" max="3" step=".01" value="0"><output class="layer-y-val">0.00</output></div>
      <div class="layer-control"><label>Depth Z</label><input type="range" class="layer-z" min="-3" max="3" step=".01" value=".3"><output class="layer-z-val">0.30</output></div>
      <div class="layer-control"><label>Scale</label><input type="range" class="layer-scale" min=".1" max="4" step=".01" value="1"><output class="layer-scale-val">1.00</output></div>
      <div class="layer-separator">Rotation</div>
      <div class="layer-control"><label>Rotate X</label><input type="range" class="layer-rx" min="-180" max="180" step="1" value="0"><output class="layer-rx-val">0°</output></div>
      <div class="layer-control"><label>Rotate Y</label><input type="range" class="layer-ry" min="-180" max="180" step="1" value="0"><output class="layer-ry-val">0°</output></div>
      <div class="layer-control"><label>Rotate Z</label><input type="range" class="layer-rz" min="-180" max="180" step="1" value="0"><output class="layer-rz-val">0°</output></div>`;
    const update = () => { mesh.position.set(state.x, state.y, state.z); mesh.scale.setScalar(state.scale); mesh.rotation.set(state.rx, state.ry, state.rz); };
    const bind = (key, selector, output, radians = false) => { const control = card.querySelector(selector), value = card.querySelector(output); control.oninput = () => { const raw = Number(control.value); state[key] = radians ? raw * Math.PI / 180 : raw; value.textContent = radians ? `${raw.toFixed(0)}°` : raw.toFixed(2); update(); }; };
    bind('x', '.layer-x', '.layer-x-val'); bind('y', '.layer-y', '.layer-y-val'); bind('z', '.layer-z', '.layer-z-val'); bind('scale', '.layer-scale', '.layer-scale-val'); bind('rx', '.layer-rx', '.layer-rx-val', true); bind('ry', '.layer-ry', '.layer-ry-val', true); bind('rz', '.layer-rz', '.layer-rz-val', true);
    card.querySelector('.layer-visible').onchange = e => { mesh.visible = e.target.checked && $('showReferences').checked; };
    card.querySelector('.remove-layer-btn').onclick = () => { scene.remove(mesh); tex.dispose(); refs.delete(id); card.remove(); };
    $('refLayerContainer').append(card);
  }
  async function upload(selected) {
    if (!selected) return;
    activeFile = selected;
    $('loadingOverlay').style.display = 'flex';
    const form = new FormData(); form.append('image', selected); form.append('mode', $('modelSelect').value); form.append('input_size', $('zipInputSize').value);
    try {
      const response = await fetch('/upload', { method: 'POST', body: form }); const data = await response.json();
      if (!response.ok) throw Error(data.error || 'Upload failed');
      await draw(data); drop.classList.add('hidden'); $('sidePanel').style.display = 'block';
    } catch (error) { alert(error.message); } finally { $('loadingOverlay').style.display = 'none'; }
  }
  $('modelSelect').onchange = () => { $('zipdepthControls').style.display = $('modelSelect').value === 'depth' ? 'block' : 'none'; if (activeFile) upload(activeFile); };
  $('modelSelect').onchange();
  $('zipInputSize').onchange = () => { if (activeFile && $('modelSelect').value === 'depth') upload(activeFile); };
  drop.onclick = () => file.click(); file.onchange = e => upload(e.target.files[0]);
  drop.ondragover = e => { e.preventDefault(); drop.classList.add('dragover'); };
  drop.ondragleave = () => drop.classList.remove('dragover');
  drop.ondrop = e => { e.preventDefault(); drop.classList.remove('dragover'); upload(e.dataTransfer.files[0]); };
  $('loadAnother').onclick = () => file.click();
  $('addRefLayer').onclick = () => $('refFileInput').click();
  $('refFileInput').onchange = e => addReference(e.target.files[0]);
  $('resetView').onclick = () => { camera.position.set(0, 0, 4); orbit.target.set(0, 0, 0); orbit.update(); };
  $('showSamLayers').onchange = e => world.children.filter(mesh => mesh.userData.kind === 'sam').forEach(mesh => mesh.visible = e.target.checked);
  $('showReferences').onchange = e => refs.forEach(ref => ref.mesh.visible = e.target.checked);
  addEventListener('resize', () => { const half = viewHeight / 2, aspect = innerWidth / innerHeight; camera.left = -half * aspect; camera.right = half * aspect; camera.top = half; camera.bottom = -half; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
})();

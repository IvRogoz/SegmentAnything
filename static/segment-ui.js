(() => {
  const $ = id => document.getElementById(id);
  const file = $('fileInput'), drop = $('dropZone'), refs = new Map();
  let source = null, activeFile = null, activeEdgeTamBox = null, imageAspect = 1, referenceNumber = 0, mainVideo = null;
  const browserDepthModels = {
    'depth-anything-v2-small-webgpu': { label: 'Depth Anything V2 Small', url: '/static/models/depth_anything_v2_vits.onnx', size: 518, order: 'rgb', mean: [123.675, 116.28, 103.53], std: [58.395, 57.12, 57.375], byteScale: true },
    'midas-v21-small-webgpu': { label: 'MiDaS v2.1 Small 256', url: '/static/models/midas_v21_small_256.onnx', size: 256, order: 'bgr', mean: [.485, .456, .406], std: [.229, .224, .225], byteScale: false }
  };
  const browserDepthSessions = new Map();
  const scene = new THREE.Scene();
  const viewHeight = 3.2;
  const camera = new THREE.OrthographicCamera(-viewHeight * innerWidth / innerHeight / 2, viewHeight * innerWidth / innerHeight / 2, viewHeight / 2, -viewHeight / 2, .1, 100);
  const renderer = new THREE.WebGLRenderer({ canvas: $('canvas'), antialias: true });
  const world = new THREE.Group(); scene.add(world);
  const raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2();
  let paintingLayer = null, isPainting = false;
  camera.position.z = 4; renderer.setSize(innerWidth, innerHeight);
  scene.add(new THREE.AmbientLight(0xffffff, 1));
  const orbit = new THREE.OrbitControls(camera, renderer.domElement); orbit.mouseButtons.RIGHT = THREE.MOUSE.ROTATE;
  const texture = src => new Promise((resolve, reject) => {
    new THREE.TextureLoader().load(src, t => resolve(t), undefined, reject);
  });
  const img = src => new Promise((resolve, reject) => {
    const image = new Image(); image.onload = () => resolve(image); image.onerror = reject; image.src = src;
  });
  const isBrowserDepthModel = () => Boolean(browserDepthModels[$('modelSelect').value]);
  const waitFor = (element, event) => new Promise((resolve, reject) => { const done = () => { element.removeEventListener(event, done); element.removeEventListener('error', fail); resolve(); }; const fail = () => { element.removeEventListener(event, done); reject(new Error('Cannot decode video')); }; element.addEventListener(event, done, { once: true }); element.addEventListener('error', fail, { once: true }); });
  async function selectEdgeTamBox(imageUrl) {
    const overlay = $('edgetamPromptOverlay'), stage = $('edgetamPromptStage'), promptImage = $('edgetamPromptImage'), promptBox = $('edgetamPromptBox'), track = $('edgetamPromptTrack'), cancel = $('edgetamPromptCancel');
    promptImage.src = imageUrl;
    if (!promptImage.complete || !promptImage.naturalWidth) await waitFor(promptImage, 'load');
    overlay.style.display = 'flex'; promptBox.style.display = 'none'; track.disabled = true;
    let dragging = false, start = null, selection = null;
    const point = event => { const bounds = stage.getBoundingClientRect(); return { x: Math.max(0, Math.min(bounds.width, event.clientX - bounds.left)), y: Math.max(0, Math.min(bounds.height, event.clientY - bounds.top)), bounds }; };
    const drawSelection = current => {
      const left = Math.min(start.x, current.x), top = Math.min(start.y, current.y), width = Math.abs(current.x - start.x), height = Math.abs(current.y - start.y);
      selection = { left, top, width, height };
      promptBox.style.display = 'block'; promptBox.style.left = `${left}px`; promptBox.style.top = `${top}px`; promptBox.style.width = `${width}px`; promptBox.style.height = `${height}px`;
      track.disabled = width < 4 || height < 4;
    };
    return new Promise(resolve => {
      const cleanup = result => { dragging = false; overlay.style.display = 'none'; promptBox.style.display = 'none'; stage.onpointerdown = stage.onpointermove = stage.onpointerup = stage.onpointercancel = null; track.onclick = cancel.onclick = null; resolve(result); };
      stage.onpointerdown = event => { if (event.button !== 0) return; event.preventDefault(); dragging = true; start = point(event); stage.setPointerCapture(event.pointerId); drawSelection(start); };
      stage.onpointermove = event => { if (dragging) drawSelection(point(event)); };
      stage.onpointerup = event => { if (!dragging || event.button !== 0) return; drawSelection(point(event)); dragging = false; stage.releasePointerCapture(event.pointerId); };
      stage.onpointercancel = () => { dragging = false; };
      cancel.onclick = () => cleanup(null);
      track.onclick = () => {
        if (!selection || track.disabled) return;
        const bounds = stage.getBoundingClientRect(), scaleX = promptImage.naturalWidth / bounds.width, scaleY = promptImage.naturalHeight / bounds.height;
        cleanup([selection.left * scaleX, selection.top * scaleY, (selection.left + selection.width) * scaleX, (selection.top + selection.height) * scaleY]);
      };
    });
  }
  async function browserDepthSession(modelId) {
    if (!navigator.gpu) throw Error('WebGPU is unavailable. This model has no fallback.');
    if (browserDepthSessions.has(modelId)) return browserDepthSessions.get(modelId);
    const config = browserDepthModels[modelId];
    let session;
    try { session = await ort.InferenceSession.create(config.url, { executionProviders: ['webgpu'] }); }
    catch (error) { throw Error(`${config.label} failed to load with WebGPU: ${error.message}`); }
    browserDepthSessions.set(modelId, session); return session;
  }
  async function inferBrowserDepth(imageSource, modelId, videoState = null, frameIndex = null) {
    const config = browserDepthModels[modelId], width = imageSource.videoWidth || imageSource.naturalWidth || imageSource.width, height = imageSource.videoHeight || imageSource.naturalHeight || imageSource.height;
    if (!width || !height) throw Error('No decoded image frame for browser depth inference.');
    const original = document.createElement('canvas'); original.width = width; original.height = height; original.getContext('2d').drawImage(imageSource, 0, 0, width, height);
    const input = document.createElement('canvas'); input.width = input.height = config.size; const inputContext = input.getContext('2d', { willReadFrequently: true }); inputContext.drawImage(original, 0, 0, config.size, config.size);
    const pixels = inputContext.getImageData(0, 0, config.size, config.size).data, tensorData = new Float32Array(3 * config.size * config.size), plane = config.size * config.size;
    for (let i = 0; i < plane; i++) {
      const pixel = i * 4, channels = config.order === 'bgr' ? [pixels[pixel + 2], pixels[pixel + 1], pixels[pixel]] : [pixels[pixel], pixels[pixel + 1], pixels[pixel + 2]];
      for (let channel = 0; channel < 3; channel++) { const value = config.byteScale ? channels[channel] : channels[channel] / 255; tensorData[channel * plane + i] = (value - config.mean[channel]) / config.std[channel]; }
    }
    const session = await browserDepthSession(modelId), inputName = session.inputNames[0], result = await session.run({ [inputName]: new ort.Tensor('float32', tensorData, [1, 3, config.size, config.size]) }), output = result[session.outputNames[0]], values = output.data;
    let minimum = Infinity, maximum = -Infinity;
    for (const value of values) { if (Number.isFinite(value)) { minimum = Math.min(minimum, value); maximum = Math.max(maximum, value); } }
    if (!Number.isFinite(minimum) || maximum <= minimum) throw Error(`${config.label} returned invalid depth.`);
    const depthSmall = document.createElement('canvas'); depthSmall.width = depthSmall.height = config.size; const depthPixels = depthSmall.getContext('2d').createImageData(config.size, config.size);
    const useTemporalSmoothing = modelId === 'midas-v21-small-webgpu' && videoState && frameIndex !== null;
    const smoothing = Number($('midasTemporalSmoothing').value);
    const canBlend = useTemporalSmoothing && videoState.previousDepth && videoState.previousDepthFrame === frameIndex - 1;
    const smoothed = useTemporalSmoothing ? new Uint8ClampedArray(plane) : null;
    for (let i = 0; i < plane; i++) {
      const rawNormalized = Math.max(0, Math.min(255, Math.round((values[i] - minimum) / (maximum - minimum) * 255)));
      const normalized = modelId === 'midas-v21-small-webgpu' ? 255 - rawNormalized : rawNormalized;
      const value = canBlend ? Math.round(videoState.previousDepth[i] * smoothing + normalized * (1 - smoothing)) : normalized;
      if (smoothed) smoothed[i] = value;
      const pixel = i * 4; depthPixels.data[pixel] = depthPixels.data[pixel + 1] = depthPixels.data[pixel + 2] = value; depthPixels.data[pixel + 3] = 255;
    }
    if (smoothed) { videoState.previousDepth = smoothed; videoState.previousDepthFrame = frameIndex; }
    depthSmall.getContext('2d').putImageData(depthPixels, 0, 0);
    const depth = document.createElement('canvas'); depth.width = width; depth.height = height; depth.getContext('2d').drawImage(depthSmall, 0, 0, width, height);
    const imageDataUrl = original.toDataURL('image/png');
    return { success: true, mode: 'depth', image_id: `webgpu:${modelId}`, image: imageDataUrl, background: imageDataUrl, masks: [], depth: depth.toDataURL('image/png'), shape: [height, width] };
  }
  function render() { requestAnimationFrame(render); orbit.update(); renderer.render(scene, camera); }
  render();
  function clearBaseWorld() { while (world.children.length) world.remove(world.children[0]); }
  function clearWorld() { clearBaseWorld(); refs.forEach(r => { scene.remove(r.mesh); scene.remove(r.overlayMesh); scene.remove(r.previewMesh); if (r.videoUrl) URL.revokeObjectURL(r.videoUrl); }); refs.clear(); paintingLayer = null; $('brushOutline').style.display = 'none'; $('refLayerContainer').innerHTML = ''; referenceNumber = 0; }
  const isVideoFile = selected => Boolean(selected && (selected.type || '').startsWith('video/'));
  async function jsonResponse(response) {
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('application/json')) return response.json();
    const text = await response.text();
    return { error: response.status === 413 ? 'Video upload is too large. Choose Local Video to use the original file without uploading it.' : `Server returned ${response.status}: ${text.slice(0, 160)}` };
  }
  let progressPollId = 0;
  const createJobId = () => crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  function startRealProgress(jobId) {
    const pollId = ++progressPollId;
    $('sidePanel').style.display = 'block';
    $('loadingProgress').style.display = 'block';
    $('loadingBar').style.width = '0%';
    $('loadingText').textContent = 'Starting EdgeTAM — 0%';
    const poll = async () => {
      while (pollId === progressPollId) {
        try {
          const response = await fetch(`/progress/${encodeURIComponent(jobId)}`, { cache: 'no-store' });
          if (response.ok) {
            const status = await response.json();
            const percent = Math.max(0, Math.min(100, Number(status.percent) || 0));
            $('loadingBar').style.width = `${percent}%`;
            $('loadingText').textContent = `${status.label} — ${Math.round(percent)}%`;
          }
        } catch (_) {}
        await new Promise(resolve => setTimeout(resolve, 150));
      }
    };
    poll();
    return success => {
      if (pollId !== progressPollId) return;
      progressPollId++;
      if (success) {
        $('loadingBar').style.width = '100%';
        $('loadingText').textContent = 'EdgeTAM complete — 100%';
      }
      $('loadingProgress').style.display = 'none';
      $('loadingText').textContent = 'EdgeTAM progress';
    };
  }
  function updateVideoControls() {
    const controls = $('videoControls');
    if (!mainVideo) { controls.style.display = 'none'; return; }
    controls.style.display = 'block';
    $('videoTimeline').max = String(mainVideo.frameCount - 1); $('videoTimeline').value = String(mainVideo.frame);
    $('videoFrameLabel').textContent = `Frame ${mainVideo.frame + 1} / ${mainVideo.frameCount}`;
    $('videoPlayStop').textContent = mainVideo.playing ? 'Stop' : 'Play';
  }
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
  async function draw(data, preserveLayers = false) {
    if (!preserveLayers) clearWorld(); source = data; imageAspect = data.shape[1] / data.shape[0];
    $('executionProvider').textContent = data.mode === 'depth' ? 'ZipDepth' : data.mode === 'edgetam' ? 'EdgeTAM' : 'SAM Segmentation';
    $('meshResolution').textContent = `${data.shape[1]} × ${data.shape[0]}`;
    $('rangeX').textContent = `-${(imageAspect * 1.35).toFixed(2)} … ${(imageAspect * 1.35).toFixed(2)}`;
    $('rangeY').textContent = '-1.35 … 1.35';
    $('rangeZ').textContent = data.mode === 'depth' ? 'relative depth' : `${data.masks.length} layers`;
    $('showSamLayers').closest('label').style.display = data.mode === 'depth' ? 'none' : '';
    const width = 2.7, height = width / imageAspect;
    if (data.mode === 'depth') {
      const nextMesh = await depthMesh(data.image, data.depth, width, height);
      if (preserveLayers) clearBaseWorld();
      world.add(nextMesh);
      return;
    }
    const [backgroundTexture, ...layerTextures] = await Promise.all([texture(data.background), ...data.masks.map(mask => texture(mask.object))]);
    if (preserveLayers) clearBaseWorld();
    const background = plane(width, height, backgroundTexture, -1); background.userData.kind = 'background'; world.add(background);
    layerTextures.forEach((layerTexture, index) => { const layer = plane(width, height, layerTexture, index * .08); layer.userData.kind = 'sam'; world.add(layer); });
  }
  async function addReference(fileToAdd) {
    if (!fileToAdd || !source) return;
    const layerIsVideo = isVideoFile(fileToAdd);
    let image, tex, video = null, videoUrl = null;
    if (layerIsVideo) {
      videoUrl = URL.createObjectURL(fileToAdd); video = document.createElement('video'); video.src = videoUrl; video.muted = true; video.playsInline = true; video.preload = 'auto';
      await new Promise((resolve, reject) => { video.onloadeddata = resolve; video.onerror = () => reject(new Error('Cannot decode layer video')); });
      image = video; tex = new THREE.VideoTexture(video);
    } else {
      const dataUrl = await new Promise((resolve, reject) => { const r = new FileReader(); r.onload = () => resolve(r.result); r.onerror = reject; r.readAsDataURL(fileToAdd); });
      [image, tex] = await Promise.all([img(dataUrl), texture(dataUrl)]);
    }
    const width = 1.25, referenceAspect = (image.videoWidth || image.naturalWidth || tex.image.width) / (image.videoHeight || image.naturalHeight || tex.image.height), height = width / referenceAspect;
    const mesh = plane(width, height, tex, .3); mesh.visible = $('showReferences').checked; scene.add(mesh);
    const maskCanvas = document.createElement('canvas'), overlayCanvas = document.createElement('canvas');
    const maskWidth = Math.min(image.videoWidth || image.naturalWidth || image.width, 1024), maskHeight = Math.round(maskWidth / referenceAspect);
    maskCanvas.width = overlayCanvas.width = maskWidth; maskCanvas.height = overlayCanvas.height = maskHeight;
    const maskContext = maskCanvas.getContext('2d'), overlayContext = overlayCanvas.getContext('2d');
    const overlayTexture = new THREE.CanvasTexture(overlayCanvas), overlayMesh = plane(width, height, overlayTexture, .3), previewTexture = new THREE.CanvasTexture(maskCanvas), previewMesh = plane(width, height, previewTexture, .3);
    overlayMesh.material.depthTest = false; overlayMesh.material.depthWrite = false; overlayMesh.material.alphaTest = 0.5; overlayMesh.renderOrder = 100; overlayMesh.visible = mesh.visible; scene.add(overlayMesh);
    previewMesh.material.color.set(0x3c9dff); previewMesh.material.opacity = .35; previewMesh.material.depthTest = false; previewMesh.material.depthWrite = false; previewMesh.renderOrder = 101; previewMesh.visible = mesh.visible; scene.add(previewMesh);
    const id = ++referenceNumber, state = { mesh, overlayMesh, previewMesh, tex, overlayTexture, previewTexture, image, maskCanvas, maskContext, overlayCanvas, overlayContext, x: 0, y: 0, z: .3, scale: 1, rx: 0, ry: 0, rz: 0, brush: 80, visible: true, showMask: true, brushMode: 'paint', video, videoUrl, startFrame: 0, endFrame: Math.max(0, (mainVideo?.frameCount || 1) - 1) };
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
    card.insertAdjacentHTML('beforeend', `<div class="layer-separator">Depth mask</div><div class="mask-actions"><button type="button" class="mask-tool paint-mask">Paint mask</button><button type="button" class="mask-tool erase-mask">Erase mask</button><button type="button" class="clear-mask">Clear</button></div><label class="chk layer-toggle"><input type="checkbox" class="show-mask" checked> Show mask preview</label><div class="layer-control"><label>Brush radius</label><input type="range" class="brush-radius" min="4" max="250" step="1" value="80"><output class="brush-radius-val">80</output></div>`);
    if (layerIsVideo) card.insertAdjacentHTML('beforeend', `<div class="layer-separator">Video sync</div><div class="video-sync-note">Main video frame range</div><div class="layer-control"><label>Start frame</label><input type="range" class="video-start" min="0" max="${state.endFrame}" step="1" value="0"><output class="video-start-val">0</output></div><div class="layer-control"><label>End frame</label><input type="range" class="video-end" min="0" max="${state.endFrame}" step="1" value="${state.endFrame}"><output class="video-end-val">${state.endFrame}</output></div>`);
    const update = () => { mesh.position.set(state.x, state.y, state.z); mesh.scale.setScalar(state.scale); mesh.rotation.set(state.rx, state.ry, state.rz); [overlayMesh, previewMesh].forEach(layer => { layer.position.copy(mesh.position); layer.scale.copy(mesh.scale); layer.rotation.copy(mesh.rotation); }); };
    const bind = (key, selector, output, radians = false) => { const control = card.querySelector(selector), value = card.querySelector(output); control.oninput = () => { const raw = Number(control.value); state[key] = radians ? raw * Math.PI / 180 : raw; value.textContent = radians ? `${raw.toFixed(0)}°` : raw.toFixed(2); update(); }; };
    bind('x', '.layer-x', '.layer-x-val'); bind('y', '.layer-y', '.layer-y-val'); bind('z', '.layer-z', '.layer-z-val'); bind('scale', '.layer-scale', '.layer-scale-val'); bind('rx', '.layer-rx', '.layer-rx-val', true); bind('ry', '.layer-ry', '.layer-ry-val', true); bind('rz', '.layer-rz', '.layer-rz-val', true);
    const updateVisibility = () => { const shown = state.visible && $('showReferences').checked; mesh.visible = overlayMesh.visible = shown; previewMesh.visible = shown && state.showMask; };
    card.querySelector('.layer-visible').onchange = e => { state.visible = e.target.checked; updateVisibility(); };
    card.querySelector('.show-mask').onchange = e => { state.showMask = e.target.checked; updateVisibility(); };
    const selectBrush = (mode, button) => { paintingLayer = state; state.brushMode = mode; document.querySelectorAll('.mask-tool').forEach(control => control.classList.remove('active')); button.classList.add('active'); renderer.domElement.style.cursor = 'crosshair'; };
    card.querySelector('.paint-mask').onclick = e => selectBrush('paint', e.currentTarget);
    card.querySelector('.erase-mask').onclick = e => selectBrush('erase', e.currentTarget);
    card.querySelector('.clear-mask').onclick = () => { maskContext.clearRect(0, 0, maskWidth, maskHeight); redrawMask(state); };
    const brush = card.querySelector('.brush-radius'), brushValue = card.querySelector('.brush-radius-val'); brush.oninput = () => { state.brush = Number(brush.value); brushValue.textContent = String(state.brush); };
    if (video) {
      video.addEventListener('seeked', () => redrawMask(state));
      const start = card.querySelector('.video-start'), end = card.querySelector('.video-end'), startValue = card.querySelector('.video-start-val'), endValue = card.querySelector('.video-end-val');
      start.oninput = () => { state.startFrame = Math.min(Number(start.value), state.endFrame); start.value = String(state.startFrame); startValue.textContent = String(state.startFrame); syncLayerVideos(mainVideo?.frame ?? 0); };
      end.oninput = () => { state.endFrame = Math.max(Number(end.value), state.startFrame); end.value = String(state.endFrame); endValue.textContent = String(state.endFrame); syncLayerVideos(mainVideo?.frame ?? 0); };
    }
    card.querySelector('.remove-layer-btn').onclick = () => { if (paintingLayer === state) { paintingLayer = null; renderer.domElement.style.cursor = ''; $('brushOutline').style.display = 'none'; } scene.remove(mesh); scene.remove(overlayMesh); scene.remove(previewMesh); tex.dispose(); overlayTexture.dispose(); previewTexture.dispose(); if (videoUrl) URL.revokeObjectURL(videoUrl); refs.delete(id); card.remove(); };
    $('refLayerContainer').append(card);
  }
  function redrawMask(state) {
    state.overlayContext.clearRect(0, 0, state.overlayCanvas.width, state.overlayCanvas.height);
    state.overlayContext.drawImage(state.image, 0, 0, state.overlayCanvas.width, state.overlayCanvas.height);
    state.overlayContext.globalCompositeOperation = 'destination-in';
    state.overlayContext.drawImage(state.maskCanvas, 0, 0);
    state.overlayContext.globalCompositeOperation = 'source-over';
    state.overlayTexture.needsUpdate = true;
    state.previewTexture.needsUpdate = true;
  }
  function syncLayerVideos(frame) {
    if (!mainVideo) return;
    refs.forEach(layer => {
      if (!layer.video) return;
      const inRange = frame >= layer.startFrame && frame <= layer.endFrame;
      layer.mesh.visible = layer.overlayMesh.visible = layer.visible && $('showReferences').checked && inRange;
      layer.previewMesh.visible = layer.visible && $('showReferences').checked && layer.showMask && inRange;
      if (!inRange) { layer.video.pause(); return; }
      const time = Math.min(layer.video.duration || 0, Math.max(0, (frame - layer.startFrame) / mainVideo.fps));
      if (Math.abs(layer.video.currentTime - time) > .001) layer.video.currentTime = time;
    });
  }
  function paintDepthMask(event, state) {
    if (!state) return;
    const bounds = renderer.domElement.getBoundingClientRect();
    pointer.set(((event.clientX - bounds.left) / bounds.width) * 2 - 1, -((event.clientY - bounds.top) / bounds.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObject(state.mesh, false)[0];
    if (!hit || !hit.uv) return;
    const context = state.maskContext, x = hit.uv.x * state.maskCanvas.width, y = (1 - hit.uv.y) * state.maskCanvas.height;
    context.save(); context.globalCompositeOperation = state.brushMode === 'erase' ? 'destination-out' : 'source-over'; context.fillStyle = '#fff'; context.beginPath(); context.arc(x, y, state.brush, 0, Math.PI * 2); context.fill(); context.restore();
    redrawMask(state);
  }
  function updateBrushOutline(event) {
    const outline = $('brushOutline');
    if (!paintingLayer) { outline.style.display = 'none'; return; }
    const bounds = renderer.domElement.getBoundingClientRect();
    pointer.set(((event.clientX - bounds.left) / bounds.width) * 2 - 1, -((event.clientY - bounds.top) / bounds.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObject(paintingLayer.mesh, false)[0];
    if (!hit) { outline.style.display = 'none'; return; }
    const localCenter = paintingLayer.mesh.worldToLocal(hit.point.clone());
    const dimensions = paintingLayer.mesh.geometry.parameters;
    const localRadiusX = paintingLayer.brush / paintingLayer.maskCanvas.width * dimensions.width;
    const localRadiusY = paintingLayer.brush / paintingLayer.maskCanvas.height * dimensions.height;
    const project = point => point.applyMatrix4(paintingLayer.mesh.matrixWorld).project(camera);
    const center = project(localCenter.clone());
    const edgeX = project(localCenter.clone().add(new THREE.Vector3(localRadiusX, 0, 0)));
    const edgeY = project(localCenter.clone().add(new THREE.Vector3(0, localRadiusY, 0)));
    const radiusX = Math.abs(edgeX.x - center.x) * bounds.width / 2;
    const radiusY = Math.abs(edgeY.y - center.y) * bounds.height / 2;
    outline.style.display = 'block'; outline.style.width = `${radiusX * 2}px`; outline.style.height = `${radiusY * 2}px`; outline.style.left = `${event.clientX - radiusX}px`; outline.style.top = `${event.clientY - radiusY}px`;
  }
  renderer.domElement.addEventListener('pointerdown', event => { if (!paintingLayer || event.button !== 0) return; isPainting = true; orbit.enabled = false; renderer.domElement.setPointerCapture(event.pointerId); updateBrushOutline(event); paintDepthMask(event, paintingLayer); });
  renderer.domElement.addEventListener('pointermove', event => { updateBrushOutline(event); if (isPainting) paintDepthMask(event, paintingLayer); });
  renderer.domElement.addEventListener('pointerup', event => { if (!isPainting || event.button !== 0) return; isPainting = false; orbit.enabled = true; renderer.domElement.releasePointerCapture(event.pointerId); });
  async function processMainVideoFrame(frame) {
    if (!mainVideo || mainVideo.processing || frame < 0 || frame >= mainVideo.frameCount) return;
    if ($('modelSelect').value === 'edgetam' && !mainVideo.edgeTamBox) {
      const promptResponse = await fetch(`/video/${mainVideo.id}/frame/0`);
      if (!promptResponse.ok) { const error = await promptResponse.json(); alert(error.error || 'Cannot decode video frame 1'); return; }
      const promptUrl = URL.createObjectURL(await promptResponse.blob());
      try { mainVideo.edgeTamBox = await selectEdgeTamBox(promptUrl); }
      finally { URL.revokeObjectURL(promptUrl); }
      if (!mainVideo.edgeTamBox) return;
    }
    mainVideo.processing = true; mainVideo.frame = frame; updateVideoControls();
    const useRealProgress = $('modelSelect').value === 'edgetam';
    const jobId = useRealProgress ? createJobId() : null;
    const stopProgress = useRealProgress ? startRealProgress(jobId) : null;
    let completed = false;
    try {
      let data;
      if (isBrowserDepthModel()) {
        const frameResponse = await fetch(`/video/${mainVideo.id}/frame/${frame}`);
        if (!frameResponse.ok) { const error = await frameResponse.json(); throw Error(error.error || 'Cannot decode video frame'); }
        const frameUrl = URL.createObjectURL(await frameResponse.blob());
        try { data = await inferBrowserDepth(await img(frameUrl), $('modelSelect').value, mainVideo, frame); }
        finally { URL.revokeObjectURL(frameUrl); }
      } else {
        const form = new FormData(); form.append('mode', $('modelSelect').value); form.append('input_size', $('zipInputSize').value); form.append('inpainting', String($('inpaintingToggle').checked)); if (jobId) form.append('job_id', jobId); if (mainVideo.edgeTamBox) form.append('box', JSON.stringify(mainVideo.edgeTamBox));
        const response = await fetch(`/video/${mainVideo.id}/infer/${frame}`, { method: 'POST', body: form }); data = await jsonResponse(response);
        if (!response.ok) throw Error(data.error || 'Frame processing failed');
      }
      await draw(data, true); syncLayerVideos(frame); completed = true;
    } catch (error) { mainVideo.playing = false; alert(error.message); }
    finally {
      if (stopProgress) stopProgress(completed);
      if (!mainVideo) return;
      mainVideo.processing = false; updateVideoControls();
      if (mainVideo.playing && frame + 1 < mainVideo.frameCount) processMainVideoFrame(frame + 1);
      else if (frame + 1 >= mainVideo.frameCount) { mainVideo.playing = false; updateVideoControls(); }
    }
  }
  async function loadMainVideo(selected) {
    activeFile = selected; if (mainVideo?.browserVideoUrl) URL.revokeObjectURL(mainVideo.browserVideoUrl); mainVideo = null; clearWorld(); $('loadingOverlay').style.display = 'flex';
    try {
      const form = new FormData(); form.append('video', selected);
      const response = await fetch('/video', { method: 'POST', body: form }); const data = await jsonResponse(response);
      if (!response.ok) throw Error(data.error || 'Video upload failed');
      mainVideo = { id: data.video_id, frameCount: data.frame_count, fps: data.fps, frame: 0, playing: false, processing: false, name: selected.name, browserVideo: null, browserVideoUrl: null, edgeTamBox: null };
      $('executionProvider').textContent = 'Video pending'; $('meshResolution').textContent = `${data.width} Ã— ${data.height}`; updateVideoControls(); drop.classList.add('hidden'); $('sidePanel').style.display = 'block';
      if ($('modelSelect').value === 'edgetam') $('loadingOverlay').style.display = 'none';
      await processMainVideoFrame(0);
    } catch (error) { alert(error.message); mainVideo = null; updateVideoControls(); }
    finally { $('loadingOverlay').style.display = 'none'; }
  }
  async function upload(selected) {
    if (!selected) return;
    if (isVideoFile(selected)) { alert('Choose Local Video. Main videos are read from their original local path and are not uploaded.'); return; }
    if (activeFile !== selected) activeEdgeTamBox = null;
    activeFile = selected;
    mainVideo = null; updateVideoControls();
    if ($('modelSelect').value === 'edgetam' && !activeEdgeTamBox) {
      const promptUrl = URL.createObjectURL(selected);
      try { activeEdgeTamBox = await selectEdgeTamBox(promptUrl); }
      finally { URL.revokeObjectURL(promptUrl); }
      if (!activeEdgeTamBox) return;
    }
    const useRealProgress = $('modelSelect').value === 'edgetam';
    $('loadingOverlay').style.display = useRealProgress ? 'none' : 'flex';
    const jobId = useRealProgress ? createJobId() : null;
    const stopProgress = useRealProgress ? startRealProgress(jobId) : null;
    let completed = false;
    const form = new FormData(); form.append('image', selected); form.append('mode', $('modelSelect').value); form.append('input_size', $('zipInputSize').value); form.append('inpainting', String($('inpaintingToggle').checked)); if (jobId) form.append('job_id', jobId); if (activeEdgeTamBox) form.append('box', JSON.stringify(activeEdgeTamBox));
    try {
      let data;
      if (isBrowserDepthModel()) { const fileUrl = URL.createObjectURL(selected); try { data = await inferBrowserDepth(await img(fileUrl), $('modelSelect').value); } finally { URL.revokeObjectURL(fileUrl); } }
      else { const response = await fetch('/upload', { method: 'POST', body: form }); data = await jsonResponse(response); if (!response.ok) throw Error(data.error || 'Upload failed'); }
      await draw(data); drop.classList.add('hidden'); $('sidePanel').style.display = 'block'; completed = true;
    } catch (error) { alert(error.message); } finally { if (stopProgress) stopProgress(completed); $('loadingOverlay').style.display = 'none'; }
  }
  async function chooseLocalVideo() {
    $('loadingOverlay').style.display = 'flex';
    try {
      const response = await fetch('/video/select-local', { method: 'POST' });
      const data = await jsonResponse(response);
      if (!response.ok) throw Error(data.error || 'Video selection failed');
      activeFile = null;
      if (mainVideo?.browserVideoUrl) URL.revokeObjectURL(mainVideo.browserVideoUrl);
      mainVideo = { id: data.video_id, frameCount: data.frame_count, fps: data.fps, frame: 0, playing: false, processing: false, name: 'Local video', browserVideo: null, browserVideoUrl: null, edgeTamBox: null };
      clearWorld(); $('executionProvider').textContent = 'Video pending'; $('meshResolution').textContent = `${data.width} Ã— ${data.height}`; updateVideoControls(); drop.classList.add('hidden'); $('sidePanel').style.display = 'block';
      if ($('modelSelect').value === 'edgetam') $('loadingOverlay').style.display = 'none';
      await processMainVideoFrame(0);
    } catch (error) { alert(error.message); mainVideo = null; updateVideoControls(); }
    finally { $('loadingOverlay').style.display = 'none'; }
  }
  let previousModel = $('modelSelect').value;
  $('modelSelect').onchange = () => { const model = $('modelSelect').value; if (model === 'edgetam' && previousModel !== 'edgetam') { activeEdgeTamBox = null; if (mainVideo) mainVideo.edgeTamBox = null; } previousModel = model; $('zipdepthControls').style.display = model === 'depth' ? 'block' : 'none'; $('midasTemporalControls').style.display = model === 'midas-v21-small-webgpu' ? 'block' : 'none'; $('inpaintingControls').style.display = model === 'segmentation' || model === 'edgetam' ? 'block' : 'none'; if (mainVideo && !mainVideo.processing) processMainVideoFrame(mainVideo.frame); else if (activeFile) upload(activeFile); };
  $('modelSelect').onchange();
  $('zipInputSize').onchange = () => { if ($('modelSelect').value !== 'depth') return; if (mainVideo && !mainVideo.processing) processMainVideoFrame(mainVideo.frame); else if (activeFile) upload(activeFile); };
  $('inpaintingToggle').onchange = () => { if (mainVideo && !mainVideo.processing) processMainVideoFrame(mainVideo.frame); else if (activeFile) upload(activeFile); };
  drop.onclick = () => file.click(); file.onchange = e => upload(e.target.files[0]);
  drop.ondragover = e => { e.preventDefault(); drop.classList.add('dragover'); };
  drop.ondragleave = () => drop.classList.remove('dragover');
  drop.ondrop = e => { e.preventDefault(); drop.classList.remove('dragover'); upload(e.dataTransfer.files[0]); };
  $('loadAnother').onclick = () => file.click();
  $('chooseLocalVideo').onclick = e => { e.stopPropagation(); chooseLocalVideo(); };
  $('chooseLocalVideoPanel').onclick = chooseLocalVideo;
  $('addRefLayer').onclick = () => $('refFileInput').click();
  $('refFileInput').onchange = e => addReference(e.target.files[0]);
  $('videoPlayStop').onclick = () => {
    if (!mainVideo) return;
    if (mainVideo.playing) { mainVideo.playing = false; updateVideoControls(); return; }
    if (mainVideo.frame >= mainVideo.frameCount - 1) mainVideo.frame = 0;
    mainVideo.playing = true; updateVideoControls();
    if (!mainVideo.processing) processMainVideoFrame(mainVideo.frame);
  };
  $('videoTimeline').oninput = () => { if (!mainVideo) return; mainVideo.playing = false; mainVideo.previousDepth = null; mainVideo.previousDepthFrame = null; const frame = Number($('videoTimeline').value); updateVideoControls(); if (!mainVideo.processing) processMainVideoFrame(frame); };
  $('midasTemporalSmoothing').oninput = () => { $('midasTemporalValue').textContent = `${Math.round(Number($('midasTemporalSmoothing').value) * 100)}%`; if (mainVideo) { mainVideo.previousDepth = null; mainVideo.previousDepthFrame = null; } };
  $('resetView').onclick = () => { camera.position.set(0, 0, 4); orbit.target.set(0, 0, 0); orbit.update(); };
  $('showSamLayers').onchange = e => world.children.filter(mesh => mesh.userData.kind === 'sam').forEach(mesh => mesh.visible = e.target.checked);
  $('showReferences').onchange = () => { refs.forEach(ref => { const inRange = !ref.video || !mainVideo || (mainVideo.frame >= ref.startFrame && mainVideo.frame <= ref.endFrame); const shown = ref.visible && $('showReferences').checked && inRange; ref.mesh.visible = ref.overlayMesh.visible = shown; ref.previewMesh.visible = shown && ref.showMask; }); };
  addEventListener('resize', () => { const half = viewHeight / 2, aspect = innerWidth / innerHeight; camera.left = -half * aspect; camera.right = half * aspect; camera.top = half; camera.bottom = -half; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
})();

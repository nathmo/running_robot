import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.164.1/examples/jsm/controls/OrbitControls.js';
import { STLLoader } from 'https://cdn.jsdelivr.net/npm/three@0.164.1/examples/jsm/loaders/STLLoader.js';

const URDF_URL = '../robotURDF/urdf/000_Assy_Full.SLDASM.urdf';
const MESH_PREFIX = '../robotURDF/';

const ui = {
  rootName: document.getElementById('root-name'),
  linkCount: document.getElementById('link-count'),
  jointCount: document.getElementById('joint-count'),
  movableCount: document.getElementById('movable-count'),
  statusMessage: document.getElementById('status-message'),
  sliderPanel: document.getElementById('slider-panel'),
  resetViewButton: document.getElementById('reset-view-button'),
};

const sceneState = {
  scene: null,
  camera: null,
  renderer: null,
  controls: null,
  robotRoot: null,
  meshLoader: new STLLoader(),
  meshCache: new Map(),
  jointControllers: [],
  defaultCamera: null,
  defaultTarget: new THREE.Vector3(),
};

main().catch((error) => {
  console.error(error);
  ui.statusMessage.textContent = `Failed to load the URDF: ${error.message}`;
});

async function main() {
  setupScene();
  const response = await fetch(URDF_URL);
  if (!response.ok) {
    throw new Error(`Unable to fetch ${URDF_URL} (${response.status})`);
  }

  const xmlText = await response.text();
  const model = parseUrdf(xmlText);

  ui.rootName.textContent = model.rootLinks.join(', ') || 'Unknown';
  ui.linkCount.textContent = String(model.links.size);
  ui.jointCount.textContent = String(model.joints.length);

  sceneState.robotRoot = new THREE.Group();
  sceneState.robotRoot.name = 'robot-root';
  sceneState.scene.add(sceneState.robotRoot);

  for (const rootLink of model.rootLinks) {
    const rootNode = new THREE.Group();
    rootNode.name = `link:${rootLink}`;
    sceneState.robotRoot.add(rootNode);
    await buildLinkBranch(model, rootLink, rootNode);
  }

  buildJointPanel();
  frameCameraToModel();

  if (sceneState.jointControllers.length === 0) {
    ui.statusMessage.textContent = 'This URDF export contains only fixed joints, so there are no sliders to move yet.';
  } else {
    ui.statusMessage.textContent = 'Drag a slider to inspect the joint travel inside the URDF limits.';
  }

  ui.movableCount.textContent = String(sceneState.jointControllers.length);
  ui.resetViewButton.addEventListener('click', () => {
    resetCamera();
  });

  window.addEventListener('resize', onResize);
  animate();
}

function setupScene() {
  const container = document.getElementById('canvas-container');

  sceneState.scene = new THREE.Scene();
  sceneState.scene.background = new THREE.Color(0x0b1220);
  sceneState.scene.fog = new THREE.Fog(0x0b1220, 5, 20);

  const aspect = container.clientWidth / container.clientHeight;
  sceneState.camera = new THREE.PerspectiveCamera(45, aspect, 0.01, 500);
  sceneState.camera.position.set(3, 2.5, 4);

  sceneState.defaultCamera = sceneState.camera.position.clone();

  sceneState.renderer = new THREE.WebGLRenderer({ antialias: true });
  sceneState.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  sceneState.renderer.setSize(container.clientWidth, container.clientHeight);
  sceneState.renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(sceneState.renderer.domElement);

  sceneState.controls = new OrbitControls(sceneState.camera, sceneState.renderer.domElement);
  sceneState.controls.enableDamping = true;
  sceneState.controls.target.set(0, 0.3, 0);

  const ambient = new THREE.AmbientLight(0xffffff, 1.8);
  sceneState.scene.add(ambient);

  const keyLight = new THREE.DirectionalLight(0xdbeeff, 2.2);
  keyLight.position.set(5, 8, 4);
  sceneState.scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0xffd8b8, 1.2);
  fillLight.position.set(-6, 4, -5);
  sceneState.scene.add(fillLight);

  const grid = new THREE.GridHelper(12, 24, 0x35506f, 0x1f3147);
  grid.position.y = -0.75;
  sceneState.scene.add(grid);

  const axes = new THREE.AxesHelper(0.35);
  sceneState.scene.add(axes);
}

function parseUrdf(xmlText) {
  const parser = new DOMParser();
  const xml = parser.parseFromString(xmlText, 'application/xml');
  const parserError = xml.querySelector('parsererror');
  if (parserError) {
    throw new Error(parserError.textContent || 'Invalid URDF XML');
  }

  const robot = xml.querySelector('robot');
  if (!robot) {
    throw new Error('No <robot> element found in URDF');
  }

  const links = new Map();
  for (const linkElement of robot.children) {
    if (linkElement.tagName !== 'link') {
      continue;
    }

    const name = linkElement.getAttribute('name');
    if (!name) {
      continue;
    }

    const visuals = [];
    for (const child of linkElement.children) {
      if (child.tagName !== 'visual') {
        continue;
      }

      const geometry = child.querySelector('geometry mesh');
      const origin = parseOrigin(child.querySelector('origin'));
      const color = parseColor(child.querySelector('material color'));
      if (!geometry) {
        continue;
      }

      visuals.push({
        mesh: geometry.getAttribute('filename') || '',
        origin,
        color,
      });
    }

    links.set(name, { name, visuals });
  }

  const joints = [];
  const childLinks = new Set();
  for (const jointElement of robot.children) {
    if (jointElement.tagName !== 'joint') {
      continue;
    }

    const name = jointElement.getAttribute('name') || '';
    const type = jointElement.getAttribute('type') || 'fixed';
    const parent = jointElement.querySelector('parent')?.getAttribute('link') || '';
    const child = jointElement.querySelector('child')?.getAttribute('link') || '';
    const origin = parseOrigin(jointElement.querySelector('origin'));
    const axis = parseVector3(jointElement.querySelector('axis')?.getAttribute('xyz') || '0 0 1');
    const limit = jointElement.querySelector('limit');
    const lower = parseFloat(limit?.getAttribute('lower') ?? 'NaN');
    const upper = parseFloat(limit?.getAttribute('upper') ?? 'NaN');

    if (child) {
      childLinks.add(child);
    }

    joints.push({
      name,
      type,
      parent,
      child,
      origin,
      axis,
      lower: Number.isFinite(lower) ? lower : null,
      upper: Number.isFinite(upper) ? upper : null,
    });
  }

  const rootLinks = [...links.keys()].filter((linkName) => !childLinks.has(linkName));

  return { links, joints, rootLinks };
}

async function buildLinkBranch(model, linkName, parentNode) {
  const link = model.links.get(linkName);
  if (!link) {
    return;
  }

  const linkGroup = new THREE.Group();
  linkGroup.name = `link:${linkName}`;
  parentNode.add(linkGroup);

  const visuals = await Promise.all(link.visuals.map((visual) => buildVisual(visual)));
  for (const visualGroup of visuals) {
    if (visualGroup) {
      linkGroup.add(visualGroup);
    }
  }

  const outgoingJoints = model.joints.filter((joint) => joint.parent === linkName);
  for (const joint of outgoingJoints) {
    if (!joint.child) {
      continue;
    }

    const jointGroup = new THREE.Group();
    jointGroup.name = `joint:${joint.name}`;
    jointGroup.position.copy(joint.origin.position);
    jointGroup.quaternion.copy(joint.origin.quaternion);
    linkGroup.add(jointGroup);

    const motionGroup = new THREE.Group();
    motionGroup.name = `motion:${joint.name}`;
    jointGroup.add(motionGroup);

    const controller = createJointController(joint, motionGroup);
    if (controller) {
      sceneState.jointControllers.push(controller);
    }

    await buildLinkBranch(model, joint.child, motionGroup);
  }
}

function createJointController(joint, motionGroup) {
  if (joint.type === 'fixed') {
    return null;
  }

  const axis = joint.axis.lengthSq() > 0 ? joint.axis.clone().normalize() : new THREE.Vector3(0, 0, 1);
  const defaultLimits = joint.type === 'continuous'
    ? { lower: -Math.PI, upper: Math.PI }
    : { lower: -0.5, upper: 0.5 };
  const lower = joint.lower ?? defaultLimits.lower;
  const upper = joint.upper ?? defaultLimits.upper;
  const value = 0;

  const controller = {
    joint,
    motionGroup,
    axis,
    lower,
    upper,
    value,
  };

  applyJointValue(controller, value);
  return controller;
}

function buildJointPanel() {
  ui.sliderPanel.innerHTML = '';

  if (sceneState.jointControllers.length === 0) {
    const emptyState = document.createElement('div');
    emptyState.className = 'empty-state';
    emptyState.textContent = 'No movable joints were found in this URDF export. The sliders appear only when the source model contains revolute, continuous, or prismatic joints.';
    ui.sliderPanel.appendChild(emptyState);
    return;
  }

  for (const controller of sceneState.jointControllers) {
    const card = document.createElement('article');
    card.className = 'slider-card';

    const header = document.createElement('header');
    const title = document.createElement('div');
    title.className = 'joint-name';
    title.textContent = controller.joint.name;

    const range = document.createElement('div');
    range.className = 'joint-range';
    range.textContent = `${formatValue(controller.lower)} to ${formatValue(controller.upper)}`;
    header.append(title, range);

    const sliderRow = document.createElement('div');
    sliderRow.className = 'slider-row';

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(controller.lower);
    slider.max = String(controller.upper);
    slider.step = controller.joint.type === 'prismatic' ? '0.001' : '0.01';
    slider.value = '0';

    const numeric = document.createElement('input');
    numeric.type = 'number';
    numeric.min = slider.min;
    numeric.max = slider.max;
    numeric.step = slider.step;
    numeric.value = '0';

    const updateValue = (nextValue) => {
      const clamped = clamp(nextValue, controller.lower, controller.upper);
      slider.value = String(clamped);
      numeric.value = String(clamped);
      controller.value = clamped;
      applyJointValue(controller, clamped);
    };

    slider.addEventListener('input', () => updateValue(parseFloat(slider.value)));
    numeric.addEventListener('change', () => updateValue(parseFloat(numeric.value || '0')));

    sliderRow.append(slider, numeric);
    card.append(header, sliderRow);
    ui.sliderPanel.appendChild(card);
  }
}

function applyJointValue(controller, value) {
  const { joint, motionGroup, axis } = controller;
  motionGroup.position.set(0, 0, 0);
  motionGroup.quaternion.identity();

  if (joint.type === 'prismatic') {
    motionGroup.position.copy(axis.clone().multiplyScalar(value));
    return;
  }

  if (joint.type === 'revolute' || joint.type === 'continuous') {
    motionGroup.quaternion.setFromAxisAngle(axis, value);
  }
}

async function buildVisual(visual) {
  const meshPath = resolveMeshPath(visual.mesh);
  if (!meshPath) {
    return null;
  }

  let geometry = sceneState.meshCache.get(meshPath);
  if (!geometry) {
    geometry = await loadMeshGeometry(meshPath);
    sceneState.meshCache.set(meshPath, geometry);
  }

  const material = new THREE.MeshStandardMaterial({
    color: visual.color ?? 0xbac6d8,
    metalness: 0.1,
    roughness: 0.85,
    flatShading: false,
  });

  const mesh = new THREE.Mesh(geometry.clone(), material);
  mesh.name = meshPath;
  mesh.castShadow = false;
  mesh.receiveShadow = false;

  const wrapper = new THREE.Group();
  wrapper.position.copy(visual.origin.position);
  wrapper.quaternion.copy(visual.origin.quaternion);
  wrapper.add(mesh);
  return wrapper;
}

function loadMeshGeometry(meshPath) {
  return new Promise((resolve, reject) => {
    sceneState.meshLoader.load(
      meshPath,
      (geometry) => {
        geometry.computeVertexNormals();
        resolve(geometry);
      },
      undefined,
      (error) => reject(error),
    );
  });
}

function resolveMeshPath(filename) {
  if (!filename) {
    return null;
  }

  if (filename.startsWith('package://000_Assy_Full.SLDASM/')) {
    return filename.replace('package://000_Assy_Full.SLDASM/', MESH_PREFIX);
  }

  return filename;
}

function parseOrigin(originElement) {
  const position = parseVector3(originElement?.getAttribute('xyz') || '0 0 0');
  const rpy = parseVector3(originElement?.getAttribute('rpy') || '0 0 0');
  const quaternion = new THREE.Quaternion().setFromEuler(
    new THREE.Euler(rpy.x, rpy.y, rpy.z, 'XYZ'),
  );

  return { position, quaternion };
}

function parseVector3(text) {
  const parts = text.trim().split(/\s+/).map((value) => Number.parseFloat(value));
  return new THREE.Vector3(parts[0] || 0, parts[1] || 0, parts[2] || 0);
}

function parseColor(colorElement) {
  const rgba = colorElement?.getAttribute('rgba');
  if (!rgba) {
    return null;
  }

  const [red, green, blue] = rgba.trim().split(/\s+/).map((value) => Number.parseFloat(value));
  return new THREE.Color(
    Number.isFinite(red) ? red : 0.73,
    Number.isFinite(green) ? green : 0.77,
    Number.isFinite(blue) ? blue : 0.83,
  );
}

function onResize() {
  const container = document.getElementById('canvas-container');
  sceneState.camera.aspect = container.clientWidth / container.clientHeight;
  sceneState.camera.updateProjectionMatrix();
  sceneState.renderer.setSize(container.clientWidth, container.clientHeight);
}

function frameCameraToModel() {
  if (!sceneState.robotRoot) {
    return;
  }

  const bounds = new THREE.Box3().setFromObject(sceneState.robotRoot);
  if (bounds.isEmpty()) {
    return;
  }

  const size = bounds.getSize(new THREE.Vector3());
  const center = bounds.getCenter(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z) * 0.75 || 1;
  const distance = radius / Math.tan(THREE.MathUtils.degToRad(sceneState.camera.fov * 0.5));

  sceneState.defaultTarget.copy(center);
  sceneState.controls.target.copy(center);
  sceneState.camera.position.set(center.x + distance * 0.7, center.y + distance * 0.5, center.z + distance * 0.7);
  sceneState.defaultCamera = sceneState.camera.position.clone();
  sceneState.controls.update();
}

function resetCamera() {
  sceneState.camera.position.copy(sceneState.defaultCamera);
  sceneState.controls.target.copy(sceneState.defaultTarget);
  sceneState.controls.update();
}

function animate() {
  requestAnimationFrame(animate);
  sceneState.controls.update();
  sceneState.renderer.render(sceneState.scene, sceneState.camera);
}

function clamp(value, lower, upper) {
  return Math.min(Math.max(value, lower), upper);
}

function formatValue(value) {
  if (!Number.isFinite(value)) {
    return 'n/a';
  }

  return Math.abs(value) >= 1 ? value.toFixed(3) : value.toFixed(4);
}
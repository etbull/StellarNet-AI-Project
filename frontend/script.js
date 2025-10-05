// === Basic Setup ===
const scene = new THREE.Scene();
let baseColor = new THREE.Color(0x02010d);
let targetColor = new THREE.Color(0x050517);
let colorLerp = 0;
let colorDirection = 1;

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 4000);
camera.position.z = 5;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// === Create circular sprite texture ===
function createCircleTexture(color = "255,255,255", opacityCenter = 1) {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");

  const gradient = ctx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size/2);
  gradient.addColorStop(0, `rgba(${color},${opacityCenter})`);
  gradient.addColorStop(0.2, `rgba(${color},0.95)`);
  gradient.addColorStop(0.5, `rgba(${color},0.6)`);
  gradient.addColorStop(1, `rgba(${color},0)`);

  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(canvas);
}

const starTexture = createCircleTexture("255,255,255", 1);

// === Helper ===
function randomSize(base, variance = 0.5) {
  return base + Math.random() * variance;
}

// === Foreground Stars ===
const starCount = 3500;
const starsGeometry = new THREE.BufferGeometry();
const starsPosition = new Float32Array(starCount * 3);
const starSpeeds = new Float32Array(starCount);
const starSizes = new Float32Array(starCount);
const starOpacities = new Float32Array(starCount);
const starFadeTimers = new Float32Array(starCount);

for (let i = 0; i < starCount; i++) {
  starsPosition[i*3] = (Math.random()-0.5)*4000;
  starsPosition[i*3+1] = (Math.random()-0.5)*4000;
  starsPosition[i*3+2] = -Math.random()*3000-200;
  starSpeeds[i] = 0.08 + Math.random()*0.5;
  starSizes[i] = randomSize(5,4);
  starOpacities[i] = Math.random();
  starFadeTimers[i] = 0;
}

starsGeometry.setAttribute("position", new THREE.BufferAttribute(starsPosition,3));
starsGeometry.setAttribute("size", new THREE.BufferAttribute(starSizes,1));
starsGeometry.setAttribute("alpha", new THREE.BufferAttribute(starOpacities,1));

const starsMaterial = new THREE.PointsMaterial({
  map: starTexture,
  color: 0xffffff,
  size: 6,
  transparent: true,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
  sizeAttenuation: true,
  opacity: 1,
});
const starField = new THREE.Points(starsGeometry, starsMaterial);
scene.add(starField);

// === Background Stars ===
const bgStarCount = 1200;
const bgStarsGeometry = new THREE.BufferGeometry();
const bgStarsPosition = new Float32Array(bgStarCount * 3);
const bgStarSpeeds = new Float32Array(bgStarCount);
const bgStarSizes = new Float32Array(bgStarCount);

for (let i=0;i<bgStarCount;i++){
  bgStarsPosition[i*3]=(Math.random()-0.5)*6000;
  bgStarsPosition[i*3+1]=(Math.random()-0.5)*6000;
  bgStarsPosition[i*3+2]=-3000-Math.random()*2000;
  bgStarSpeeds[i]=0.03+Math.random()*0.12;
  bgStarSizes[i]=randomSize(3,3);
}
bgStarsGeometry.setAttribute("position",new THREE.BufferAttribute(bgStarsPosition,3));
bgStarsGeometry.setAttribute("size",new THREE.BufferAttribute(bgStarSizes,1));

const bgStarsMaterial=new THREE.PointsMaterial({
  map: starTexture,
  color:0xaaaaaa,
  size:4,
  transparent:true,
  blending:THREE.AdditiveBlending,
  depthWrite:false,
  sizeAttenuation:true
});
const bgStarField=new THREE.Points(bgStarsGeometry,bgStarsMaterial);
scene.add(bgStarField);

// === Cosmic Dust ===
const dustCount = 500;
const dustGeometry = new THREE.BufferGeometry();
const dustPosition = new Float32Array(dustCount*3);
const dustSpeeds = new Float32Array(dustCount);
const dustColors = new Float32Array(dustCount*4); // RGBA
const dustFadeTimers = new Float32Array(dustCount);

for(let i=0;i<dustCount;i++){
  dustPosition[i*3]=(Math.random()-0.5)*5000;
  dustPosition[i*3+1]=(Math.random()-0.5)*5000;
  dustPosition[i*3+2]=-2000-Math.random()*2000;
  dustSpeeds[i]=0.008+Math.random()*0.015;
  if(Math.random()>0.5){
    dustColors[i*4]=180/255; dustColors[i*4+1]=100/255; dustColors[i*4+2]=255/255;
  }else{
    dustColors[i*4]=100/255; dustColors[i*4+1]=150/255; dustColors[i*4+2]=255/255;
  }
  dustColors[i*4+3]=Math.random()*0.5; // initial alpha
  dustFadeTimers[i]=0;
}
dustGeometry.setAttribute("position",new THREE.BufferAttribute(dustPosition,3));
dustGeometry.setAttribute("color",new THREE.BufferAttribute(dustColors,4)); // use RGBA

const dustMaterial = new THREE.PointsMaterial({
  size:50,
  vertexColors:true,
  transparent:true,
  blending:THREE.AdditiveBlending,
  depthWrite:false,
  sizeAttenuation:true,
  map:starTexture,
});
const dustField = new THREE.Points(dustGeometry,dustMaterial);
scene.add(dustField);

// Animation loop
function animate(){
  requestAnimationFrame(animate);
  const time = performance.now()*0.001;

  // Background color
  colorLerp += 0.0003 * colorDirection;
  if (colorLerp >= 1 || colorLerp <= 0) colorDirection *= -1;
  scene.background = baseColor.clone().lerp(targetColor, colorLerp);

  // Slight camera rotation for motion effect
  camera.rotation.x = Math.sin(time * 0.1) * 0.002;
  camera.rotation.y = Math.sin(time * 0.07) * 0.002;

  // === Foreground Stars ===
  const pos = starsGeometry.attributes.position.array;
  const opac = starOpacities;
  for (let i = 0; i < starCount; i++) {
    pos[i * 3 + 2] += starSpeeds[i];
    opac[i] += (Math.random() - 0.5) * 0.02;
    opac[i] = Math.min(Math.max(opac[i], 0.1), 1);

    if (opac[i] < 1 && starFadeTimers[i] > 0) {
      starFadeTimers[i] -= 0.016;
      opac[i] = 1 - (starFadeTimers[i] / 5);
      if (opac[i] > 1) opac[i] = 1;
    }

    // Respawn when star moves past camera
    if (pos[i * 3 + 2] > camera.position.z + 100) {
      pos[i * 3] = (Math.random() - 0.5) * 4000;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 4000;
      pos[i * 3 + 2] = camera.position.z - 3000 - Math.random() * 1000;
      opac[i] = 0;
      starFadeTimers[i] = 5;
    }
  }
  starsGeometry.attributes.position.needsUpdate = true;
  starsGeometry.attributes.alpha.needsUpdate = true;

  // === Background Stars ===
  const bgPos = bgStarsGeometry.attributes.position.array;
  for (let i = 0; i < bgStarCount; i++) {
    bgPos[i * 3 + 2] += bgStarSpeeds[i];

    // Respawn background star when it passes the camera
    if (bgPos[i * 3 + 2] > camera.position.z + 200) {
      bgPos[i * 3] = (Math.random() - 0.5) * 6000;
      bgPos[i * 3 + 1] = (Math.random() - 0.5) * 6000;
      bgPos[i * 3 + 2] = camera.position.z - 4000 - Math.random() * 2000;
    }
  }
  bgStarsGeometry.attributes.position.needsUpdate = true;

  // === Cosmic Dust ===
  const dustPos = dustGeometry.attributes.position.array;
  const dustCol = dustGeometry.attributes.color.array;
  for (let i = 0; i < dustCount; i++) {
    dustPos[i * 3 + 2] += dustSpeeds[i];

    // Fade-in
    if (dustFadeTimers[i] > 0) {
      dustFadeTimers[i] -= 0.016;
      dustCol[i * 4 + 3] = 1 - (dustFadeTimers[i] / 5);
      if (dustCol[i * 4 + 3] > 1) dustCol[i * 4 + 3] = 1;
    }

    // Respawn behind camera
    if (dustPos[i * 3 + 2] > camera.position.z + 150) {
      dustPos[i * 3] = (Math.random() - 0.5) * 5000;
      dustPos[i * 3 + 1] = (Math.random() - 0.5) * 5000;
      dustPos[i * 3 + 2] = camera.position.z - 2500 - Math.random() * 2000;
      dustCol[i * 4 + 3] = 0;
      dustFadeTimers[i] = 5;
    }
  }
  dustGeometry.attributes.position.needsUpdate = true;
  dustGeometry.attributes.color.needsUpdate = true;

  renderer.render(scene, camera);
}


animate();

// === Handle Resize ===
window.addEventListener("resize",()=>{
  camera.aspect=window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth,window.innerHeight);
});

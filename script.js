import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'



const canvas = document.querySelector('canvas.webgl')
const scene = new THREE.Scene()
scene.background = new THREE.Color(0x05050f)

const ambientLight = new THREE.AmbientLight(0xffffff, 0.4)
scene.add(ambientLight)

const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
directionalLight.position.set(100, 200, 100)
scene.add(directionalLight)

const boxGeo = new THREE.BoxGeometry(1,1,1)
const material = new THREE.MeshStandardMaterial({
        transparent: true,
        opacity: 0.8
    })
const mesh = new THREE.Mesh(boxGeo, material)


window.addEventListener('resize', () => {
    sizes.width = window.innerWidth
    sizes.height = window.innerHeight
    camera.aspect = sizes.width / sizes.height
    camera.updateProjectionMatrix()
    renderer.setSize(sizes.width, sizes.height)
})

scene.add(mesh)
/**
 * canvas
 * scene
 * renderer
 * camera
 * size resizer 
 * 
 * tick function
 */
const sizes = {
    height: window.innerHeight,
    width: window.innerWidth
}

const camera = new THREE.PerspectiveCamera(45, sizes.width / sizes.height, 0.1, 5000)
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
renderer.setSize(sizes.width, sizes.height)
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))

const controls = new OrbitControls(camera, canvas)
controls.enableDamping = true
const clock = new THREE.Clock()

const tick = () =>
{
    
    const elapsedTime = clock.getElapsedTime()
    // console.log(elapsedTime)
    renderer.render(scene, camera)
    window.requestAnimationFrame(tick)
}

tick()
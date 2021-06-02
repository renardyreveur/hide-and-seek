import './style.css'
import * as THREE from 'three'
import { game } from './game_state'

// Window sizes
const sizes = {
    width: window.innerWidth,
    height: window.innerHeight
}

// Create a three.js Scene and set a camera -> add to document
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, sizes.width/sizes.height, 1, 500);
// const camera = new THREE.OrthographicCamera( -sizes.width/2, sizes.width/2, sizes.height/2, -sizes.height/2);
const renderer = new THREE.WebGLRenderer();
const clock = new THREE.Clock()
renderer.setSize(sizes.width, sizes.height);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
document.body.appendChild( renderer.domElement );


// In the event of a window resize
window.addEventListener('resize', () =>
{
	console.log("RESIZED!")
	sizes.width = window.innerWidth
	sizes.height = window.innerHeight
	camera.aspect = sizes.width / sizes.height
	camera.updateProjectionMatrix()
	renderer.setSize(sizes.width, sizes.height)
	renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
})


// Basic mesh configurations for the world and agents
const geometry = new THREE.BoxGeometry();
const wall_material = new THREE.MeshBasicMaterial ({color: 0x00ff00 });
const hider_material = new THREE.MeshBasicMaterial ({color: 0xf5ef42 });
const seeker_material = new THREE.MeshBasicMaterial ({color: 0x5442f5 });

// List to hold agents and walls mesh
const agent_group = new THREE.Group();
const wall_group = new THREE.Group();
let mesh = new THREE.Mesh(geometry, wall_material);
// scene.add(mesh)
// wall_group.add(mesh)

// Connect to Python Websocket server returning game states in protobuf binaries
const socket = new WebSocket('ws://127.0.0.1:7393/')
socket.binaryType = 'arraybuffer'

// On websocket message returns decode protobuf and render
socket.onmessage = function (event) {
	let game_state = game.GameState.decode(new Uint8Array(event.data))
    // console.log(game_state)

	let agent_info = game_state.agents
	const wall_info = game_state.walls

	// Draw wall once for the moment.
	if (wall_group.children.length === 0) {
		const mesh = new THREE.InstancedMesh(geometry, wall_material, wall_info.length)
		wall_group.add(mesh)
		scene.add(wall_group)

		for (let i = 0; i < wall_info.length; i++) {
			const position = new THREE.Vector3(
				wall_info[i].x,
				wall_info[i].y,
				0
			)
			const matrix = new THREE.Matrix4()
			matrix.setPosition(position)
			mesh.setMatrixAt(i, matrix)
		}
	}

	// Remove from scene if not being returned by server
	// let agent_uids = agent_info.map(elem => elem.uid)
	// for (let child in scene.children){
	// 	if (child.name) {
	// 		if (!agent_uids.includes(child.name)){
	// 			scene.remove(child)
	// 		}
	// 	}
	// }

	// Agent movement
	// for (const {agent_class, location, uid} in agent_info){
	// 	// const ant = agent_mesh.find(element => element[0] === uid)[1]
	// 	let ant;
	// 	agent_group.traverse( function(object){
	// 		if (object.name){
	// 			if (object.name === uid){
	// 				ant = object;
	// 			}
	// 		}
	// 	});
	//
	// 	if (typeof ant == 'undefined') {
	// 		let mesh = new THREE.Mesh(geometry, agent_class === "2" ? hider_material : seeker_material);
	// 		mesh.name = uid
	// 		agent_group.add(mesh)
	// 	}else{
	// 		let selectedAgent = scene.getObjectByName(uid)
	// 		selectedAgent.position.x = location.x
	// 		selectedAgent.position.y = location.y
	// 	}
	// }
	// 	socket.send("HEHE")

};

// scene.add(wall_group);
// for (const chi in wall_group.children){
// 	console.log(chi)
// }
// scene.add(agent_group);

camera.position.x = window.innerWidth / 4
camera.position.y = window.innerHeight / 4
camera.position.z = 500;


// three.js animate rendering
const tick = () =>
{
	const elapsedTime = clock.getElapsedTime()
	renderer.render(scene, camera);
	window.requestAnimationFrame(tick);
}

tick();
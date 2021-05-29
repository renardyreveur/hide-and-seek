import * as THREE from 'three';
import { game } from './game_state'

// Create a three.js Scene and set a camera -> add to document
const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(window.innerWidth / -2, window.innerWidth / 2,
												window.innerHeight / 2, window.innerWidth / -2);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild( renderer.domElement );

const geometry = new THREE.BoxGeometry();
const wall_material = new THREE.MeshBasicMaterial ({color: 0x00ff00 });
const hider_material = new THREE.MeshBasicMaterial ({color: 0x5442f5 });
const seeker_material = new THREE.MeshBasicMaterial ({color: 0xf5ef42 });


// Connect to Python Websocket server returning game states in protobuf binaries
const socket = new WebSocket('ws://127.0.0.1:7393/')
socket.binaryType = 'arraybuffer'

// On websocket message returns decode protobuf and render
socket.onmessage = function (event) {
	let game_state = game.GameState.decode(new Uint8Array(event.data))
    console.log(game_state)

	let agents = game_state.agents
	let walls = game_state.walls

	for (let i = 0; i< agents.length; i++){
		let a_class = agents[i].agentClass;
		let mesh;
		if (a_class === 2){
			mesh = new THREE.Mesh(geometry, hider_material);
		}
		else{
			mesh = new THREE.Mesh(geometry, seeker_material);
		}
		mesh.position.x = agents[i].location.x
		mesh.position.y = agents[i].location.y
		scene.add(mesh);
	}


	for (let i = 0; i < walls.length; i++) {
		let mesh = new THREE.Mesh(geometry, wall_material);
		mesh.position.x = walls[i].x
		mesh.position.y = walls[i].y
		scene.add(mesh);
	}

};


camera.position.x = window.innerWidth / 2
camera.position.y = window.innerHeight / 2
camera.position.z = 10;


// three.js animate rendering
function animate() {
	requestAnimationFrame(animate);
	renderer.render(scene, camera);
}

animate();
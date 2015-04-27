# CS-150, Final Project
## Robert Schneider

---

## Context
This project is a simulation of a RC Helicopter. The user has the ability to move his/her helicopter using the W,A,S,D,X, and spacebar as controls. The helicopter will have the ability to shoot rockets at other helicopters in order to destory them  

## Required components
1. Objects and Transformations will be represented using QuatRBT.[Helicopters, Obsticals in the scene, Rockets.]
2. AI Heli object will use looped Keyframing to provide the user with an enemy helicopter that follows a path.
3. The user will have 3 camera views to cycle through. an Ease-in camera will be used for the 3rd-person view, and the areal(looking down -y world axis)view. the third view will be a first person camera based in front of the helicopter.

## Other Features
1. Multiple shaders will be used, including a phong lighting with textures + ambient lighting + diffuse.
2. Each object will cast a shadow using projection matrix.
3. Animations of object positions and orientation will be implemented using parametric functions of the animation clock.
4. Interactivity using the mouse and keyboard to control other objects.
5. Tubes, octahedrons, spheres, planes, and a Helicopter geometry object (which I will make myself) (maybe some trees).
6. Multiple camera views are implemented to get different game feels.
7. A help menu explains all the functionality. Simply click h to see the help menu in the terminal window.
8. Collision detection will be implemented so the user can shoot at other ships. All collisions will be determinded spherically (setting a radius of collision from the frame, and using that to detect if a frame+its radius comes in contact with another frame+ its radius)
9. Fog of war has been added, but doesnt seem to show up.
10. Helicopter explodes when collides with rockets and other objects.
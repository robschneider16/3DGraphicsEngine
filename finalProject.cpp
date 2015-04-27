////////////////////////////////////////////////////////////////////////
//
//   Westmont College
//   CS175 : 3D Computer Graphics
//   Professor David Hunter
//	 Code by Robert Schneider
//
//   Code derived from book code for _Foundations of 3D Computer Graphics_
//   by Steven Gortler.  See AUTHORS file for more details.
//
////////////////////////////////////////////////////////////////////////
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#if __GNUG__
#   include <memory>
#endif

#include <GL/glew.h>
#ifdef __MAC__
#   include <GLUT/glut.h>
#else
#   include <GL/glut.h>
#endif

// This is to fix a linker bug
#include <pthread.h>

#include "cvec.h"
#include "geometrymaker.h"
#include "ppm.h"
#include "glsupport.h"
#include "matrix4.h"
#include "quat.h"
#include "quatrbt.h"
//#include "helirbt.h"

using namespace std; // for string, vector, iostream, and other standard C++ stuff
//using namespace tr1; // for shared_ptr

// G L O B A L S ///////////////////////////////////////////////////

// --------- IMPORTANT --------------------------------------------------------
// Before you start working on this assignment, set the following variable
// properly to indicate whether you want to use OpenGL 2.x with GLSL 1.0 or
// OpenGL 3.x+ with GLSL 1.3.
//
// Set g_Gl2Compatible = true to use GLSL 1.0 and g_Gl2Compatible = false to
// use GLSL 1.3. Make sure that your machine supports the version of GLSL you
// are using. In particular, on Mac OS X currently there is no way of using
// OpenGL 3.x with GLSL 1.3 when GLUT is used.
//
// If g_Gl2Compatible=true, shaders with -gl2 suffix will be loaded.
// If g_Gl2Compatible=false, shaders with -gl3 suffix will be loaded.
// To complete the assignment you only need to edit the shader files that get
// loaded
// ----------------------------------------------------------------------------
static const bool g_Gl2Compatible = true;

static const float g_frustMinFov = 60.0;  // A minimal of 60 degree field of view
static float g_frustFovY = g_frustMinFov; // FOV in y direction (updated by updateFrustFovY)

static const float g_frustNear = -0.1;    // near plane
static const float g_frustFar = -50.0;    // far plane
static const float g_groundY = -.1;      // y coordinate of the ground
static const float g_groundSize = 100.0;   // half the ground length

static int g_windowWidth = 512;
static int g_windowHeight = 512;
static bool g_mouseClickDown = false;     // is the mouse button pressed
static bool g_mouseLClickButton, g_mouseRClickButton, g_mouseMClickButton;
static int g_mouseClickX, g_mouseClickY;  // coordinates for mouse click event
static int g_activeShader = 2;			  // which shader to use
static int g_objToManip = 0;  			  // object to manipulate 

static int cam = 0;						  // which camera quat to use as the eye
static float cameraAlphaValue = 0.10;
static int selectedtexture = 0;

// Animation globals for time-based animation
static const float g_animStart = 0.0;
static const float g_animMax = 1.0; 
static float g_animClock = g_animStart; // clock parameter runs from g_animStart to g_animMax then repeats
static float g_animSpeed = 0.1;         // clock units per second
static int g_elapsedTime = 0;           // keeps track of how long it takes between frames
static float g_animIncrement = g_animSpeed/60.0; // updated by idle() based on GPU speed

//Global Keystrokes to manipulate the heli
static bool acendKey = false;
static bool decendKey = false;
static bool forwardKey = false;
static bool backwardKey = false;
static bool leftTurnKey = false;
static bool rightTurnKey = false;
static bool strafeLeftKey = false;
static bool strafeRightKey = false;


static float exp_Acc = -0.001;
static bool shoot = false;
static bool shadowsOn = true;

static bool explodedHuh = false;
static bool explodedHuha1 = false;






struct ShaderState {
  GlProgram program;

  // Handles to uniform variables
  GLint h_uLight; // two lights
  GLint h_uProjMatrix;
  GLint h_uModelViewMatrix;
  GLint h_uNormalMatrix;
  GLint h_uColor;
  GLint h_uTexUnit0; 

  // Handles to vertex attributes
  GLint h_aPosition;
  GLint h_aNormal;
  GLint h_aTexCoord0;

  ShaderState(const char* vsfn, const char* fsfn) {
    readAndCompileShader(program, vsfn, fsfn);

    const GLuint h = program; // short hand

    // Retrieve handles to uniform variables
    h_uLight = safe_glGetUniformLocation(h, "uLight");
    h_uProjMatrix = safe_glGetUniformLocation(h, "uProjMatrix");
    h_uModelViewMatrix = safe_glGetUniformLocation(h, "uModelViewMatrix");
    h_uNormalMatrix = safe_glGetUniformLocation(h, "uNormalMatrix");
    h_uColor = safe_glGetUniformLocation(h, "uColor");
    h_uTexUnit0 = safe_glGetUniformLocation(h, "uTexUnit0"); // textures

    // Retrieve handles to vertex attributes
    h_aPosition = safe_glGetAttribLocation(h, "aPosition");
    h_aNormal = safe_glGetAttribLocation(h, "aNormal");
    h_aTexCoord0 = safe_glGetAttribLocation(h, "aTexCoord0"); // textures

    if (!g_Gl2Compatible)
      glBindFragDataLocation(h, 0, "fragColor");
    checkGlErrors();
  }
};

static const int g_numShaders = 5;
static const char * const g_shaderFiles[g_numShaders][2] = {
  {"./shaders/basic-gl3.vshader", "./shaders/phong-gl3.fshader"},
  {"./shaders/basic-gl3.vshader", "./shaders/solid-gl3.fshader"},
  {"./shaders/basic-gl3.vshader", "./shaders/diffusetexture-gl3.fshader"},
  {"./shaders/basic-gl3.vshader", "./shaders/solidtexture-gl3.fshader"},
  {"./shaders/basic-gl3.vshader", "./shaders/shadow-gl3.fshader"}
};
static const char * const g_shaderFilesGl2[g_numShaders][2] = {
  {"./shaders/basic-gl2.vshader", "./shaders/phong-gl2.fshader"},
  {"./shaders/basic-gl2.vshader", "./shaders/solid-gl2.fshader"},
  {"./shaders/basic-gl2.vshader", "./shaders/diffusetexture-gl2.fshader"},
  {"./shaders/basic-gl2.vshader", "./shaders/solidtexture-gl2.fshader"},
  {"./shaders/basic-gl2.vshader", "./shaders/shadow-gl2.fshader"}
};

static vector<shared_ptr<ShaderState> > g_shaderStates; // our global shader states

static shared_ptr<GlTexture> g_tex0, g_tex1, g_tex2, g_tex3 ,g_tex4 , g_tex5, g_tex6; // our global texture instance
//tex_o = ground

// --------- Geometry

// Macro used to obtain relative offset of a field within a struct
#define FIELD_OFFSET(StructType, field) &(((StructType *)0)->field)

// A vertex with floating point Position, Normal, and one set of teXture coordinates;
struct VertexPNX {
  Cvec3f p, n; // position and normal vectors
  Cvec2f x; // texture coordinates

  VertexPNX() {}

  VertexPNX(float x, float y, float z,
            float nx, float ny, float nz,
            float u, float v)
    : p(x,y,z), n(nx, ny, nz), x(u, v) 
  {}

  VertexPNX(const Cvec3f& pos, const Cvec3f& normal, const Cvec2f& texCoords)
    :  p(pos), n(normal), x(texCoords) {}

  VertexPNX(const Cvec3& pos, const Cvec3& normal, const Cvec2& texCoords)
    : p(pos[0], pos[1], pos[2]), n(normal[0], normal[1], normal[2]), x(texCoords[0], texCoords[1]) {}

  // Define copy constructor and assignment operator from GenericVertex so we can
  // use make* function templates from geometrymaker.h
  VertexPNX(const GenericVertex& v) {
    *this = v;
  }

  VertexPNX& operator = (const GenericVertex& v) {
    p = v.pos;
    n = v.normal;
    x = v.tex;
    return *this;
  }
};

struct Geometry {
  GlBufferObject vbo, ibo;
  int vboLen, iboLen;

  Geometry(VertexPNX *vtx, unsigned short *idx, int vboLen, int iboLen) {
    this->vboLen = vboLen;
    this->iboLen = iboLen;

    // Now create the VBO and IBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(VertexPNX) * vboLen, vtx, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned short) * iboLen, idx, GL_STATIC_DRAW);
  }

  void draw(const ShaderState& curSS) {
    // Enable the attributes used by our shader
    safe_glEnableVertexAttribArray(curSS.h_aPosition);
    safe_glEnableVertexAttribArray(curSS.h_aNormal);
    safe_glEnableVertexAttribArray(curSS.h_aTexCoord0);

    // bind vertex buffer object
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    safe_glVertexAttribPointer(curSS.h_aPosition, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPNX), FIELD_OFFSET(VertexPNX, p));
    safe_glVertexAttribPointer(curSS.h_aNormal, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPNX), FIELD_OFFSET(VertexPNX, n));
    safe_glVertexAttribPointer(curSS.h_aTexCoord0, 4, GL_FLOAT, GL_FALSE, sizeof(VertexPNX), FIELD_OFFSET(VertexPNX, x));

    // bind index buffer object
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

    // draw!
    glDrawElements(GL_TRIANGLES, iboLen, GL_UNSIGNED_SHORT, 0);

    // Disable the attributes used by our shader
    safe_glDisableVertexAttribArray(curSS.h_aPosition);
    safe_glDisableVertexAttribArray(curSS.h_aNormal);
    safe_glDisableVertexAttribArray(curSS.h_aTexCoord0); // textures
  }
};

// Vertex buffer and index buffer associated with the different geometries
static shared_ptr<Geometry> g_plane, g_plane1, g_rocket, g_rocketNose, g_sphere1, g_ground, g_cube, g_cube1, g_cube2, g_heliBProp, g_heliWing, g_heliFeet, g_heliFeetPoles, g_heliProp,g_heliRut, g_heliRotor, g_heliBot, g_heliTail, g_heliBack, g_heliLFBody, g_heliBotBody, g_heliTopBody, g_heliCockpit, g_rect, g_sphere, g_octahedron1, g_octahedron2, g_tube, g_tube2 ,g_tube1, g_wings, g_jet, g_tube3; 


// --------- Scene ------------//

static const Cvec3 up = Cvec3(0, 1, 0); 	//up vector for object frame cross products. 
static const Cvec3 g_light = Cvec3(0.0, 20.0, 0.0);

//scale of the helicopters size
static float hs = 2.50;

static QuatRBT mainHeli = QuatRBT(Cvec3(0, 0.6, -4));
static bool R1Shot = false;
static bool R2Shot = false;
static QuatRBT mainHeliRocket1 = mainHeli*QuatRBT(Cvec3(.175*hs, -.118 *hs, .125*hs));
static QuatRBT mainHeliRocket2 = mainHeli*QuatRBT(Cvec3(-.175*hs, -.118*hs, .125*hs));

static QuatRBT aiHeli1 = QuatRBT(Cvec3(0,3,-10));

//different camera angles. first person, third person, and areal view.
static const int numCameras = 3;
static QuatRBT cameraViews[numCameras] = {QuatRBT(Cvec3( 0, 2, 6), Quat::makeXRotation(-10)),
									   	                    QuatRBT(Cvec3( 0,20, 0)) * QuatRBT(Quat::makeXRotation(-90)),
										                      QuatRBT(Cvec3( 0, 0,-1.2)) };
static QuatRBT g_eyeRbt = cameraViews[cam];

static QuatRBT inv_g_eyeRbt = inv(g_eyeRbt);
static const int numObjects = 43;

//randomly populated envorinmet of objects
static QuatRBT g_objectRbt[numObjects] =  {QuatRBT(g_light),
                                           QuatRBT(Cvec3(0, -.04, -4)),//plane under heli
                                           QuatRBT(Cvec3(0, 10, 0) ,Quat::makeXRotation(90)),//index 2-12 == square2
                                           QuatRBT(Cvec3(10, 10, 4)),//square 3
                                           QuatRBT(Cvec3(10, 0, -4)),//square 4
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeXRotation(rand()%360)),//squares 5
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeYRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeZRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeXRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeYRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeZRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeXRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeYRotation(rand()%360)),//squares 12 
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeZRotation(rand()%360)),//tubes 13
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeXRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeYRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeZRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeXRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeYRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeZRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeXRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeYRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeZRotation(rand()%360)),//Tubes 22
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeXRotation(rand()%360)),//Octahedrons23
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeYRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeZRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeXRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeYRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeZRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeXRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeYRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeZRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeXRotation(rand()%360)),//octaherdons32
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeYRotation(rand()%360)),//spheres 33
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeZRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeXRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeYRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeZRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeXRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeYRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeZRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeXRotation(rand()%360)),
                                           QuatRBT(Cvec3(((rand() % 80)-40), (rand() % 15), ((rand() % 80)-40)),Quat::makeXRotation(rand()%360))//spheres 42
                                           }; // each object gets its own RBT  
                                              //objects 2-12 are squares
                                              //objects 13-22 are tubes
                                              //objects 23-32 are octahedrons
                                              //objects 33-42 are spheres



//array of random object colors. for each object
static Cvec3 objectColors[numObjects] = {Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         Cvec3((rand() %10)*.1 , (rand() %10)*.1 , (rand() %10)*.1 ),
                                         }; 



//array of bezier curve objects, with a multidementional array for its controlpoints.
static const int g_numControlPoints = 4;
static const int g_numBezCurves = 5;
static QuatRBT g_controlPoints[g_numBezCurves][g_numControlPoints] ={{QuatRBT(Cvec3( 0,4,-12), (Quat::makeZRotation(-10) * Quat::makeYRotation(90)  * Quat::makeXRotation(-30)        )      ),
                                                                      QuatRBT(Cvec3(-6,4,-12), (Quat::makeZRotation(-45) * Quat::makeYRotation(130) * Quat::makeXRotation(-30)        )      ),
                                                                      QuatRBT(Cvec3(-12,4,-6), (Quat::makeZRotation(-45) * Quat::makeYRotation(160) * Quat::makeXRotation(-30)        )      ),
                                                                      QuatRBT(Cvec3(-12,4,0),  (Quat::makeZRotation(-10) * Quat::makeYRotation(180) * Quat::makeXRotation(-30)        )      ) },
                                                                      //^^ is       
                                                                   {  QuatRBT(Cvec3(-12,4,0),  (Quat::makeZRotation(-10) * Quat::makeYRotation(180) * Quat::makeXRotation(-30)        )      ),
                                                                      QuatRBT(Cvec3(-12,4,6),  (Quat::makeZRotation(-45) * Quat::makeYRotation(200) * Quat::makeXRotation(-30)        )      ),
                                                                      QuatRBT(Cvec3(-6,7,10),  (Quat::makeZRotation(-45) * Quat::makeYRotation(215) * Quat::makeXRotation(10)         )      ),
                                                                      QuatRBT(Cvec3(-4,8,12),  (Quat::makeZRotation(-10) * Quat::makeYRotation(225) * Quat::makeXRotation(20)         )      ) },

                                                                   {  QuatRBT(Cvec3(-4,8,12),  (Quat::makeZRotation(-10) * Quat::makeYRotation(225) * Quat::makeXRotation(20)         )      ),
                                                                      QuatRBT(Cvec3(-2,9,14),  (Quat::makeZRotation(5) * Quat::makeYRotation(260) * Quat::makeXRotation(30)         )      ),
                                                                      QuatRBT(Cvec3(2,4,14),   (Quat::makeZRotation(5) * Quat::makeYRotation(270) * Quat::makeXRotation(20)         )      ),
                                                                      QuatRBT(Cvec3(4,3,12),   (Quat::makeZRotation(10) * Quat::makeYRotation(280) * Quat::makeXRotation(0)          )      ) },

                                                                   {  QuatRBT(Cvec3(4,3,12),   (Quat::makeZRotation(10) * Quat::makeYRotation(280) * Quat::makeXRotation(0)          )      ),
                                                                      QuatRBT(Cvec3(6,2,10),   (Quat::makeZRotation(45) * Quat::makeYRotation(290) * Quat::makeXRotation(-50)        )      ),
                                                                      QuatRBT(Cvec3(12,4,6),   (Quat::makeZRotation(45) * Quat::makeYRotation(320) * Quat::makeXRotation(-30)        )      ),
                                                                      QuatRBT(Cvec3(12,4,0),   (Quat::makeZRotation(10) * Quat::makeYRotation(  360) * Quat::makeXRotation(-40)        )      ) },

                                                                   {  QuatRBT(Cvec3(12,4,0),   (Quat::makeZRotation(10) * Quat::makeYRotation(360) * Quat::makeXRotation(-40)        )      ),
                                                                      QuatRBT(Cvec3(12,4,-6),  (Quat::makeZRotation(45) * Quat::makeYRotation(390) * Quat::makeXRotation(-30)        )      ),
                                                                      QuatRBT(Cvec3(6,4,-12),  (Quat::makeZRotation(45) * Quat::makeYRotation(420) * Quat::makeXRotation(-30)        )      ),
                                                                      QuatRBT(Cvec3(0,4,-12),  (Quat::makeZRotation(-10) * Quat::makeYRotation(450) * Quat::makeXRotation(-30)        )      ) }};


//these are modifications/QuatRBTs that will update each ships component when it explodes.
static QuatRBT crashRBT[14] ={QuatRBT(Cvec3(((rand()%10)*.01)-.05, ((rand()%10)*-.01) , ((rand()%10)*.01)-.05  )),
                              QuatRBT(Cvec3(((rand()%10)*.01)-.05, ((rand()%10)*-.01) , ((rand()%10)*.01)-.05  )),
                              QuatRBT(Cvec3(((rand()%10)*.01)-.05, ((rand()%10)*-.01) , ((rand()%10)*.01)-.05  )),
                              QuatRBT(Cvec3(((rand()%10)*.01)-.05, ((rand()%10)*-.01) , ((rand()%10)*.01)-.05  )),
                              QuatRBT(Cvec3(((rand()%10)*.01)-.05, ((rand()%10)*-.01) , ((rand()%10)*.01)-.05  )),
                              QuatRBT(Cvec3(((rand()%10)*.01)-.05, ((rand()%10)*-.01) , ((rand()%10)*.01)-.05  )),
                              QuatRBT(Cvec3(((rand()%10)*.01)-.05, ((rand()%10)*-.01) , ((rand()%10)*.01)-.05  )),
                              QuatRBT(Cvec3(((rand()%10)*.01)-.05, ((rand()%10)*-.01) , ((rand()%10)*.01)-.05  )),
                              QuatRBT(Cvec3(((rand()%10)*.01)-.05, ((rand()%10)*-.01) , ((rand()%10)*.01)-.05  )),
                              QuatRBT(Cvec3(((rand()%10)*.01)-.05, ((rand()%10)*-.01) , ((rand()%10)*.01)-.05  )),
                              QuatRBT(Cvec3(((rand()%10)*.01)-.05, ((rand()%10)*-.01) , ((rand()%10)*.01)-.05  )),
                              QuatRBT(Cvec3(((rand()%10)*.01)-.05, ((rand()%10)*-.01) , ((rand()%10)*.01)-.05  )),
                              QuatRBT(Cvec3(((rand()%10)*.01)-.05, ((rand()%10)*-.01) , ((rand()%10)*.01)-.05  )),
                              QuatRBT(Cvec3(((rand()%10)*.01)-.05, ((rand()%10)*-.01) , ((rand()%10)*.01)-.05  ))
                            };
//when the explode function is called, each object in the helicopter will become a part of this array, and will get updated with the QuatRBTs above to show an explosion.
static QuatRBT HeliComponentsRBT[14];




///////////////// END OF G L O B A L S //////////////////////////////////////////////////



static void initGround() {
   // A x-z plane at y = g_groundY of dimension [-g_groundSize, g_groundSize]^2
  VertexPNX vtx[4] = {
    VertexPNX(-g_groundSize, g_groundY, -g_groundSize, 0, 1, 0, 0, 0),   // The ground is going to
    VertexPNX(-g_groundSize, g_groundY,  g_groundSize, 0, 1, 0, 0, 50),  // be really big, so we are
    VertexPNX( g_groundSize, g_groundY,  g_groundSize, 0, 1, 0, 50, 50), // going to tile the texture
    VertexPNX( g_groundSize, g_groundY, -g_groundSize, 0, 1, 0, 50, 0),  // 10 by 10 instead of stretching
  };
  unsigned short idx[] = {0, 1, 2, 0, 2, 3};
  g_ground.reset(new Geometry(&vtx[0], &idx[0], 4, 6));

}

static void initSphere(){
  int ibLen, vbLen;
  getSphereVbIbLen(40, 40, vbLen, ibLen);
  vector<VertexPNX> vtx(vbLen);
  vector<unsigned short> idx(ibLen);
  makeSphere( 2, 40, 40, vtx.begin(), idx.begin());
  g_sphere.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

}

static void InitObjects(){

  // each kind of geometry needs to be initialized here
  int ibLen, vbLen;
  getCubeVbIbLen(vbLen, ibLen);
  // Temporary storage for vertex and index buffers
  vector<VertexPNX> vtx(vbLen);
  vector<unsigned short> idx(ibLen);
  makeCube(1, vtx.begin(), idx.begin());
  g_cube1.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

  // each kind of geometry needs to be initialized here
  getCubeVbIbLen(vbLen, ibLen);
  // Temporary storage for vertex and index buffers
  vtx.resize(vbLen);
  idx.resize(ibLen);
  makeCube(2, vtx.begin(), idx.begin());
  g_cube2.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

  //plane
  getPlaneVbIbLen(vbLen, ibLen);
  vtx.resize(vbLen);
  idx.resize(ibLen);
  makePlane(5, vtx.begin(), idx.begin());
  g_plane1.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

  //sphere
  getSphereVbIbLen(40, 40, vbLen, ibLen);
  vtx.resize(vbLen);
  idx.resize(ibLen);
  makeSphere(1.0, 40, 40, vtx.begin(), idx.begin());
  g_sphere1.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));
  
  //octahedron
  getOctahedronVbIbLen(vbLen, ibLen);
  vtx.resize(vbLen);
  idx.resize(ibLen);
  makeOctahedron(1.0, vtx.begin(), idx.begin());
  g_octahedron1.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

  //octahedron
  getOctahedronVbIbLen(vbLen, ibLen);
  vtx.resize(vbLen);
  idx.resize(ibLen);
  makeOctahedron(2.0, vtx.begin(), idx.begin());
  g_octahedron2.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

  //tube
  getTubeVbIbLen(20, vbLen, ibLen);
  vtx.resize(vbLen);
  idx.resize(ibLen);
  makeTube(1.0, 3.0, 20, vtx.begin(), idx.begin());
  g_tube1.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

  //tube
  getTubeVbIbLen(20, vbLen, ibLen);
  vtx.resize(vbLen);
  idx.resize(ibLen);
  makeTube(3.0, 6.0, 20, vtx.begin(), idx.begin());
  g_tube2.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

}


static void initRocket(){
  // each kind of geometry needs to be initialized here
  int ibLen, vbLen;
  getTubeVbIbLen(8, vbLen, ibLen);
  // Temporary storage for vertex and index buffers
  vector<VertexPNX> vtx(vbLen);
  vector<unsigned short> idx(ibLen);
  makeTube(.05, .4, 8, vtx.begin(), idx.begin());
  g_rocket.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

  // each kind of geometry needs to be initialized here
  getOctahedronVbIbLen(vbLen, ibLen);
  // Temporary storage for vertex and index buffers
  vtx.resize(vbLen);
  idx.resize(ibLen);
  makeOctahedron(.065, vtx.begin(), idx.begin());
  g_rocketNose.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));


}


static void initHeli(){
  //cockpit
  int ibLen, vbLen;
  getSphereVbIbLen(8, 8, vbLen, ibLen);
  vector<VertexPNX> vtx(vbLen);
  vector<unsigned short> idx(ibLen);
  makeSphere(0.099 * hs, 8, 8, vtx.begin(), idx.begin());
  g_heliCockpit.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));
  
  //lower body of helcopter
  getRectVbIbLen(vbLen, ibLen);
  vtx.resize(vbLen);
  idx.resize(ibLen);
  //make rect params (length, width, height, vtxbegin, idxbegin)
  makeRect(.3*hs, .2*hs ,.1*hs, vtx.begin(), idx.begin());
  g_heliBotBody.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

  //upper body of helicopter
  getTubeVbIbLen( 8, vbLen, ibLen);
  vtx.resize(vbLen);
  idx.resize(ibLen);
  //make tube params (radius, height, slices, vtxbegin, idxbegin)
  makeTube(.1*hs, .3*hs , 8, vtx.begin(), idx.begin());
  g_heliTopBody.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

  //lower-front body of helicopter
  getTubeVbIbLen( 16, vbLen, ibLen);
  vtx.resize(vbLen);
  idx.resize(ibLen);
  //make tube params (radius, height, slices, vtxbegin, idxbegin)
  makeTube(.1*hs, .1*hs , 16, vtx.begin(), idx.begin());
  g_heliLFBody.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

  //undercarriage of helicopter
  getTubeVbIbLen( 40, vbLen, ibLen);
  vtx.resize(vbLen);
  idx.resize(ibLen);
  //make tube params (radius, height, slices, vtxbegin, idxbegin)
  makeTube(.1*hs, .314*hs , 40, vtx.begin(), idx.begin());
  g_heliBot.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

  //heli Back body
  getOctahedronVbIbLen(vbLen, ibLen);
  vtx.resize(vbLen);
  idx.resize(ibLen);
  makeOctahedron(.14*hs, vtx.begin(), idx.begin());
  g_heliBack.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

  //heli tail pole
  getTubeVbIbLen( 40, vbLen, ibLen);
  vtx.resize(vbLen);
  idx.resize(ibLen);
  //make tube params (radius, height, slices, vtxbegin, idxbegin)
  makeTube(.01*hs, .7*hs , 40, vtx.begin(), idx.begin());
  g_heliTail.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

  //heli rotor
  getTubeVbIbLen( 40, vbLen, ibLen);
  vtx.resize(vbLen);
  idx.resize(ibLen);
  //make tube params (radius, height, slices, vtxbegin, idxbegin)
  makeTube(.006*hs, .007*hs , 40, vtx.begin(), idx.begin());
  g_heliRotor.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

  //helcopter top prop blades
  getRectVbIbLen(vbLen, ibLen);
  vtx.resize(vbLen);
  idx.resize(ibLen);
  //make rect params (length, width, height, vtxbegin, idxbegin)
  makeRect(1.2*hs, .04*hs ,.001*hs, vtx.begin(), idx.begin());
  g_heliProp.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

  //helcopter tail prop blades
  getRectVbIbLen(vbLen, ibLen);
  vtx.resize(vbLen);
  idx.resize(ibLen);
  //make rect params (length, width, height, vtxbegin, idxbegin)
  makeRect(.3*hs, .02*hs ,.001*hs, vtx.begin(), idx.begin());
  g_heliBProp.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

  //tail rutter
  getRectVbIbLen(vbLen, ibLen);
  vtx.resize(vbLen);
  idx.resize(ibLen);
  //make rect params (length, width, height, vtxbegin, idxbegin)
  makeRect(.15*hs, .02*hs ,.04*hs, vtx.begin(), idx.begin());
  g_heliRut.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

  //heli feet landing poles
  getTubeVbIbLen( 40, vbLen, ibLen);
  vtx.resize(vbLen);
  idx.resize(ibLen);
  //make tube params (radius, height, slices, vtxbegin, idxbegin)
  makeTube(.007*hs, .17*hs , 8, vtx.begin(), idx.begin());
  g_heliFeetPoles.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

  //helli feet
  getTubeVbIbLen( 40, vbLen, ibLen);
  vtx.resize(vbLen);
  idx.resize(ibLen);
  //make tube params (radius, height, slices, vtxbegin, idxbegin)
  makeTube(.017*hs, .4*hs , 40, vtx.begin(), idx.begin());
  g_heliFeet.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));

  //Wings
  getRectVbIbLen(vbLen, ibLen);
  vtx.resize(vbLen);
  idx.resize(ibLen);
  //make rect params (length, width, height, vtxbegin, idxbegin)
  makeRect(.15*hs, .07*hs ,.015*hs, vtx.begin(), idx.begin());
  g_heliWing.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));
}

static void explodeMain(){
  for(int i = 0; i < 14; i++){
      HeliComponentsRBT[i] = mainHeli;
    }
    exp_Acc = 0.00;
    explodedHuh = true;
}

static void explodeA1(){
  for(int i = 0; i < 14; i++){
      HeliComponentsRBT[i] = aiHeli1;
    }
    exp_Acc = 0.00;
    explodedHuha1 = true;
}
// takes a projection matrix and send to the the shaders
static void sendProjectionMatrix(const ShaderState& SS, const Matrix4& projMatrix) {
  GLfloat glmatrix[16];
  projMatrix.writeToColumnMajorMatrix(glmatrix); // send projection matrix
  safe_glUniformMatrix4fv(SS.h_uProjMatrix, glmatrix);
}

// takes MVM and its normal matrix to the shaders
static void sendModelViewNormalMatrix(const ShaderState& SS, const Matrix4& MVM, const Matrix4& NMVM) {
  GLfloat glmatrix[16];
  MVM.writeToColumnMajorMatrix(glmatrix); // send MVM
  safe_glUniformMatrix4fv(SS.h_uModelViewMatrix, glmatrix);

  NMVM.writeToColumnMajorMatrix(glmatrix); // send NMVM
  safe_glUniformMatrix4fv(SS.h_uNormalMatrix, glmatrix);
}

// update g_frustFovY from g_frustMinFov, g_windowWidth, and g_windowHeight
static void updateFrustFovY() {
  if (g_windowWidth >= g_windowHeight)
    g_frustFovY = g_frustMinFov;
  else {
    const double RAD_PER_DEG = 0.5 * CS175_PI/180;
    g_frustFovY = atan2(sin(g_frustMinFov * RAD_PER_DEG) * g_windowHeight / g_windowWidth, cos(g_frustMinFov * RAD_PER_DEG)) / RAD_PER_DEG;
  }
}

static Matrix4 makeProjectionMatrix() {
  return Matrix4::makeProjection(
           g_frustFovY, g_windowWidth / static_cast <double> (g_windowHeight),
           g_frustNear, g_frustFar);
}
//draws the shadows of the helicopter if shadows are active, and if its not exploding.
static void drawHeliShadow(QuatRBT whichHeli){
  // short hand for current shader state
  // build & send proj. matrix to vshader
  const Matrix4 projmat = makeProjectionMatrix();
  const Matrix4 eyeRbt = rigTFormToMatrix(g_eyeRbt);
  const Matrix4 invEyeRbt = inv(eyeRbt);// store inverse so we don't have to recompute it
  const Cvec3 eyeLight = g_light;//Cvec3(invEyeRbt * Cvec4(g_light ,1)); // g_light position in eye coordinates
  const Matrix4 g_lightRbt = Matrix4::makeTranslation(g_light);
  const ShaderState& curSS = *g_shaderStates[4];
  glUseProgram(curSS.program);//use the shadow shader
  sendProjectionMatrix(curSS, projmat); // send projection matrix to shader not sure if needed 
  safe_glUniform3f(curSS.h_uLight, eyeLight[0], eyeLight[1], eyeLight[2]); // shaders need light positions

  //helicopter cockpit
  Matrix4 MVM = invEyeRbt * Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(whichHeli);
  Matrix4 NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  safe_glUniform3f(curSS.h_uColor, 0, 0, 0); 
  g_heliCockpit->draw(curSS);

  //helicopter lower front body
  MVM = invEyeRbt * Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, -.04*hs , .014*hs)) * QuatRBT(Quat::makeXRotation(-110)));
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliLFBody->draw(curSS);

  //helicopter Top body                          //this is for moving the object to its correct location relative to its helicopter frame
  MVM = invEyeRbt * (Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0 , .15*hs)))) * Matrix4::makeTranslation(Cvec3(0,0.001, 0));
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliTopBody->draw(curSS);

  //helicopter Bottom body
  MVM = invEyeRbt * Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, -.04*hs , .15*hs)));
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliBotBody->draw(curSS);

  //helicopter  bACK
  MVM = invEyeRbt * Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0 , .3*hs)) * QuatRBT(Quat::makeZRotation(45)) * QuatRBT(Quat::makeYRotation(180)));
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliBack->draw(curSS);

  //heli tail
  MVM = invEyeRbt * Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0 , .7*hs)));
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliTail->draw(curSS);

  //heli rotor
  QuatRBT rotorY = QuatRBT(Quat::makeYRotation(g_animClock*720000 * g_animSpeed));
  MVM = invEyeRbt * Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0.106*hs , .14*hs)) * QuatRBT(Quat::makeXRotation(90)));
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliRotor->draw(curSS);

  //heli top blades
  MVM = invEyeRbt * (Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0.11*hs , .14*hs)) * rotorY))* Matrix4::makeTranslation(Cvec3(0,0.01, 0));
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliProp->draw(curSS);
  //heli top blades2
  MVM = invEyeRbt * (Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0.11*hs , .14*hs)) * QuatRBT(Quat::makeYRotation(90)) * rotorY))* Matrix4::makeTranslation(Cvec3(0,0.01, 0));
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliProp->draw(curSS);

  //heli back blades
  QuatRBT rotorX = QuatRBT(Quat::makeYRotation(g_animClock*720000 * g_animSpeed));
  MVM = invEyeRbt * Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0 , 1*hs)) * QuatRBT(Quat::makeZRotation(90)) * rotorX);
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliBProp->draw(curSS);
  MVM = invEyeRbt * Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0 , 1*hs)) * QuatRBT(Quat::makeZRotation(90)) * QuatRBT(Quat::makeYRotation(90)) * rotorX);
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliBProp->draw(curSS);

  //heli back rutter
  MVM = invEyeRbt * Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0.113 , 1.01*hs)) * QuatRBT(Quat::makeXRotation(-60)));
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliRut->draw(curSS);


  //heli feet
  MVM = invEyeRbt * Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0.15*hs, -.2*hs , .18*hs)));
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliFeet->draw(curSS);
  MVM = invEyeRbt * Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(-0.15*hs, -.2*hs , 0.18*hs)));
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliFeet->draw(curSS);

  //wings
  MVM = invEyeRbt * Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0.15*hs, -0.06*hs , .16*hs)) * QuatRBT(Quat::makeYRotation(-80)));
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliWing->draw(curSS);
  MVM = invEyeRbt * Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(-0.15*hs, -0.06*hs , .16*hs)) * QuatRBT(Quat::makeYRotation(80)));
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliWing->draw(curSS);

  //helifeet holders
  MVM = invEyeRbt * Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(-0.115*hs, -.135*hs , 0.22*hs)) * QuatRBT(Quat::makeZRotation(60) * Quat::makeYRotation(90)));
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliFeetPoles->draw(curSS);
  MVM = invEyeRbt * Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0.115*hs, -.135*hs , 0.22*hs)) * QuatRBT(Quat::makeZRotation(-60) * Quat::makeYRotation(90)));
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliFeetPoles->draw(curSS);
  MVM = invEyeRbt * Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(-0.115*hs, -.135*hs , 0.11*hs)) * QuatRBT(Quat::makeZRotation(60) * Quat::makeYRotation(90)));
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliFeetPoles->draw(curSS);
  MVM = invEyeRbt * Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0.115*hs, -.135*hs , 0.11*hs)) * QuatRBT(Quat::makeZRotation(-60) * Quat::makeYRotation(90)));
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliFeetPoles->draw(curSS);
}
//draws the shadows of every non-heli object
static void drawObjectShadows(){
  // short hand for current shader state
  // build & send proj. matrix to vshader
  const Matrix4 projmat = makeProjectionMatrix();
  const Matrix4 eyeRbt = rigTFormToMatrix(g_eyeRbt);
  const Matrix4 invEyeRbt = inv(eyeRbt);// store inverse so we don't have to recompute it
  const Cvec3 eyeLight = g_light;//Cvec3(invEyeRbt * Cvec4(g_light ,1)); // g_light position in eye coordinates
  const Matrix4 g_lightRbt = Matrix4::makeTranslation(g_light);
  const ShaderState& curSS = *g_shaderStates[4];
  glUseProgram(curSS.program);//use the shadow shader
  sendProjectionMatrix(curSS, projmat); // send projection matrix to shader not sure if needed 
  safe_glUniform3f(curSS.h_uLight, eyeLight[0], eyeLight[1], eyeLight[2]); // shaders need light positions
  Matrix4 MVM, NMVM;

  //draw the shadows for the whole array of objects.
  //OBJECT 1 IS HELIPAD PLANE
  //objects 2-12 are squares
  //objects 13-22 are tubes
  //objects 23-32 are octahedrons
  //objects 33-42 are spheres
  for(int i = 2; i <= 12; i++){
    MVM = invEyeRbt  * Matrix4::makeShadowMatrix(g_lightRbt)  * rigTFormToMatrix(g_objectRbt[i]);
    NMVM = normalMatrix(MVM);
    sendModelViewNormalMatrix(curSS, MVM, NMVM);
    safe_glUniform3f(curSS.h_uColor, 0,0,0);
    safe_glUniform1i(curSS.h_uTexUnit0, 0); // texture unit 0 for ground
    g_cube2->draw(curSS);
  }
  for(int i = 13; i <= 22; i++){
    MVM = invEyeRbt  * Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(g_objectRbt[i]);
    NMVM = normalMatrix(MVM);
    sendModelViewNormalMatrix(curSS, MVM, NMVM);
    g_tube2->draw(curSS);
  }
  for(int i = 23; i <= 32; i++){
    MVM = invEyeRbt  * Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(g_objectRbt[i]);
    NMVM = normalMatrix(MVM);
    sendModelViewNormalMatrix(curSS, MVM, NMVM);
    g_octahedron2->draw(curSS);
  }
  for(int i = 33; i <= 42; i++){
    MVM = invEyeRbt  * Matrix4::makeShadowMatrix(g_lightRbt) * rigTFormToMatrix(g_objectRbt[i]);
    NMVM = normalMatrix(MVM);
    sendModelViewNormalMatrix(curSS, MVM, NMVM);
    g_sphere1->draw(curSS);
  }
}

//draws the contolable helicopter
static void drawHeliMain(){
  // short hand for current shader state
  // build & send proj. matrix to vshader
  QuatRBT whichHeli = mainHeli;
  Matrix4 projmat = makeProjectionMatrix();
  Matrix4 invEyeRbt = inv(rigTFormToMatrix(g_eyeRbt));   // store inverse so we don't have to recompute it
  Cvec3 eyeLight = Cvec3( Cvec4(g_light ,1)); //g_light position in eye coordinates
  ShaderState& curSS = *g_shaderStates[g_activeShader];  // alias for currently selected shader
  glUseProgram(curSS.program); // select shader we want to use
  sendProjectionMatrix(curSS, projmat); // send projection matrix to shader
  safe_glUniform3f(curSS.h_uLight, eyeLight[0], eyeLight[1], eyeLight[2]); // shaders need light positions


  //helicopter cockpit
  Matrix4 MVM;
  if(explodedHuh == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[0]);
  }else{
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli);
  }
  Matrix4 NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  safe_glUniform3f(curSS.h_uColor, .4, .4, 1); // blue color
  safe_glUniform1i(curSS.h_uTexUnit0, 4); // texture unit 1 for helicopter texture
  g_heliCockpit->draw(curSS);

  //helicopter lower front body
  if(explodedHuh == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[1] * QuatRBT(Cvec3(0, -.04*hs , .014*hs)) * QuatRBT(Quat::makeXRotation(-110)));
  }else{
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, -.04*hs , .014*hs)) * QuatRBT(Quat::makeXRotation(-110)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliLFBody->draw(curSS);

  //helicopter Top body                          //this is for moving the object to its correct location relative to its helicopter frame
  if(explodedHuh == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[2] * QuatRBT(Cvec3(0, 0 , .15*hs)));
  }else{
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0 , .15*hs)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  safe_glUniform3f(curSS.h_uColor, .7, .7, .7);
  safe_glUniform1i(curSS.h_uTexUnit0, 1); // texture unit 1 for heli texture
  g_heliTopBody->draw(curSS);

  //helicopter Bottom body
  if(explodedHuh == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[3] * QuatRBT(Cvec3(0, -.04*hs , .15*hs)));
  }else{
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, -.04*hs , .15*hs)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliBotBody->draw(curSS);

  //heli bottom
  if(explodedHuh == true){  
    MVM = invEyeRbt * (rigTFormToMatrix(HeliComponentsRBT[4] * QuatRBT(Cvec3(0, -.094*hs , .14*hs))) * Matrix4::makeScale(Cvec3(1, .18, 1)));
  }else{ 
    MVM = invEyeRbt * (rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, -.094*hs , .14*hs))) * Matrix4::makeScale(Cvec3(1, .18, 1)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliBot->draw(curSS);

  //helicopter  bACK
  if(explodedHuh == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[5] *  QuatRBT(Cvec3(0, 0 , .3*hs)) * QuatRBT(Quat::makeZRotation(45)) * QuatRBT(Quat::makeYRotation(180)));
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0 , .3*hs)) * QuatRBT(Quat::makeZRotation(45)) * QuatRBT(Quat::makeYRotation(180)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliBack->draw(curSS);

  //heli tail
  if(explodedHuh == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[6] * QuatRBT(Cvec3(0, 0 , .7*hs)));
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0 , .7*hs)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliTail->draw(curSS);

  
  //heli rotor
  QuatRBT rotorY = QuatRBT(Quat::makeYRotation(g_animClock*720000 * g_animSpeed));
  if(explodedHuh == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[7] *  QuatRBT(Cvec3(0, 0.106*hs , .14*hs)) * QuatRBT(Quat::makeXRotation(90)));
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0.106*hs , .14*hs)) * QuatRBT(Quat::makeXRotation(90)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliRotor->draw(curSS);

  //heli top blades
  if(explodedHuh == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[8] * QuatRBT(Cvec3(0, 0.11*hs , .14*hs)) * rotorY); 
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0.11*hs , .14*hs)) * rotorY);
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliProp->draw(curSS);
  //heli top blades2
  if(explodedHuh == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[8] * QuatRBT(Cvec3(0, 0.11*hs , .14*hs)) * QuatRBT(Quat::makeYRotation(90)) * rotorY); 
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0.11*hs , .14*hs))* QuatRBT(Quat::makeYRotation(90))  * rotorY);
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliProp->draw(curSS);

  //heli back blades
  QuatRBT rotorX = QuatRBT(Quat::makeYRotation(g_animClock*720000 * g_animSpeed));
  if(explodedHuh == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[9] * QuatRBT(Cvec3(0, 0 , 1*hs)) * QuatRBT(Quat::makeZRotation(90)) * rotorX);
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0 , 1*hs)) * QuatRBT(Quat::makeZRotation(90)) * rotorX);
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliBProp->draw(curSS);
  if(explodedHuh == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[9] * QuatRBT(Cvec3(0, 0 , 1*hs)) * QuatRBT(Quat::makeZRotation(90)) * QuatRBT(Quat::makeYRotation(90)) * rotorX);
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0 , 1*hs)) * QuatRBT(Quat::makeZRotation(90)) * QuatRBT(Quat::makeYRotation(90)) * rotorX);
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliBProp->draw(curSS);

  //heli back rutter
  if(explodedHuh == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[10] * QuatRBT(Cvec3(0, 0.113 , 1.01*hs)) * QuatRBT(Quat::makeXRotation(-60)));
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0.113 , 1.01*hs)) * QuatRBT(Quat::makeXRotation(-60)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliRut->draw(curSS);


  //heli feet
  if(explodedHuh == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[11] *  QuatRBT(Cvec3(0.15*hs, -.2*hs , .18*hs)));
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0.15*hs, -.2*hs , .18*hs)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliFeet->draw(curSS);
  if(explodedHuh == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[11] *  QuatRBT(Cvec3(-0.15*hs, -.2*hs , .18*hs)));
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(-0.15*hs, -.2*hs , .18*hs)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliFeet->draw(curSS);

  //wings
  if(explodedHuh == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[12] * QuatRBT(Cvec3(0.15*hs, -0.06*hs , .16*hs)) * QuatRBT(Quat::makeYRotation(-80)));
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0.15*hs, -0.06*hs , .16*hs)) * QuatRBT(Quat::makeYRotation(-80)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliWing->draw(curSS);
  if(explodedHuh == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[12] * QuatRBT(Cvec3(-0.15*hs, -0.06*hs , .16*hs)) * QuatRBT(Quat::makeYRotation(80)));
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(-0.15*hs, -0.06*hs , .16*hs)) * QuatRBT(Quat::makeYRotation(80)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliWing->draw(curSS);

  //helifeet holders

  if(explodedHuh == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[13] * QuatRBT(Cvec3(-0.115*hs, -.135*hs , 0.17*hs)) * QuatRBT(Quat::makeZRotation(60) * Quat::makeYRotation(90)));
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(-0.115*hs, -.135*hs , 0.17*hs)) * QuatRBT(Quat::makeZRotation(60) * Quat::makeYRotation(90)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliFeetPoles->draw(curSS);

  if(explodedHuh == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[13] * QuatRBT(Cvec3(0.115*hs, -.135*hs , 0.17*hs)) * QuatRBT(Quat::makeZRotation(-60) * Quat::makeYRotation(90)));
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0.115*hs, -.135*hs , 0.17*hs)) * QuatRBT(Quat::makeZRotation(-60) * Quat::makeYRotation(90)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliFeetPoles->draw(curSS);

  if(shadowsOn && !explodedHuh){
    drawHeliShadow(whichHeli);}
}
//draws the helicopter on the Bez path
static void drawHeliA1(){
  // short hand for current shader state
  // build & send proj. matrix to vshader
  QuatRBT whichHeli = aiHeli1;
  Matrix4 projmat = makeProjectionMatrix();
  Matrix4 invEyeRbt = inv(rigTFormToMatrix(g_eyeRbt));   // store inverse so we don't have to recompute it
  Cvec3 eyeLight = Cvec3( Cvec4(g_light ,1)); //g_light position in eye coordinates
  ShaderState& curSS = *g_shaderStates[g_activeShader];  // alias for currently selected shader
  glUseProgram(curSS.program); // select shader we want to use
  sendProjectionMatrix(curSS, projmat); // send projection matrix to shader
  safe_glUniform3f(curSS.h_uLight, eyeLight[0], eyeLight[1], eyeLight[2]); // shaders need light positions


  //helicopter cockpit
  Matrix4 MVM;
  if(explodedHuha1 == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[0]);
  }else{
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli);
  }
  Matrix4 NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  safe_glUniform3f(curSS.h_uColor, .4, .4, 1); // blue color
  safe_glUniform1i(curSS.h_uTexUnit0, 4); // texture unit 1 for helicopter texture
  g_heliCockpit->draw(curSS);

  //helicopter lower front body
  if(explodedHuha1 == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[1] * QuatRBT(Cvec3(0, -.04*hs , .014*hs)) * QuatRBT(Quat::makeXRotation(-110)));
  }else{
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, -.04*hs , .014*hs)) * QuatRBT(Quat::makeXRotation(-110)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliLFBody->draw(curSS);

  //helicopter Top body                          //this is for moving the object to its correct location relative to its helicopter frame
  if(explodedHuha1 == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[2] * QuatRBT(Cvec3(0, 0 , .15*hs)));
  }else{
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0 , .15*hs)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  safe_glUniform3f(curSS.h_uColor, .7, .7, .7);
  safe_glUniform1i(curSS.h_uTexUnit0, 1); // texture unit 1 for heli texture
  g_heliTopBody->draw(curSS);

  //helicopter Bottom body
  if(explodedHuha1 == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[3] * QuatRBT(Cvec3(0, -.04*hs , .15*hs)));
  }else{
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, -.04*hs , .15*hs)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliBotBody->draw(curSS);

  //heli bottom
  if(explodedHuha1 == true){  
    MVM = invEyeRbt * (rigTFormToMatrix(HeliComponentsRBT[4] * QuatRBT(Cvec3(0, -.094*hs , .14*hs))) * Matrix4::makeScale(Cvec3(1, .18, 1)));
  }else{ 
    MVM = invEyeRbt * (rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, -.094*hs , .14*hs))) * Matrix4::makeScale(Cvec3(1, .18, 1)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliBot->draw(curSS);

  //helicopter  bACK
  if(explodedHuha1 == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[5] *  QuatRBT(Cvec3(0, 0 , .3*hs)) * QuatRBT(Quat::makeZRotation(45)) * QuatRBT(Quat::makeYRotation(180)));
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0 , .3*hs)) * QuatRBT(Quat::makeZRotation(45)) * QuatRBT(Quat::makeYRotation(180)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliBack->draw(curSS);

  //heli tail
  if(explodedHuha1 == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[6] * QuatRBT(Cvec3(0, 0 , .7*hs)));
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0 , .7*hs)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliTail->draw(curSS);

  
  //heli rotor
  QuatRBT rotorY = QuatRBT(Quat::makeYRotation(g_animClock*720000 * g_animSpeed));
  if(explodedHuha1 == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[7] *  QuatRBT(Cvec3(0, 0.106*hs , .14*hs)) * QuatRBT(Quat::makeXRotation(90)));
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0.106*hs , .14*hs)) * QuatRBT(Quat::makeXRotation(90)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliRotor->draw(curSS);

  //heli top blades
  if(explodedHuha1 == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[8] * QuatRBT(Cvec3(0, 0.11*hs , .14*hs)) * rotorY); 
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0.11*hs , .14*hs)) * rotorY);
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliProp->draw(curSS);
  //heli top blades2
  if(explodedHuha1 == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[8] * QuatRBT(Cvec3(0, 0.11*hs , .14*hs)) * QuatRBT(Quat::makeYRotation(90)) * rotorY); 
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0.11*hs , .14*hs))* QuatRBT(Quat::makeYRotation(90))  * rotorY);
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliProp->draw(curSS);

  //heli back blades
  QuatRBT rotorX = QuatRBT(Quat::makeYRotation(g_animClock*720000 * g_animSpeed));
  if(explodedHuha1 == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[9] * QuatRBT(Cvec3(0, 0 , 1*hs)) * QuatRBT(Quat::makeZRotation(90)) * rotorX);
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0 , 1*hs)) * QuatRBT(Quat::makeZRotation(90)) * rotorX);
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliBProp->draw(curSS);
  if(explodedHuha1 == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[9] * QuatRBT(Cvec3(0, 0 , 1*hs)) * QuatRBT(Quat::makeZRotation(90)) * QuatRBT(Quat::makeYRotation(90)) * rotorX);
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0 , 1*hs)) * QuatRBT(Quat::makeZRotation(90)) * QuatRBT(Quat::makeYRotation(90)) * rotorX);
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliBProp->draw(curSS);

  //heli back rutter
  if(explodedHuha1 == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[10] * QuatRBT(Cvec3(0, 0.113 , 1.01*hs)) * QuatRBT(Quat::makeXRotation(-60)));
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0, 0.113 , 1.01*hs)) * QuatRBT(Quat::makeXRotation(-60)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliRut->draw(curSS);


  //heli feet
  if(explodedHuha1 == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[11] *  QuatRBT(Cvec3(0.15*hs, -.2*hs , .18*hs)));
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0.15*hs, -.2*hs , .18*hs)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliFeet->draw(curSS);
  if(explodedHuha1 == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[11] *  QuatRBT(Cvec3(-0.15*hs, -.2*hs , .18*hs)));
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(-0.15*hs, -.2*hs , .18*hs)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliFeet->draw(curSS);

  //wings
  if(explodedHuha1 == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[12] * QuatRBT(Cvec3(0.15*hs, -0.06*hs , .16*hs)) * QuatRBT(Quat::makeYRotation(-80)));
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0.15*hs, -0.06*hs , .16*hs)) * QuatRBT(Quat::makeYRotation(-80)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliWing->draw(curSS);
  if(explodedHuha1 == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[12] * QuatRBT(Cvec3(-0.15*hs, -0.06*hs , .16*hs)) * QuatRBT(Quat::makeYRotation(80)));
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(-0.15*hs, -0.06*hs , .16*hs)) * QuatRBT(Quat::makeYRotation(80)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliWing->draw(curSS);

  //helifeet holders
  
  if(explodedHuha1 == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[13] * QuatRBT(Cvec3(-0.115*hs, -.135*hs , 0.17*hs)) * QuatRBT(Quat::makeZRotation(60) * Quat::makeYRotation(90)));
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(-0.115*hs, -.135*hs , 0.17*hs)) * QuatRBT(Quat::makeZRotation(60) * Quat::makeYRotation(90)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliFeetPoles->draw(curSS);
  if(explodedHuha1 == true){  
    MVM = invEyeRbt * rigTFormToMatrix(HeliComponentsRBT[13] * QuatRBT(Cvec3(0.115*hs, -.135*hs , 0.17*hs)) * QuatRBT(Quat::makeZRotation(-60) * Quat::makeYRotation(90)));
  }else{ 
    MVM = invEyeRbt * rigTFormToMatrix(whichHeli * QuatRBT(Cvec3(0.115*hs, -.135*hs , 0.17*hs)) * QuatRBT(Quat::makeZRotation(-60) * Quat::makeYRotation(90)));
  }
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_heliFeetPoles->draw(curSS);

  if(shadowsOn && !explodedHuha1)
    drawHeliShadow(whichHeli);
}
//draws a rocket.
static void drawRocket(QuatRBT r){
  // short hand for current shader state
  // build & send proj. matrix to vshader
  QuatRBT rock = r;
  if(explodedHuh){
    rock = QuatRBT(Cvec3(0,-4,0));
  }
  Matrix4 projmat = makeProjectionMatrix();
  Matrix4 invEyeRbt = inv(rigTFormToMatrix(g_eyeRbt));// store inverse so we don't have to recompute it
  Cvec3 eyeLight = Cvec3(invEyeRbt * Cvec4(g_light ,1)); // g_light position in eye coordinates
  ShaderState& curSS = *g_shaderStates[g_activeShader]; // alias for currently selected shader
  glUseProgram(curSS.program); // select shader we want to use
  sendProjectionMatrix(curSS, projmat); // send projection matrix to shader
  safe_glUniform3f(curSS.h_uLight, eyeLight[0], eyeLight[1], eyeLight[2]); // shaders need light positions

  //rocket body
  Matrix4 MVM = invEyeRbt * rigTFormToMatrix(rock);
  Matrix4 NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  safe_glUniform3f(curSS.h_uColor, 1, 0, 0); // blue color
  safe_glUniform1i(curSS.h_uTexUnit0, 1); // texture unit 1 for helicopter texture
  g_rocket->draw(curSS);

  MVM = invEyeRbt * rigTFormToMatrix(rock*QuatRBT(Cvec3(0,0,-.2)));
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  g_rocketNose->draw(curSS);
}
//check to see if rocket1 has collided with AI ship
static bool Rocket1HeliCollisionCheck(){
  float CR = .9;
  float ox, oy, oz, ty, tz, tx;
  bool answer = false;
  ox = mainHeliRocket1.getTranslation()[0];
  oy = mainHeliRocket1.getTranslation()[1];
  oz = mainHeliRocket1.getTranslation()[2];
  //if it collides with the sun
    QuatRBT k = aiHeli1;
    tx = (k.getTranslation())[0];
    ty = (k.getTranslation())[1];
    tz = (k.getTranslation())[2];
    if( ((ox >= tx-CR) && (ox <= tx+CR)) && ((oy >= ty-CR) && (oy <= ty+CR)) && ((oz >= tz-CR) && (oz <= tz+CR)) ){
          answer = true;
        }
      return answer;
}
//check to see if rocket2 has collided with AI ship
static bool Rocket2HeliCollisionCheck(){
  float CR = .9;
  float ox, oy, oz, ty, tz, tx;
  bool answer = false;
  ox = mainHeliRocket2.getTranslation()[0];
  oy = mainHeliRocket2.getTranslation()[1];
  oz = mainHeliRocket2.getTranslation()[2];
  //if it collides with the sun
    QuatRBT k = aiHeli1;
    tx = (k.getTranslation())[0];
    ty = (k.getTranslation())[1];
    tz = (k.getTranslation())[2];
    if( ((ox >= tx-CR) && (ox <= tx+CR)) && ((oy >= ty-CR) && (oy <= ty+CR)) && ((oz >= tz-CR) && (oz <= tz+CR)) ){
          answer = true;
        }
    return answer;
}

//checks if the object in collideds with any other object in the scene.
static bool Collision(QuatRBT object){
  float CR = hs+.3;
  float ox, oy, oz, ty, tz, tx;
  bool answer = false;
  ox = object.getTranslation()[0];
  oy = object.getTranslation()[1];
  oz = object.getTranslation()[2];
  //if it collides with the sun
  for(int i = 0; i < 1; i++){
    QuatRBT k = g_objectRbt[i];
    tx = (k.getTranslation())[0];
    ty = (k.getTranslation())[1];
    tz = (k.getTranslation())[2];
    if( ((ox >= tx-CR) && (ox <= tx+CR)) && ((oy >= ty-CR) && (oy <= ty+CR)) && ((oz >= tz-CR) && (oz <= tz+CR)) ){
          answer = true;
        }
    }
  //if it collides with a cube.
  //colision radius is .95
    CR= hs - .5;
  for(int i = 2; i <= 12; i++){
    QuatRBT k = g_objectRbt[i];
    tx = (k.getTranslation())[0];
    ty = (k.getTranslation())[1];
    tz = (k.getTranslation())[2];
    if( ((ox >= tx-CR) && (ox <= tx+CR)) && ((oy >= ty-CR) && (oy <= ty+CR)) && ((oz >= tz-CR) && (oz <= tz+CR)) ){
          answer = true;
        }  
  }

  //if it collides with a octahedron.
  //colision radius is .95
    CR= hs - .4;
  for(int i = 23; i <= 32; i++){
    QuatRBT k = g_objectRbt[i];
    tx = (k.getTranslation())[0];
    ty = (k.getTranslation())[1];
    tz = (k.getTranslation())[2];
    if( ((ox >= tx-CR) && (ox <= tx+CR)) && ((oy >= ty-CR) && (oy <= ty+CR)) && ((oz >= tz-CR) && (oz <= tz+CR)) ){
          answer = true;
        }  
  }

  //colides with sphere
  CR= hs - .3;
  for(int i = 33; i <= 42; i++){
    QuatRBT k = g_objectRbt[i];
    tx = (k.getTranslation())[0];
    ty = (k.getTranslation())[1];
    tz = (k.getTranslation())[2];
    if( ((ox >= tx-CR) && (ox <= tx+CR)) && ((oy >= ty-CR) && (oy <= ty+CR)) && ((oz >= tz-CR) && (oz <= tz+CR)) ){
          answer = true;
        }  
  }
  return answer;
  //if it collides with a tube surface.
  //collisions with tubes are turned off
}

//draws everything
static void drawScene(){
  // short hand for current shader state
  // build & send proj. matrix to vshader
  Matrix4 projmat = makeProjectionMatrix();
  Matrix4 eyeRbt = rigTFormToMatrix(g_eyeRbt);
  Matrix4 g_lightRbt = Matrix4::makeTranslation(g_light);
  Matrix4 invEyeRbt = inv(eyeRbt);// store inverse so we don't have to recompute it
  Cvec3 eyeLight = Cvec3(invEyeRbt * Cvec4(g_light ,0)); // g_light position in eye coordinates
  ShaderState& curSS = *g_shaderStates[g_activeShader]; // alias for currently selected shader
  glUseProgram(curSS.program); // select shader we want to use
  sendProjectionMatrix(curSS, projmat); // send projection matrix to shader
  safe_glUniform3f(curSS.h_uLight, eyeLight[0], eyeLight[1], eyeLight[2]); // shaders need light positions

  // a couple of animation
  g_objectRbt[22] = g_objectRbt[22] * QuatRBT(Quat::makeYRotation(g_animSpeed*5));
  g_objectRbt[10] = g_objectRbt[10] * QuatRBT(Quat::makeZRotation(g_animSpeed*5));
  g_objectRbt[10] = g_objectRbt[1] * QuatRBT(Quat::makeYRotation(g_animSpeed*3)) * inv(g_objectRbt[1]) * g_objectRbt[10];
  g_objectRbt[6] = g_objectRbt[1] * QuatRBT(Quat::makeZRotation(g_animSpeed*3)) * inv(g_objectRbt[1]) * g_objectRbt[6]; 
  g_objectRbt[29] = g_objectRbt[1] * QuatRBT(Quat::makeXRotation(g_animSpeed*3)) * inv(g_objectRbt[1]) * g_objectRbt[29];
  g_objectRbt[5] = g_objectRbt[2] * QuatRBT(Quat::makeXRotation(g_animSpeed*3)) * inv(g_objectRbt[2]) * g_objectRbt[5];
  g_objectRbt[12] = g_objectRbt[12] * QuatRBT(Quat::makeYRotation(g_animSpeed*3));
  g_objectRbt[32] = g_objectRbt[32] * QuatRBT(Quat::makeXRotation(g_animSpeed*3));
  g_objectRbt[26] = g_objectRbt[26] * QuatRBT(Quat::makeZRotation(g_animSpeed*3));
  g_objectRbt[28] = g_objectRbt[28] * QuatRBT(Quat::makeZRotation(g_animSpeed*3));
  g_objectRbt[32] = g_objectRbt[32] * QuatRBT(Quat::makeYRotation(g_animSpeed*3));
  g_objectRbt[17] = g_objectRbt[17] * QuatRBT(Quat::makeXRotation(g_animSpeed*3));
  g_objectRbt[22] = g_objectRbt[22] * QuatRBT(Quat::makeYRotation(g_animSpeed*3));
  g_objectRbt[30] = g_objectRbt[30] * QuatRBT(Quat::makeXRotation(g_animSpeed*3));
  g_objectRbt[9] = g_objectRbt[0] * QuatRBT(Quat::makeXRotation(g_animSpeed*3)) * inv(g_objectRbt[0]) * g_objectRbt[6]; 
  g_objectRbt[16] = g_objectRbt[0] * QuatRBT(Quat::makeYRotation(g_animSpeed*3)) * inv(g_objectRbt[0]) * g_objectRbt[6]; 
  g_objectRbt[26] = g_objectRbt[0] * QuatRBT(Quat::makeZRotation(g_animSpeed*3)) * inv(g_objectRbt[0]) * g_objectRbt[6]; 
  g_objectRbt[31] = g_objectRbt[0] * QuatRBT(Quat::makeXRotation(g_animSpeed*3)) * inv(g_objectRbt[0]) * g_objectRbt[6]; 
  g_objectRbt[39] = g_objectRbt[0] * QuatRBT(Quat::makeYRotation(g_animSpeed*3)) * inv(g_objectRbt[0]) * g_objectRbt[6]; 





  //ground
  Matrix4 MVM = invEyeRbt * Matrix4();
  Matrix4 NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  safe_glUniform3f(curSS.h_uColor, .1, .1, .1);
  safe_glUniform1i(curSS.h_uTexUnit0, selectedtexture); // texture unit 0 for ground
  g_ground->draw(curSS);


  drawHeliMain();
  drawHeliA1();
  //rockets are their own rbt. get updated if they are shot.
    drawRocket(mainHeliRocket1);
    drawRocket(mainHeliRocket2);




  //draw the whole array of objects.
  //OBJECT 1 IS HELIPAD PLANE
  //objects 2-12 are squares
  //objects 13-22 are tubes
  //objects 23-32 are octahedrons
  //objects 33-42 are spheres
  MVM = invEyeRbt * rigTFormToMatrix(g_objectRbt[1]);
    NMVM = normalMatrix(MVM);
    Cvec3 col = objectColors[1];
    sendModelViewNormalMatrix(curSS, MVM, NMVM);
    safe_glUniform3f(curSS.h_uColor, col(0), col(1), col(2));
    safe_glUniform1i(curSS.h_uTexUnit0, 4); // texture unit 4 for helipad
    g_plane1->draw(curSS);


  for(int i = 2; i <= 12; i++){
    MVM = invEyeRbt * rigTFormToMatrix(g_objectRbt[i]);
    NMVM = normalMatrix(MVM);
    col = objectColors[i];
    sendModelViewNormalMatrix(curSS, MVM, NMVM);
    safe_glUniform3f(curSS.h_uColor, col(0), col(1), col(2));
    safe_glUniform1i(curSS.h_uTexUnit0, 2); // texture unit 0 for ground
    g_cube2->draw(curSS);
  }
  for(int i = 13; i <= 22; i++){
    MVM = invEyeRbt * rigTFormToMatrix(g_objectRbt[i]);
    NMVM = normalMatrix(MVM);
    col = objectColors[i];
    sendModelViewNormalMatrix(curSS, MVM, NMVM);
    safe_glUniform3f(curSS.h_uColor, col(0), col(1), col(2));
    safe_glUniform1i(curSS.h_uTexUnit0, 1); // texture unit 0 for ground
    g_tube2->draw(curSS);
  }
  for(int i = 23; i <= 32; i++){
    MVM = invEyeRbt * rigTFormToMatrix(g_objectRbt[i]);
    NMVM = normalMatrix(MVM);
    col = objectColors[i];
    sendModelViewNormalMatrix(curSS, MVM, NMVM);
    safe_glUniform3f(curSS.h_uColor, col(0), col(1), col(2));
    safe_glUniform1i(curSS.h_uTexUnit0, 4); // texture unit 0 for ground
    g_octahedron2->draw(curSS);
  }
  for(int i = 33; i <= 42; i++){
    MVM = invEyeRbt * rigTFormToMatrix(g_objectRbt[i]);
    NMVM = normalMatrix(MVM);
    col = objectColors[i];
    sendModelViewNormalMatrix(curSS, MVM, NMVM);
    safe_glUniform3f(curSS.h_uColor,col(0), col(1), col(2));
    safe_glUniform1i(curSS.h_uTexUnit0, 2); // texture unit 0 for ground
    g_sphere1->draw(curSS);
  }

  if(shadowsOn)
    drawObjectShadows();



  ShaderState& sunSS = *g_shaderStates[3]; // alias for currently selected shader
  glUseProgram(sunSS.program); // select shader we want to use
  sendProjectionMatrix(sunSS, projmat); // send projection matrix to shader
  safe_glUniform3f(sunSS.h_uLight, eyeLight[0], eyeLight[1], eyeLight[2]); // shaders need light positions

  MVM = invEyeRbt * g_lightRbt;
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(sunSS, MVM, NMVM);
  safe_glUniform3f(sunSS.h_uColor, 1, .2, 0);
  safe_glUniform1i(sunSS.h_uTexUnit0, 3); // texture unit 3 for sun
  g_sphere->draw(sunSS);
}
//ease in camera
static void updateCamera(){
  Cvec3 t, c1, c0; 
  t = (mainHeli * cameraViews[cam]).getTranslation();
  c0 = g_eyeRbt.getTranslation();
  for(int i = 0; i < 3; i++){
  c1[i] = (1- cameraAlphaValue) * c0[i] + cameraAlphaValue * t[i];
  }
  Quat o, f1, f0;
  if(cam != 1){  
    o = (mainHeli * cameraViews[cam]).getRotation();
    f0 = g_eyeRbt.getRotation();
    f1[0] = (1- cameraAlphaValue) * f0[0] + cameraAlphaValue * o[0]; 
    f1[1] = (1- cameraAlphaValue) * f0[1] + cameraAlphaValue * o[1]; 
    f1[2] = (1- cameraAlphaValue) * f0[2] + cameraAlphaValue * o[2];
    f1[3] = (1- cameraAlphaValue) * f0[3] + cameraAlphaValue * o[3]; 
    g_eyeRbt.setRotation(f1);  
  }
  g_eyeRbt.setTranslation(c1);
}
//moves the helicopter based on eaystrokes.
static void updateShipFrame(){
  QuatRBT destination;
	//change later on to make functions and acceleration.
	//move up with respect to the helicopters frame
	if(acendKey == true){
    destination = mainHeli * QuatRBT(Cvec3(0, 4*g_animSpeed/5, 0));
    if(Collision(destination) == false){
      mainHeli = destination;
    }else{explodeMain();}
  }

	//move down w/rsp to heli frame
	if(decendKey == true){
    Cvec3 posn = mainHeli.getTranslation();
    if(posn[1] >= 0.6){
      destination = mainHeli * QuatRBT(Cvec3(0, -4*g_animSpeed/5, 0));
      if(Collision(destination) == false){
        mainHeli = destination;
      }else{explodeMain();}
    }
  }

	//rotate about y axis of heli
	if(leftTurnKey == true){
    mainHeli = mainHeli * QuatRBT(Quat::makeYRotation(16*g_animSpeed));
  }	

	//rotate about y axis of heli 	
	if(rightTurnKey == true){
    mainHeli = mainHeli * QuatRBT(Quat::makeYRotation(-16*g_animSpeed));
  }

	//translate left w/ rsp to heli frame	
	if(strafeLeftKey == true){
    destination = mainHeli * QuatRBT(Cvec3(-4*g_animSpeed/5, 0, 0));
    if(Collision(destination) == false){
      mainHeli = destination;
    }else{explodeMain();}
  }

	//translate right w/ rsp to heli frame	
	if(strafeRightKey == true){
    destination = mainHeli * QuatRBT(Cvec3(4*g_animSpeed/5, 0, 0));
    if (Collision(destination) == false){
      mainHeli = destination;
    }else{explodeMain();}
  }

	//translate -z w/ rsp to heli frame	
	if(forwardKey == true){
    destination = mainHeli * QuatRBT(Cvec3(0, 0, -4*g_animSpeed/5));
    if (Collision(destination) == false){
      mainHeli = destination;
    }else{explodeMain();}
  }

	//translate +z w/rsp to heli frame
	if(backwardKey == true){
    destination = mainHeli * QuatRBT(Cvec3(0, 0, 4*g_animSpeed/5));
    if (Collision(destination) == false){
      mainHeli = destination;
    }else{explodeMain();}

  }
  
  if(!R1Shot){
    mainHeliRocket1 = mainHeli*QuatRBT(Cvec3(.175*hs, -.118 *hs, .125*hs));
  }
  if(!R2Shot){
    mainHeliRocket2 = mainHeli*QuatRBT(Cvec3(-.175*hs, -.118 *hs, .125*hs));
  }

  //Ship idle just sways alittle 
  //I didnt like this, but it adds alittle jiggle to the ship
  //float cc1 = (((rand() % 10) * .001)-.004);
  //float cc2 = (((rand() % 10) * .001)-.004);
  //float cc3 = (((rand() % 10) * .001)-.004);
  //mainHeli = mainHeli * QuatRBT(Cvec3(cc1, cc2, cc3));

  if(shoot){
    if(R1Shot == true){
      shoot = false;
      R2Shot = true;
      R1Shot = false;
      mainHeliRocket1 = mainHeli*QuatRBT(Cvec3(.175*hs, -.118 *hs, .125*hs));
    }else{
      shoot = false;
      R1Shot = true;
      R2Shot = false;
      mainHeliRocket2 = mainHeli*QuatRBT(Cvec3(-.175*hs, -.118 *hs, .125*hs));    
    }
  }

  if(R1Shot){
    mainHeliRocket1 = mainHeliRocket1*QuatRBT(Cvec3(0,0,-10*g_animSpeed));
  }
  if(R2Shot){
    mainHeliRocket2 = mainHeliRocket2*QuatRBT(Cvec3(0,0,-10*g_animSpeed));
  }

  //if(!g_mouseClickDown)
    updateCamera();
}
//trans fact of the bezier Quats
static Cvec3 getPosnAtTime(float tim){
  float t, p0, p1, p2, p3, tmax, p;
  QuatRBT cPtemp0, cPtemp1, cPtemp2, cPtemp3;
  Cvec3 output, linear, cP0, cP1, cP2, cP3;
  tmax = tim / g_numBezCurves;
  p = tim;
  t = tim;
  if(p < .2){
    cPtemp0 = g_controlPoints[0][0];
    cPtemp1 = g_controlPoints[0][1];
    cPtemp2 = g_controlPoints[0][2];
    cPtemp3 = g_controlPoints[0][3];
    t = t*5;
  }
  else if(p <.4){
    cPtemp0 = g_controlPoints[1][0];
    cPtemp1 = g_controlPoints[1][1];
    cPtemp2 = g_controlPoints[1][2];
    cPtemp3 = g_controlPoints[1][3];
    t = (t-.2)*5;
  }
  else if(p < .6){
    cPtemp0 = g_controlPoints[2][0];
    cPtemp1 = g_controlPoints[2][1];
    cPtemp2 = g_controlPoints[2][2];
    cPtemp3 = g_controlPoints[2][3];
    t = (t-.4)*5;  
  }
  else if(p < .8){
    cPtemp0 = g_controlPoints[3][0];
    cPtemp1 = g_controlPoints[3][1];
    cPtemp2 = g_controlPoints[3][2];
    cPtemp3 = g_controlPoints[3][3];
    t = (t-.6)*5; 
  }
  else{
    cPtemp0 = g_controlPoints[4][0];
    cPtemp1 = g_controlPoints[4][1];
    cPtemp2 = g_controlPoints[4][2];
    cPtemp3 = g_controlPoints[4][3];
    t = (t-.8)*5;  
  }
  cP0 = cPtemp0.getTranslation();
  cP1 = cPtemp1.getTranslation();
  cP2 = cPtemp2.getTranslation();
  cP3 = cPtemp3.getTranslation();
  for(int i = 0; i < 4; i++){
    p0 = cP0(i);
    p1 = cP1(i);
    p2 = cP2(i);
    p3 = cP3(i);
    output(i) = ((1-t)*(1-t)*(1-t)*p0)+((3*t)*(1-t)*(1-t)*p1)+((3*t*t)*(1-t)*p2)+((t*t*t)*p3);
  }
  return output;
}
//lin fact of Bez curve Quats
static Quat getOrientationAtTime(float tim){
  float t, p0, p1, p2, p3, tmax, p;
  QuatRBT cPtemp0, cPtemp1, cPtemp2, cPtemp3;
  Quat cP0, cP1, cP2, cP3, answer;
  Cvec4 output;
  tmax = tim / g_numBezCurves;
  p = tim;
  t = tim;
  if(p < .2){
    cPtemp0 = g_controlPoints[0][0];
    cPtemp1 = g_controlPoints[0][1];
    cPtemp2 = g_controlPoints[0][2];
    cPtemp3 = g_controlPoints[0][3];
    t = t*5;
  }
  else if(p <.4){
    cPtemp0 = g_controlPoints[1][0];
    cPtemp1 = g_controlPoints[1][1];
    cPtemp2 = g_controlPoints[1][2];
    cPtemp3 = g_controlPoints[1][3];
    t = (t-.2)*5;
  }
  else if(p < .6){
    cPtemp0 = g_controlPoints[2][0];
    cPtemp1 = g_controlPoints[2][1];
    cPtemp2 = g_controlPoints[2][2];
    cPtemp3 = g_controlPoints[2][3];
    t = (t-.4)*5;  
  }
  else if(p < .8){
    cPtemp0 = g_controlPoints[3][0];
    cPtemp1 = g_controlPoints[3][1];
    cPtemp2 = g_controlPoints[3][2];
    cPtemp3 = g_controlPoints[3][3];
    t = (t-.6)*5; 
  }
  else{
    cPtemp0 = g_controlPoints[4][0];
    cPtemp1 = g_controlPoints[4][1];
    cPtemp2 = g_controlPoints[4][2];
    cPtemp3 = g_controlPoints[4][3];
    t = (t-.8)*5;  
  }
  cP0 = cPtemp0.getRotation();
  cP1 = cPtemp1.getRotation();
  cP2 = cPtemp2.getRotation();
  cP3 = cPtemp3.getRotation();
  for(int i = 0; i <= 4; i++){
    p0 = cP0(i);
    p1 = cP1(i);
    p2 = cP2(i);
    p3 = cP3(i);
    output(i) = ((1-t)*(1-t)*(1-t)*p0)+((3*t)*(1-t)*(1-t)*p1)+((3*t*t)*(1-t)*p2)+((t*t*t)*p3);
  }
  answer = Quat(output(0), output(1), output(2), output(3));
  return answer;
}

static void display() {
  glUseProgram(g_shaderStates[g_activeShader]->program);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);   // clear framebuffer color&depth
  drawScene();
  glutSwapBuffers();                                    // show the back buffer (where we rendered stuff)
  checkGlErrors();

  // Calculate frames per second 
  static int oldTime = -1;
  static int frames = 0;
  static int lastTime = -1;

  int currentTime = glutGet(GLUT_ELAPSED_TIME); // returns milliseconds
  g_elapsedTime = currentTime - lastTime;       // how long last frame took
  lastTime = currentTime;

        if (oldTime < 0)
                oldTime = currentTime;

        frames++;

        if (currentTime - oldTime >= 5000) // report FPS every 5 seconds
        {
                cout << "Frames per second: "
                        << float(frames)*1000.0/(currentTime - oldTime) << endl;
                cout << "Elapsed ms since last frame: " << g_elapsedTime << endl;
                oldTime = currentTime;
                frames = 0;
        }
}

static void reshape(const int w, const int h) {
  g_windowWidth = w;
  g_windowHeight = h;
  glViewport(0, 0, w, h);
  updateFrustFovY();
  glutPostRedisplay();
}


static void mouse(const int button, const int state, const int x, const int y) {
  g_mouseClickX = x;
  g_mouseClickY = g_windowHeight - y - 1;  // conversion from GLUT window-coordinate-system to OpenGL window-coordinate-system

  g_mouseLClickButton |= (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN);
  g_mouseRClickButton |= (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN);
  g_mouseMClickButton |= (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN);

  g_mouseLClickButton &= !(button == GLUT_LEFT_BUTTON && state == GLUT_UP);
  g_mouseRClickButton &= !(button == GLUT_RIGHT_BUTTON && state == GLUT_UP);
  g_mouseMClickButton &= !(button == GLUT_MIDDLE_BUTTON && state == GLUT_UP);

  g_mouseClickDown = g_mouseLClickButton || g_mouseRClickButton || g_mouseMClickButton;
}


static void motion(const int x, const int y) {
  const double dx = x - g_mouseClickX;
  const double dy = g_windowHeight - y - 1 - g_mouseClickY;

  QuatRBT m,a;
  Cvec3 tli, tlf;
  if (g_mouseLClickButton && !g_mouseRClickButton) { // left button down?
    m = linFact(QuatRBT(normalize(Quat::makeXRotation(-dy) * Quat::makeYRotation(dx))));
    g_objectRbt[g_objToManip] =  g_objectRbt[g_objToManip] * m ;
  }
  else if (g_mouseRClickButton && !g_mouseLClickButton) { // right button down?
    //m = QuatRBT(Cvec3(dx, dy, 0));
    tli = g_objectRbt[g_objToManip].getTranslation();
    tlf = tli + (Cvec3(dx,dy,0) * 0.01);
    g_objectRbt[g_objToManip].setTranslation(tlf);
  }
  else if (g_mouseMClickButton || (g_mouseLClickButton && g_mouseRClickButton)) {  // middle or (left and right) button down?
    tli = g_objectRbt[g_objToManip].getTranslation();
    tlf = tli + (Cvec3(0,0,-dy) * 0.01);
    g_objectRbt[g_objToManip].setTranslation(tlf);
 
  }
  g_mouseClickX = x;
  g_mouseClickY = g_windowHeight - y - 1;
}


static void idle(){
  g_animIncrement = g_animSpeed * g_elapsedTime / 1000; // rescale animation increment
  g_animClock += g_animIncrement;           // Update animation clock 
         if (g_animClock >= g_animMax)       // and cycle to start if necessary.
     g_animClock = g_animStart;

          //the following if's check collisions.
          for(int p = 0; p < 14; p++){
            HeliComponentsRBT[p] = HeliComponentsRBT[p] * crashRBT[p];
          }
          if(Collision(mainHeliRocket1))
            R1Shot = false;

          if(Collision(mainHeliRocket1))
            R2Shot = false;

          if(Collision(aiHeli1))
            explodeA1();

          if(Rocket2HeliCollisionCheck()){
            R2Shot = false;
            explodeA1();
          }
          if(Rocket1HeliCollisionCheck()){
            R1Shot = false;
            explodeA1();
          }

          //update frames
          updateShipFrame();
          if(explodedHuha1 == false){
            aiHeli1 = QuatRBT(getPosnAtTime(g_animClock), getOrientationAtTime(g_animClock));
          }
         glutPostRedisplay();  // for animation
}


static void keyboardUp(const unsigned char key, const int x, const int y){
	//these set the booleans to false when the ey is not pressed.
	 //Movement Cases
	switch (key){
  case 'w':
  	forwardKey = false;
  	break;
  case 'a':
  	leftTurnKey = false;
  	break;
  case 's':
  	backwardKey = false;
  	break;
  case 'd':
  	rightTurnKey = false;
  	break;
  case 'q':
  	strafeLeftKey = false;
  	break;
  case 'e':
  	strafeRightKey = false;
  	break;
  case 'x':
  	decendKey = false;
  	break;
  case ' ':
  	acendKey = false;
  	break;
  }
  	//dont know if this is needed.
    glutPostRedisplay();
}

static void keyboard(const unsigned char key, const int x, const int y) {
  switch (key) {
  case 27:
    exit(0);                                  // ESC
  case 'h':
    cout << " ============== H E L P ==============\n\n"
    << "=====How to move=====\n"
    << "w\t\tMove forward\n"
    << "s\t\tMove backward\n"
    << "a\t\tturn left\n"
    << "d\t\tturn right\n"
    << "x\t\tdecend\n"
    << "spacebar\taccend\n"
    << " \n"
    << " =====Other keyboard actions=====\n"
    << "h\t\thelp menu\n"
    << "c\t\tsave screenshot\n"
    << "o\t\tCycle object to manipulate\n"
    << "f\t\tCycle fragment shader\n"
    << "l\t\ttoggle shadows on and off\n"
    << "r\t\tshoot rockets\n"
    << "m\t\tchange the environment texture\n"
    << "v\t\tCycle camera views\n"
    << "t\t\treset explosions\n"
    << "+\t\tIncrease animation speed\n"
    << "-\t\tDecrease animation speed\n"
    << "drag left mouse to rotate selected object\n" 
    << "drag middle mouse to translate in/out selected object \n" 
    << "drag right mouse to translate up/down/left/right selected object\n" 
    << endl;
    break;
  //Movement Cases
  case 'w':
  	forwardKey = true;
  	break;
  case 'a':
  	leftTurnKey = true;
  	break;
  case 't':
    explodedHuh = explodedHuha1 = false;
    break;
  case 's':
  	backwardKey = true;
  	break;
  case 'd':
  	rightTurnKey = true;
  	break;
  case 'q':
  	strafeLeftKey = true;
  	break;
  case 'e':
  	strafeRightKey = true;
  	break;
  case 'x':
  	decendKey = true;
  	break;
  case ' ':
  	acendKey = true;
  	break;
  case 'r':
    shoot = true;
    break;

  //other functions

  case'l':
    if(shadowsOn == true){
      shadowsOn = false;
    }else{
      shadowsOn=true;
    }
    break;
  case 'c':
    glFlush();
    writePpmScreenshot(g_windowWidth, g_windowHeight, "out.ppm");
    cout << "Screenshot written to out.ppm." << endl;
    break;
  case 'o':
    g_objToManip = (g_objToManip +1) % numObjects;
      break;
  case '+':
    g_animSpeed *= 1.05;
    break;
  case 'v':
    cam = (cam+1) % numCameras;
      g_eyeRbt = cameraViews[cam];
    break;
  case 'm':
    if(selectedtexture > 5){
      selectedtexture = 5;}
      else{ if(selectedtexture < 2){
          selectedtexture = 6;
        }else{
          selectedtexture = 0;
      }}

    break;
  case '-':
    g_animSpeed *= 0.95;
    break;
  case 'k':
    if(explodedHuh == false){
      explodeMain();
    }else{
      explodedHuh = false;
    }
    break;
  case 'f':
    g_activeShader = (g_activeShader + 1) % g_numShaders;
    switch (g_activeShader) {
      case 0:
     cout << "Using solid shader." << endl;
      break;
      case 1:
     cout << "Using solid phong shader." << endl;
      break;
      case 2:
     cout << "Using Diffuse Texture Phong shader." << endl;
      break;
    }
    break;
  }
  glutPostRedisplay();
}



static void initGlutState(int argc, char * argv[]) {
  glutInit(&argc, argv);                                  // initialize Glut based on cmd-line args
  glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH);  //  RGBA pixel channels and double buffering
  glutInitWindowSize(g_windowWidth, g_windowHeight);      // create a window
  glutCreateWindow("Project 8: Final Project");    // title the window

  glutDisplayFunc(display);                               // display rendering callback
  glutReshapeFunc(reshape);                               // window reshape callback
  glutMotionFunc(motion);                                 // mouse movement callback
  glutMouseFunc(mouse);                                   // mouse click callback
  glutIdleFunc(idle);             // idle callback for animation
  glutKeyboardFunc(keyboard);
  glutKeyboardUpFunc(keyboardUp);
}

static void initGLState() {
  glClearColor(0.2, 0.2, .5, 1);
  glClearDepth(-0.9);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glCullFace(GL_BACK);
  //glEnable(GL_CULL_FACE); // Enable if you don't want to render back faces,
                            // but make sure it's disabled to show inside of tube.
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_GREATER);
  glReadBuffer(GL_BACK);
  glEnable(GL_BLEND); // Enable alpha blending
  GLfloat fcolor[4] = { .5, .5, .5, 1};
  
  glFogi(GL_FOG_MODE, GL_LINEAR);
  glFogfv(GL_FOG_COLOR, fcolor);
  glFogf(GL_FOG_DENSITY, .01);
  glFogf(GL_FOG_START, 1.0);
  glFogf(GL_FOG_END, 100);
  glHint(GL_FOG_HINT, GL_NICEST);
  glEnable(GL_FOG);



  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  if (!g_Gl2Compatible)
    glEnable(GL_FRAMEBUFFER_SRGB);
}
     
static void initShaders() {
  g_shaderStates.resize(g_numShaders);
  for (int i = 0; i < g_numShaders; ++i) {
    if (g_Gl2Compatible)
      g_shaderStates[i].reset(new ShaderState(g_shaderFilesGl2[i][0], g_shaderFilesGl2[i][1]));
    else
      g_shaderStates[i].reset(new ShaderState(g_shaderFiles[i][0], g_shaderFiles[i][1]));
  }
}

static void initGeometry() {
  initGround();
  initHeli();
  initSphere();
  initRocket();
  InitObjects();
}
static void loadTexture(GLuint texHandle, const char *ppmFilename) {
  int texWidth, texHeight;
  vector<PackedPixel> pixData;

  ppmRead(ppmFilename, texWidth, texHeight, pixData);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texHandle);
  glTexImage2D(GL_TEXTURE_2D, 0, g_Gl2Compatible ? GL_RGB : GL_SRGB, texWidth, texHeight,
               0, GL_RGB, GL_UNSIGNED_BYTE, &pixData[0]);
  checkGlErrors();
}
static void initTextures() {
  g_tex0.reset(new GlTexture());
  g_tex1.reset(new GlTexture());
  g_tex2.reset(new GlTexture());
  g_tex3.reset(new GlTexture());
  g_tex4.reset(new GlTexture());
  g_tex5.reset(new GlTexture());
  g_tex6.reset(new GlTexture());

  loadTexture(*g_tex0, "grass256.ppm");
  loadTexture(*g_tex1, "brick.ppm");
  loadTexture(*g_tex2, "marble256.ppm");
  loadTexture(*g_tex3, "sun.ppm");
  loadTexture(*g_tex4, "hull.ppm");
  loadTexture(*g_tex5, "water.ppm");
  loadTexture(*g_tex6, "wood.ppm");

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, *g_tex0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); // repeat ground texture
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, *g_tex1);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, *g_tex2);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, *g_tex3);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

  glActiveTexture(GL_TEXTURE4);
  glBindTexture(GL_TEXTURE_2D, *g_tex4);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

  glActiveTexture(GL_TEXTURE5);
  glBindTexture(GL_TEXTURE_2D, *g_tex5);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

  glActiveTexture(GL_TEXTURE6);
  glBindTexture(GL_TEXTURE_2D, *g_tex6);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
}

int main(int argc, char * argv[]) {
  try {
    initGlutState(argc,argv);

    glewInit(); // load the OpenGL extensions

    cout << (g_Gl2Compatible ? "Will use OpenGL 2.x / GLSL 1.0" : "Will use OpenGL 3.x / GLSL 1.3") << endl;
    if ((!g_Gl2Compatible) && !GLEW_VERSION_3_0)
      throw runtime_error("Error: card/driver does not support OpenGL Shading Language v1.3");
    else if (g_Gl2Compatible && !GLEW_VERSION_2_0)
      throw runtime_error("Error: card/driver does not support OpenGL Shading Language v1.0");

    initGLState();
    initShaders();
    initGeometry();
    initTextures();

    glutMainLoop();
    return 0;
  }
  catch (const runtime_error& e) {
    cout << "Exception caught: " << e.what() << endl;
    return -1;
  }
}


 // This was added by Lewis to work around a weird LD bug. Also add include file up top, and add link to makefile.. See 
 // http://stackoverflow.com/questions/20007961/error-running-a-compiled-c-file-uses-opengl-error-inconsistency-detected
void DUMMY_TO_FIX_LD_ERROR () {
  int i = pthread_getconcurrency();
}
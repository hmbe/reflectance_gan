#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/filesystem.h>
#include <learnopengl/shader.h>
#include <learnopengl/camera.h>
#include <learnopengl/model.h>

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <GL/glut.h>
#include <GL/glext.h>

#include <dirent.h>
#include <string>
#include <cstring>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <ctime>

#include<random>

#define GL_BGR_EXT 0x80E0

using namespace cv;
using namespace std;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);
void renderSphere();
void renderCube();
void renderQuad();

//LMG
void loadModel(Model ourModel);
void renderModel();

// settings
// LMG 1280 x 720 -> 512 x 512
const unsigned int SCR_WIDTH = 256;
const unsigned int SCR_HEIGHT = 256;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));

// LMG 800 x 600 -> 512 x 512
// useless
float lastX = 512.0f / 2.0;
float lastY = 512.0f / 2.0;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;


//LMG
void Capture_OpenGLViewPort(string filename)
{
    GLubyte * bits; //RGB bits
    GLint viewport[4]; //current viewport
        //get current viewport
    glGetIntegerv(GL_VIEWPORT, viewport);

        int rows = viewport[3];
        int cols = viewport[2];

    bits = new GLubyte[cols * 3 * rows];

        //read pixel from frame buffer
    glFinish(); //finish all commands of OpenGL
    glPixelStorei(GL_PACK_ALIGNMENT,1); //or glPixelStorei(GL_PACK_ALIGNMENT,4);
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    glPixelStorei(GL_PACK_SKIP_ROWS, 0);
    glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
    glReadPixels(0, 0, cols, rows, GL_BGR_EXT, GL_UNSIGNED_BYTE, bits);

    Mat cap_GL(rows, cols, CV_8UC3);

    for (int i = 0; i < cap_GL.rows; i++)
    {
        for (int j = 0; j < cap_GL.cols; j++)
        {
            cap_GL.at<Vec3b>(i, j)[0] = (uchar)(bits[(rows - i - 1) * 3 * cols + j * 3 + 0]);
            cap_GL.at<Vec3b>(i, j)[1] = (uchar)(bits[(rows - i - 1) * 3 * cols + j * 3 + 1]);
            cap_GL.at<Vec3b>(i, j)[2] = (uchar)(bits[(rows - i - 1) * 3 * cols + j * 3 + 2]);
        }
    }

    imwrite(filename , cap_GL);
    delete [] bits; 
}

float RandomFloat(float min, float max)
{
    // this  function assumes max > min, you may want 
    // more robust error checking for a non-debug build
    assert(max > min); 
    float random = ((float) rand()) / (float) RAND_MAX;

    // generate (in your case) a float between 0 and (4.5-.78)
    // then add .78, giving you a float between .78 and 4.5
    float range = max - min;  
    return (random*range) + min;
}


int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

    // glfw window creation
    // --------------------
    // SCR_WIDTH AND SCR_HEIGHT mean the size of windows.
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    glfwMakeContextCurrent(window);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    } // GLFW init

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    
    /*
    //LMG
    //disable mouse callback
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    */

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);
    // set depth function to less than AND equal for skybox depth trick.
    glDepthFunc(GL_LEQUAL);
    // enable seamless cubemap sampling for lower mip levels in the pre-filter map.
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    // build and compile shaders
    // -------------------------
    Shader pbrShader("pbr.vs", "pbr.fs");
    Shader equirectangularToCubemapShader("cubemap.vs", "equirectangular_to_cubemap.fs");
    Shader irradianceShader("cubemap.vs", "irradiance_convolution.fs");
    Shader prefilterShader("cubemap.vs", "prefilter.fs");
    Shader brdfShader("brdf.vs", "brdf.fs");
    Shader backgroundShader("background.vs", "background.fs");

    //LMG
    Shader normalShader("normal.vs", "normal.fs");


    pbrShader.use();
    pbrShader.setInt("irradianceMap", 0);
    pbrShader.setInt("prefilterMap", 1);
    pbrShader.setInt("brdfLUT", 2);

    pbrShader.setVec3("albedo", 0.5f, 0.0f, 0.0f);
    pbrShader.setFloat("ao", 1.0f);

    backgroundShader.use();
    backgroundShader.setInt("environmentMap", 0);

  
    // lights
    // ------
    
    glm::vec3 lightPositions[] = {
        glm::vec3(-10.0f,  10.0f, 10.0f),
        glm::vec3( 10.0f,  10.0f, 10.0f),
        glm::vec3(-10.0f, -10.0f, 10.0f),
        glm::vec3( 10.0f, -10.0f, 10.0f),
    };
    glm::vec3 lightColors[] = {
        glm::vec3(300.0f, 300.0f, 300.0f),
        glm::vec3(300.0f, 300.0f, 300.0f),
        glm::vec3(300.0f, 300.0f, 300.0f),
        glm::vec3(300.0f, 300.0f, 300.0f)
    };
    int nrRows = 7;
    int nrColumns = 7;
    float spacing = 2.5;



    //LMG
        
    //string obj_path = "/media/minki/Local_Disk/LearnOpenGL-master/resources/meshes_v1/bottle/";
    //string hdri_path = "/media/minki/Local_Disk/LearnOpenGL-master/resources/hdri/";

    //string obj_path = "/media/minki/Local_Disk/LearnOpenGL-master/resources/new3/obj/";
    //string hdri_path = "/media/minki/Local_Disk/LearnOpenGL-master/resources/new3/hdri/";

    string obj_path = "/media/minki/Local_Disk/LearnOpenGL-master/resources/1101/test_obj/";
    string hdri_path = "/media/minki/Local_Disk/LearnOpenGL-master/resources/1101/test_hdri/";

    
    DIR *dir;
    struct dirent *ent;

    vector<string> obj_paths;
    vector<string> obj_names;
    vector<string> hdri_paths;
    vector<string> hdri_names;

    // set obj path
    if ((dir = opendir (obj_path.c_str())) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            string file_name = ent->d_name;
            if(ssize_t pos = file_name.find(".obj") != string::npos){
                obj_names.push_back(file_name);
                obj_paths.push_back(obj_path + file_name);
            }   
            //printf ("%s\n", ent->d_name);
        }
        closedir (dir);
    } else {
        /* could not open directory */
        perror ("");
        return EXIT_FAILURE;
    }

    // set hdri path
    if ((dir = opendir (hdri_path.c_str())) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            string file_name = ent->d_name;
            if(ssize_t pos = file_name.find(".hdr") != string::npos){
                hdri_names.push_back(file_name);
                hdri_paths.push_back(hdri_path + file_name);
            }
            //printf ("%s\n", ent->d_name);
        }
        closedir (dir);
    } else {
        /* could not open directory */
        perror ("");
        return EXIT_FAILURE;
    }

    std::ofstream out("param.txt");

    if ( !(out.is_open()) ) {
        std::cout << "param.txt is not opened" << endl;
        return -1;
    }

    // render loop
    // -----------
    unsigned int captureFBO;
    unsigned int captureRBO;
    glGenFramebuffers(1, &captureFBO);
    glGenRenderbuffers(1, &captureRBO);

    //LMG
    unsigned int normalFBO;
    unsigned int normalRBO;
    glGenFramebuffers(1, &normalFBO);

    //int epoch = 0;
    //search glGen -> change codes

    srand(time(NULL));
    default_random_engine generator;
    normal_distribution<double> distribution(0.0f, 0.2f);

    int start_epoch = 0;
    int end_epoch = 50;
    for(int epoch = start_epoch; epoch < end_epoch; epoch++){

        //LMG
        for(int j=0; j<hdri_paths.size(); j++){

            // pbr: setup framebuffer
            // ----------------------
            
            glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
            glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 512, 512);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, captureRBO);

            // pbr: load the HDR environment map
            // ---------------------------------
            stbi_set_flip_vertically_on_load(true);
            int width, height, nrComponents;
            //float *data = stbi_loadf(FileSystem::getPath("resources/textures/hdr/newport_loft.hdr").c_str(), &width, &height, &nrComponents, 0);
            float *data = stbi_loadf((hdri_paths[j]).c_str(), &width, &height, &nrComponents, 0);

            unsigned int hdrTexture;
            if (data)
            {
                glGenTextures(1, &hdrTexture);
                glBindTexture(GL_TEXTURE_2D, hdrTexture);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, data); // note how we specify the texture's data value to be float

                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

                stbi_image_free(data);
            }
            else
            {
                std::cout << "Failed to load HDR image." << std::endl;
            }

            // pbr: setup cubemap to render to and attach to framebuffer
            // ---------------------------------------------------------
            unsigned int envCubemap;
            glGenTextures(1, &envCubemap);
            glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
            for (unsigned int i = 0; i < 6; ++i)
            {
                //LMG -> change cube map texture size 512 to 256
                glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 512, 512, 0, GL_RGB, GL_FLOAT, nullptr);
            }
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); // enable pre-filter mipmap sampling (combatting visible dots artifact)
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            // pbr: set up projection and view matrices for capturing data onto the 6 cubemap face directions
            // ----------------------------------------------------------------------------------------------
            glm::mat4 captureProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
            glm::mat4 captureViews[] =
            {
                glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
                glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
                glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
                glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
                glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
                glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
            };

            // pbr: convert HDR equirectangular environment map to cubemap equivalent
            // ----------------------------------------------------------------------
            equirectangularToCubemapShader.use();
            equirectangularToCubemapShader.setInt("equirectangularMap", 0);
            equirectangularToCubemapShader.setMat4("projection", captureProjection);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, hdrTexture);

            // LMG -> change viewport size from 512 to 256
            // change this with upper LMG
            glViewport(0, 0, 512, 512); // don't forget to configure the viewport to the capture dimensions.
            glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
            for (unsigned int i = 0; i < 6; ++i)
            {
                equirectangularToCubemapShader.setMat4("view", captureViews[i]);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, envCubemap, 0);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                renderCube();
            }
            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            // then let OpenGL generate mipmaps from first mip face (combatting visible dots artifact)
            glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
            glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

            // pbr: create an irradiance cubemap, and re-scale capture FBO to irradiance scale.
            // --------------------------------------------------------------------------------
            unsigned int irradianceMap;
            glGenTextures(1, &irradianceMap);
            glBindTexture(GL_TEXTURE_CUBE_MAP, irradianceMap);
            for (unsigned int i = 0; i < 6; ++i)
            {
                glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 32, 32, 0, GL_RGB, GL_FLOAT, nullptr);
            }
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
            glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 32, 32);

            // pbr: solve diffuse integral by convolution to create an irradiance (cube)map.
            // -----------------------------------------------------------------------------
            irradianceShader.use();
            irradianceShader.setInt("environmentMap", 0);
            irradianceShader.setMat4("projection", captureProjection);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);

            glViewport(0, 0, 32, 32); // don't forget to configure the viewport to the capture dimensions.
            glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
            for (unsigned int i = 0; i < 6; ++i)
            {
                irradianceShader.setMat4("view", captureViews[i]);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, irradianceMap, 0);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                renderCube();
            }
            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            // pbr: create a pre-filter cubemap, and re-scale capture FBO to pre-filter scale.
            // --------------------------------------------------------------------------------
            unsigned int prefilterMap;
            glGenTextures(1, &prefilterMap);
            glBindTexture(GL_TEXTURE_CUBE_MAP, prefilterMap);
            for (unsigned int i = 0; i < 6; ++i)
            {
                glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 128, 128, 0, GL_RGB, GL_FLOAT, nullptr);
            }
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); // be sure to set minifcation filter to mip_linear 
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            // generate mipmaps for the cubemap so OpenGL automatically allocates the required memory.
            glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

            // pbr: run a quasi monte-carlo simulation on the environment lighting to create a prefilter (cube)map.
            // ----------------------------------------------------------------------------------------------------
            prefilterShader.use();
            prefilterShader.setInt("environmentMap", 0);
            prefilterShader.setMat4("projection", captureProjection);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);

            glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
            unsigned int maxMipLevels = 5;
            for (unsigned int mip = 0; mip < maxMipLevels; ++mip)
            {
                // resize framebuffer according to mip-level size.
                unsigned int mipWidth  = 128 * std::pow(0.5, mip);
                unsigned int mipHeight = 128 * std::pow(0.5, mip);
                glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
                glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, mipWidth, mipHeight);
                glViewport(0, 0, mipWidth, mipHeight);

                float roughness = (float)mip / (float)(maxMipLevels - 1);
                prefilterShader.setFloat("roughness", roughness);
                for (unsigned int i = 0; i < 6; ++i)
                {
                    prefilterShader.setMat4("view", captureViews[i]);
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, prefilterMap, mip);

                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                    renderCube();
                }
            }
            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            // pbr: generate a 2D LUT from the BRDF equations used.
            // ----------------------------------------------------
            unsigned int brdfLUTTexture;
            glGenTextures(1, &brdfLUTTexture);

            // pre-allocate enough memory for the LUT texture.
            glBindTexture(GL_TEXTURE_2D, brdfLUTTexture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, 512, 512, 0, GL_RG, GL_FLOAT, 0);
            // be sure to set wrapping mode to GL_CLAMP_TO_EDGE
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            // then re-configure capture framebuffer object and render screen-space quad with BRDF shader.
            glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
            glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 512, 512);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brdfLUTTexture, 0);

            glViewport(0, 0, 512, 512);
            brdfShader.use();
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            renderQuad();

            glBindFramebuffer(GL_FRAMEBUFFER, 0);


            // initialize static shader uniforms before rendering
            // --------------------------------------------------
            glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
            pbrShader.use();
            pbrShader.setMat4("projection", projection);
            backgroundShader.use();
            backgroundShader.setMat4("projection", projection);

            // then before rendering, configure the viewport to the original framebuffer's screen dimensions
            int scrWidth, scrHeight;
            glfwGetFramebufferSize(window, &scrWidth, &scrHeight);
            glViewport(0, 0, scrWidth, scrHeight);

            //set distance..
            // tan 45 = 1.61977519054(camera zoom)
            // under line / dist = tan 45

            // 0.55 * x / 1.414 = 1.619775 .... x = 4.1643
            // x * 4.1643 / dist = 1.619775 .... 0.389 * dist = value
            // 1.2 to 1.5

            // epoch, file_name, hdri_name, albedo(F0)(r, g, b), roughness, metallic, distance, angle_s, angle_t, cam_coord(x, y, z)

            const float PI = 3.14159265359;

            for(int i=0; i<obj_paths.size(); i++){
                
                //srand((unsigned int)time(NULL));
                //

                float distance = RandomFloat(1.2f, 1.5f);
                float angle_s = RandomFloat(0.0f, 2*PI);
                float angle_t = RandomFloat(0.0f, 2*PI);

                //(r * math.cos(s) * math.sin(t), r * math.sin(s) * math.sin(t), r * math.cos(t))
                vector<float> cam_coord;
                cam_coord.push_back(distance * cos(angle_s) * sin(angle_t));
                cam_coord.push_back(abs(distance * sin(angle_s) * sin(angle_t)));
                // for upper side rendering
                cam_coord.push_back(distance * cos(angle_t));

                // if metallic==1.0, diffuse param replace F0 value
                vector<float> albedo_param;
                for(int k =0; k<3; k++){
                    albedo_param.push_back( RandomFloat(0.0f, 1.0f) );
                }

                float metallic_param = 1.0f;

                //normal distribution, 0, 0.2
                float roughness_param = distribution(generator);
                roughness_param = abs(roughness_param);
                if(roughness_param > 1)
                    roughness_param = 1;
                //float roughness_param = RandomFloat(0.0f, 0.5f);
                
                // set param manually(12/13)
                // distance, angle_s, angle_t, cam_coord[3], albedo_param[3], roughness_param


                // per-frame time logic
                // --------------------
                float currentFrame = glfwGetTime();
                deltaTime = currentFrame - lastFrame;
                lastFrame = currentFrame;

                // input
                // -----
                processInput(window);

                // render
                // ------
                glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                //LMG
                //camera test
                //camera.Position = glm::vec3(0.0, 2.0, 0.1);
                camera.Position = glm::vec3(cam_coord[0], cam_coord[1], cam_coord[2]);
                camera.Front = -(camera.Position - glm::vec3(0.0,0.0,0.0));

                // render scene, supplying the convoluted irradiance map to the final shader.
                // ------------------------------------------------------------------------------------------
                pbrShader.use();
                glm::mat4 view = camera.GetViewMatrix();
                pbrShader.setMat4("view", view);
                pbrShader.setVec3("camPos", camera.Position);
                
                // bind pre-computed IBL data
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_CUBE_MAP, irradianceMap);
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_CUBE_MAP, prefilterMap);
                glActiveTexture(GL_TEXTURE2);
                glBindTexture(GL_TEXTURE_2D, brdfLUTTexture);

                // LMG
                // render the loaded model
                //Model ourModel(FileSystem::getPath("resources/objects/nanosuit/nanosuit.obj"));
                //Model ourModel(FileSystem::getPath("resources/meshes_v1/bottle/08.obj"));

                float** ior_list;
                ior_list = (float**)malloc(sizeof(float*) * 9);
                for(int iln=0; iln < 9; iln++){
                    ior_list[iln] = (float*)malloc(sizeof(float) * 3);
                }

                float** k_list;
                k_list = (float**)malloc(sizeof(float*) * 9);
                for(int kln=0; kln < 9; kln++){
                    k_list[kln] = (float*)malloc(sizeof(float) * 3);
                }

                std::cout << epoch << endl;
                std::cout << obj_paths[i] << endl;
                std::cout << hdri_paths[j] << endl;

                Model ourModel(obj_paths[i]);

                glm::mat4 model = glm::mat4(1.0f);

                pbrShader.setVec3("albedo", albedo_param[0], albedo_param[1], albedo_param[2]);
                pbrShader.setFloat("ao", 1.0f);
                pbrShader.setFloat("metallic", metallic_param);
                pbrShader.setFloat("roughness", roughness_param);

                pbrShader.setMat4("model", model);
                
                renderSphere();
                //call it only once
                //loadModel(ourModel);
                //temporary dont use params
                //renderModel();


                // render skybox (render as last to prevent overdraw)
                backgroundShader.use();
                backgroundShader.setMat4("view", view);
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
                //glBindTexture(GL_TEXTURE_CUBE_MAP, irradianceMap); // display irradiance map
                //glBindTexture(GL_TEXTURE_CUBE_MAP, prefilterMap); // display prefilter map


                //renderCube();


                // render BRDF map to screen
                //brdfShader.Use();
                //renderQuad();
                
                //LMG
                //string filename =  "_" + std::to_string(epoch) + "_" + obj_names[i].substr(0, obj_names[i].length() - 4) + "_" + hdri_names[j].substr(0, hdri_names[j].length() - 4) + string(".png");
                //Capture_OpenGLViewPort(string("./input/") + filename);
                Capture_OpenGLViewPort(string("BRDFTEST.png"));

                // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
                // -------------------------------------------------------------------------------
                glfwSwapBuffers(window); // swap current buffer -> update
                glfwPollEvents();


                //LMG
                //for normal rendering
                glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

           
                normalShader.use();

                normalShader.setMat4("model", model);
                normalShader.setMat4("view", view);
                normalShader.setMat4("projection", projection);
                normalShader.setVec3("camPos", camera.Position);
                
                //no sphere normal 
                loadModel(ourModel);
                //renderModel();

                //render sphere normal
                /*
                //glm::mat4 sphere_model = glm::mat4(0.380f * distance);
                glm::mat4 sphere_model_tmp = glm::mat4(distance);
                sphere_model_tmp = glm::scale(sphere_model_tmp, glm::vec3(1.0f, 1.0f, 1.0f));	// it's a bit too big for our scene, so scale it down
                normalShader.setMat4("model", sphere_model_tmp);
                renderSphere();
                Capture_OpenGLViewPort(string("./sphere_normal.png"));
                */

                //LMG
                string normal_filename = "_" + std::to_string(epoch) + "_" + obj_names[i].substr(0, obj_names[i].length() - 4) + "_" + hdri_names[j].substr(0, hdri_names[j].length() - 4) +string("_normal") + string(".png");
                Capture_OpenGLViewPort(string("./normal/") + normal_filename);

                // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
                // -------------------------------------------------------------------------------
                glfwSwapBuffers(window); // swap current buffer -> update
                glfwPollEvents();


                //LMG
                //render Sphere(reflectance map)
                glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                
                pbrShader.use();

                //scale sphere
                //glm::mat4 sphere_model = glm::mat4(0.380f * distance);
                glm::mat4 sphere_model = glm::mat4(distance);

                sphere_model = glm::scale(sphere_model, glm::vec3(1.0f, 1.0f, 1.0f));
                pbrShader.setMat4("model", sphere_model);

                // bind pre-computed IBL data
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_CUBE_MAP, irradianceMap);
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_CUBE_MAP, prefilterMap);
                glActiveTexture(GL_TEXTURE2);
                glBindTexture(GL_TEXTURE_2D, brdfLUTTexture);
                
                renderSphere();
                
                //LMG
                string sphere_filename =  "_" + std::to_string(epoch) + "_" + obj_names[i].substr(0, obj_names[i].length() - 4) + "_" + hdri_names[j].substr(0, hdri_names[j].length() - 4) +string("_sphere")+ string(".png");
                Capture_OpenGLViewPort(string("./sphere/") + sphere_filename);


                // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
                // -------------------------------------------------------------------------------
                glfwSwapBuffers(window); // swap current buffer -> update
                glfwPollEvents();

                glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                //render envmap
                pbrShader.setVec3("albedo", 1.0f, 1.0f, 1.0f);
                pbrShader.setFloat("ao", 1.0f);
                pbrShader.setFloat("metallic", 1.0f);
                pbrShader.setFloat("roughness", 0.0f);
                renderSphere();

                //LMG
                string env_filename =  "_" + std::to_string(epoch) + "_" + obj_names[i].substr(0, obj_names[i].length() - 4) + "_" + hdri_names[j].substr(0, hdri_names[j].length() - 4) +string("_env")+ string(".png");
                Capture_OpenGLViewPort(string("./env/") + env_filename);

                // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
                // -------------------------------------------------------------------------------
                glfwSwapBuffers(window); // swap current buffer -> update
                glfwPollEvents();

                


                // epoch, file_name, hdri_name, albedo(F0)(r, g, b), roughness, metallic, distance, angle_s, angle_t, cam_coord(x, y, z)
                out << epoch << " " \
                    << obj_names[i].substr(0, obj_names[i].length() - 4) << " " \
                    << hdri_names[j].substr(0, hdri_names[j].length() - 4) << " " \
                    << albedo_param[0] << " " << albedo_param[1] << " " << albedo_param[2] << " " \
                    << roughness_param << " " \
                    << metallic_param << " " \
                    << distance << " " \
                    << angle_s << " " << angle_t << " " \
                    << cam_coord[0] << " " << cam_coord[1] << " " << cam_coord[2] << endl;

            }
            // hdrTexture envCubemap irradianceMap prefilterMap brdfLUTTexture
            glDeleteTextures(1, &hdrTexture);
            glDeleteTextures(1, &envCubemap);
            glDeleteTextures(1, &irradianceMap);
            glDeleteTextures(1, &prefilterMap);
            glDeleteTextures(1, &brdfLUTTexture);
        }
    }
    

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    float cameraSpeed = 2.5 * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}


// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(yoffset);
}

// renders (and builds at first invocation) a sphere
// -------------------------------------------------

// declare VAO -> Vertex Array Object
unsigned int sphereVAO = 0;
unsigned int indexCount;
void renderSphere()
{
    // scale sphere
    float scale = 0.380f;
    //float scale = 0.2f;

    // if Array is not binded
    if (sphereVAO == 0)
    {
        // VAO binding -> manage attribute pointers
        glGenVertexArrays(1, &sphereVAO);
        // Vertex Buffer Object VBO

        unsigned int vbo, ebo;
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ebo);

        std::vector<glm::vec3> positions;
        std::vector<glm::vec2> uv;
        std::vector<glm::vec3> normals;
        std::vector<unsigned int> indices;

        
        //64
        const unsigned int X_SEGMENTS = 64;
        const unsigned int Y_SEGMENTS = 64;
        const float PI = 3.14159265359;
        for (unsigned int y = 0; y <= Y_SEGMENTS; ++y)
        {
            for (unsigned int x = 0; x <= X_SEGMENTS; ++x)
            {
                float xSegment = (float)x / (float)X_SEGMENTS;
                float ySegment = (float)y / (float)Y_SEGMENTS;
                float xPos = std::cos(xSegment * 2.0f * PI) * std::sin(ySegment * PI) * scale;
                float yPos = std::cos(ySegment * PI) * scale;
                float zPos = std::sin(xSegment * 2.0f * PI) * std::sin(ySegment * PI) * scale;

                positions.push_back(glm::vec3(xPos, yPos, zPos));
                uv.push_back(glm::vec2(xSegment, ySegment));
                normals.push_back(glm::vec3(xPos, yPos, zPos));
            }
        }

        bool oddRow = false;
        for (int y = 0; y < Y_SEGMENTS; ++y)
        {
            if (!oddRow) // even rows: y == 0, y == 2; and so on
            {
                for (int x = 0; x <= X_SEGMENTS; ++x)
                {
                    indices.push_back(y       * (X_SEGMENTS + 1) + x);
                    indices.push_back((y + 1) * (X_SEGMENTS + 1) + x);
                }
            }
            else
            {
                for (int x = X_SEGMENTS; x >= 0; --x)
                {
                    indices.push_back((y + 1) * (X_SEGMENTS + 1) + x);
                    indices.push_back(y       * (X_SEGMENTS + 1) + x);
                }
            }
            oddRow = !oddRow;
        }
        indexCount = indices.size();

        // data-> 3 positions, 2 uvs, 3 normals
        std::vector<float> data;
        for (int i = 0; i < positions.size(); ++i)
        {
            data.push_back(positions[i].x);
            data.push_back(positions[i].y);
            data.push_back(positions[i].z);
            if (uv.size() > 0)
            {
                data.push_back(uv[i].x);
                data.push_back(uv[i].y);
            }
            if (normals.size() > 0)
            {
                data.push_back(normals[i].x);
                data.push_back(normals[i].y);
                data.push_back(normals[i].z);
            }
        }
        glBindVertexArray(sphereVAO);
        // second step -> binding
        // bind buffer to type

        // data
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        // third step -> copy vertex data
        // currently binded buffer type, data size, data, manage type
        glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), &data[0], GL_STATIC_DRAW);
        
        // indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);
        // link vertex attributes
        // use stride to designate values
        float stride = (3 + 2 + 3) * sizeof(float);

        glEnableVertexAttribArray(0);
        // location of variables in vertex shader, size of vertex(ex. vec3), data type, normalize, stride for attributes, void type offset of start location
        // location=0(3), location=1(2), location=2(3), location=0(3), ..... repeat!
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
        
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
        
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, (void*)(5 * sizeof(float)));
    }

    glBindVertexArray(sphereVAO);
    glDrawElements(GL_TRIANGLE_STRIP, indexCount, GL_UNSIGNED_INT, 0);

}

// renderCube() renders a 1x1 3D cube in NDC.
// -------------------------------------------------
unsigned int cubeVAO = 0;
unsigned int cubeVBO = 0;
void renderCube()
{
    // initialize (if necessary)
    if (cubeVAO == 0)
    {
        float vertices[] = {
            // back face
            -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
             1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
             1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 0.0f, // bottom-right         
             1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
            -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
            -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 1.0f, // top-left
            // front face
            -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
             1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 0.0f, // bottom-right
             1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
             1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
            -1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f, // top-left
            -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
            // left face
            -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
            -1.0f,  1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-left
            -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
            -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
            -1.0f, -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-right
            -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
            // right face
             1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
             1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
             1.0f,  1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-right         
             1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
             1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
             1.0f, -1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-left     
            // bottom face
            -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
             1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 1.0f, // top-left
             1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
             1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
            -1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f, // bottom-right
            -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
            // top face
            -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
             1.0f,  1.0f , 1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
             1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 1.0f, // top-right     
             1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
            -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
            -1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 0.0f  // bottom-left        
        };
        glGenVertexArrays(1, &cubeVAO);
        glGenBuffers(1, &cubeVBO);
        // fill buffer
        glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        // link vertex attributes
        glBindVertexArray(cubeVAO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
    // render Cube
    glBindVertexArray(cubeVAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);
}

// renderQuad() renders a 1x1 XY quad in NDC
// -----------------------------------------
unsigned int quadVAO = 0;
unsigned int quadVBO;
void renderQuad()
{
    if (quadVAO == 0)
    {
        float quadVertices[] = {
            // positions        // texture Coords
            -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
             1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
             1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
        };
        // setup plane VAO
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    }
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}

// under construction
// declare VAO -> Vertex Array Object

unsigned int modelVAO = 0;
unsigned int indexCount_model = 0;

void loadModel(Model model){
    int vertice_num = 0;
    int uv_num = 0;
    int normal_num = 0;

    // if Array is not binded
    // VAO binding -> manage attribute pointers
    glGenVertexArrays(1, &modelVAO);
    // Vertex Buffer Object VBO

    unsigned int vbo, ebo;
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    std::vector<glm::vec3> positions;
    std::vector<glm::vec2> uv;
    std::vector<glm::vec3> normals;
    
    std::vector<unsigned int> indices;

    //cout << "model's mesh size is " << model.meshes.size() << endl;
    //cout << "model's vertices size is " << model.meshes[0].vertices.size() << endl;
    //cout << "model's indices size is " << model.meshes[0].indices.size() << endl;

    // make data
    std::vector<float> data;

    for (int m=0; m< model.meshes.size(); m++)
    {
        // model.meshes[i].vertices[i].
        // data-> 3 positions, 2 uvs, 3 normals

        for (int i=0; i < model.meshes[m].indices.size(); ++i){
            indices.push_back(model.meshes[m].indices[i]);
        }

        indexCount_model = indices.size();

        for (int i = 0; i < model.meshes[m].vertices.size(); ++i)
        {
            data.push_back(model.meshes[m].vertices[i].Position.x);
            data.push_back(model.meshes[m].vertices[i].Position.y);
            data.push_back(model.meshes[m].vertices[i].Position.z);

            if (&(model.meshes[m].vertices[i].TexCoords) != NULL)
            {
                data.push_back(model.meshes[m].vertices[i].TexCoords.x);
                data.push_back(model.meshes[m].vertices[i].TexCoords.y);
                uv_num++;
            }
            if (&(model.meshes[m].vertices[i].Normal) != NULL)
            {
                data.push_back(model.meshes[m].vertices[i].Normal.x);
                data.push_back(model.meshes[m].vertices[i].Normal.y);
                data.push_back(model.meshes[m].vertices[i].Normal.z);
                normal_num++;
            }
            vertice_num++;
        }
    }

    
    //cout << "model's vertice_num is " << vertice_num << endl;
    //cout << "model's normal_num is " << normal_num << endl;
    //cout << "model's uv_num is " << uv_num << endl;

    // vao 바인딩 한 다음 ebo 바인딩.. vao 내부에 ebo 호출 내용 저장
    glBindVertexArray(modelVAO);
    // second step -> binding
    // bind buffer to type

    // data
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    // third step -> copy vertex data
    // currently binded buffer type, data size, data, manage type
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), &data[0], GL_STATIC_DRAW);
    
    // test -> no indices
    // indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);
    
    // link vertex attributes
    // use stride to designate values
    float stride = (3 + 2 + 3) * sizeof(float);

    glEnableVertexAttribArray(0);
    // location of variables in vertex shader, size of vertex(ex. vec3), data type, normalize, stride for attributes, void type offset of start location
    // connect attribute pointer with vbo
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
    
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
    
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, (void*)(5 * sizeof(float)));

    // for renderModel..
    glBindVertexArray(modelVAO);
    glDrawElements(GL_TRIANGLES, indexCount_model, GL_UNSIGNED_INT, 0);

    // necessary!!
    // delete vbo, ebo, modelVAO

    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &modelVAO);
}


void renderModel()
{
    // must call loadModel first!!

    //v1
    //draw object with ebo
    
    glBindVertexArray(modelVAO);
    glDrawElements(GL_TRIANGLES, indexCount_model, GL_UNSIGNED_INT, 0);

    //v2
    /*
    glBindVertexArray(modelVAO);
    glDrawArrays(GL_TRIANGLES, 0, vertice_num);
    //glBindVertexArray(0);
    */
}
#ifndef	SHADER_H
#define SHADER_H

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <GL/glew.h>
#ifdef __MAC__
#	include <GLUT/glut.h>
#else
#	include <GL/glut.h>
#endif



static void safe_glUseProgram(const GLuint program)
{
	if (GLEW_VERSION_2_0) glUseProgram(program); else glUseProgramObjectARB(program);
}

static void safe_glActiveTexture(const GLint n)
{
	if (GLEW_VERSION_2_0) glActiveTexture(n); else glActiveTextureARB(n);
}

static GLint safe_glGetUniformLocation(const GLuint program, const char varname[])
{
	GLint r;
	if (GLEW_VERSION_2_0) r = glGetUniformLocation(program, varname); else r = glGetUniformLocationARB(program, varname);
	if (r < 0) std::cerr << "Warning: cannot bind uniform variable " << varname << " (either doesn't exist or has been optimized away). glUniform calls for it won't have any effect.\n";
	return r;
}
static GLint safe_glGetAttribLocation(const GLuint program, const char varname[])
{
	GLint r;
	if (GLEW_VERSION_2_0) r = glGetAttribLocation(program, varname); else r = glGetAttribLocationARB(program, varname);
	if (r < 0) std::cerr << "Warning: cannot bind attribute variable " << varname << " (either doesn't exist or has been optimized away). glAttrib calls for it won't have any effect.\n";
	return r;
}

static void safe_glUniformMatrix4fv(const GLint varhandle, const GLfloat data[])
{
	if (varhandle < 0) return;
	if (GLEW_VERSION_2_0) glUniformMatrix4fv(varhandle, 1, GL_FALSE, data); else glUniformMatrix4fvARB(varhandle, 1, GL_FALSE, data);
}
static void safe_glUniform1i(const GLint varhandle, const GLint a)
{
	if (varhandle < 0) return;
	if (GLEW_VERSION_2_0) glUniform1i(varhandle, a); else glUniform1iARB(varhandle, a);
}
static void safe_glUniform2i(const GLint varhandle, const GLint a, const GLint b)
{
	if (varhandle < 0) return;
	if (GLEW_VERSION_2_0) glUniform2i(varhandle, a, b); else glUniform2iARB(varhandle, a, b);
}
static void safe_glUniform3i(const GLint varhandle, const GLint a, const GLint b, const GLint c)
{
	if (varhandle < 0) return;
	if (GLEW_VERSION_2_0) glUniform3i(varhandle, a, b, c); else glUniform3iARB(varhandle, a, b, c);
}
static void safe_glUniform4i(const GLint varhandle, const GLint a, const GLint b, const GLint c, const GLint d)
{
	if (varhandle < 0) return;
	if (GLEW_VERSION_2_0) glUniform4i(varhandle, a, b, c, d); else glUniform4iARB(varhandle, a, b, c, d);
}
static void safe_glUniform1f(const GLint varhandle, const GLfloat a)
{
	if (varhandle < 0) return;
	if (GLEW_VERSION_2_0) glUniform1f(varhandle, a); else glUniform1fARB(varhandle, a);
}
static void safe_glUniform2f(const GLint varhandle, const GLfloat a, const GLfloat b)
{
	if (varhandle < 0) return;
	if (GLEW_VERSION_2_0) glUniform2f(varhandle, a, b); else glUniform2fARB(varhandle, a, b);
}
static void safe_glUniform3f(const GLint varhandle, const GLfloat a, const GLfloat b, const GLfloat c)
{
	if (varhandle < 0) return;
	if (GLEW_VERSION_2_0) glUniform3f(varhandle, a, b, c); else glUniform3fARB(varhandle, a, b, c);
}
static void safe_glUniform4f(const GLint varhandle, const GLfloat a, const GLfloat b, const GLfloat c, const GLfloat d)
{
	if (varhandle < 0) return;
	if (GLEW_VERSION_2_0) glUniform4f(varhandle, a, b, c, d); else glUniform4fARB(varhandle, a, b, c, d);
}
static void safe_glVertexAttrib1f(const GLint varhandle, const GLfloat a)
{
	if (varhandle < 0) return;
	if (GLEW_VERSION_2_0) glVertexAttrib1f(varhandle, a); else glVertexAttrib1fARB(varhandle, a);
}
static void safe_glVertexAttrib2f(const GLint varhandle, const GLfloat a, const GLfloat b)
{
	if (varhandle < 0) return;
	if (GLEW_VERSION_2_0) glVertexAttrib2f(varhandle, a, b); else glVertexAttrib2fARB(varhandle, a, b);
}
static void safe_glVertexAttrib3f(const GLint varhandle, const GLfloat a, const GLfloat b, const GLfloat c)
{
	if (varhandle < 0) return;
	if (GLEW_VERSION_2_0) glVertexAttrib3f(varhandle, a, b, c); else glVertexAttrib3fARB(varhandle, a, b, c);
}
static void safe_glVertexAttrib4f(const GLint varhandle, const GLfloat a, const GLfloat b, const GLfloat c, const GLfloat d)
{
	if (varhandle < 0) return;
	if (GLEW_VERSION_2_0) glVertexAttrib4f(varhandle, a, b, c, d); else glVertexAttrib4fARB(varhandle, a, b, c, d);
}




/* ------------------------------------------------------------------- */

static int getFileLength(FILE * const fp)
{
	int r(0);
	std::rewind(fp);
	while(std::fgetc(fp) != EOF) ++r;
	std::rewind(fp);
	return r;
}


static int GetShaderSize(const char *filename)
{
	int count = -1;
	FILE * const fd = std::fopen(filename, "r");
	if (fd)
    {
    	count = getFileLength(fd)+1;
    	std::fclose(fd);
    }
    return count;
}

static void LoadShader(const char *filename, int shader_size, GLchar *shader)
{
	FILE * const fd = std::fopen(filename, "r");
	if (!fd) { std::cerr << "Could not open shader file " << filename << ". Exiting...\n"; assert(0); }  
	std::fseek(fd, 0, SEEK_SET);
	const int count = (int)std::fread(shader, 1, shader_size, fd);
	shader[count] = '\0';
	if (std::ferror(fd)) { std::cerr << "Error reading file " << filename << ". Exiting...\n"; assert(0); }
	std::fclose(fd);
}

static void printInfoLog(GLuint obj, const char * filename)
{
	GLint infologLength = 0;
  	GLint charsWritten  = 0;
  	char *infoLog;
  	glGetObjectParameterivARB(obj, GL_OBJECT_INFO_LOG_LENGTH_ARB, &infologLength);
  	if (infologLength <= 0) return;
	infoLog = new char[infologLength];
  	glGetInfoLogARB(obj, infologLength, &charsWritten, infoLog);
  	std::cerr << "\nMsg [" << filename << "]:\n--------------\n\n" << infoLog;
  	delete [] infoLog;
}

static void LoadShaderFromFile(const char *filename, GLchar **shader)
{
	const int shader_size = GetShaderSize(filename);
	if (shader_size < 0) { std::cerr << "Shader " << filename << " not found. Exiting...\n"; assert(0); }
	LoadShader(filename, shader_size, *shader = (GLchar *)malloc(shader_size));
}

static bool ShaderInit(const char *vertexfile, const char *fragfile, GLuint * const h_program)
{
	GLchar *vertex_shader		= 0;	/// vertex shader
	GLchar *frag_shader			= 0;	/// fragment (pixel) shader
	GLuint	h_vertex_shader;			/// handle to the vertex shader
	GLuint	h_frag_shader;				/// handle to the fragment (pixel) shader
	GLint	vertex_compiled = 1;		/// status of vertex shader compilation
	GLint	frag_compiled = 1;			/// status of fragment shader compilation
	GLint	prog_linked = 1;			/// status of linking

	// Load the vertex and fragment shaders
	LoadShaderFromFile(vertexfile, &vertex_shader);
	LoadShaderFromFile(fragfile, &frag_shader);

	// compile and link the programs
	if (GLEW_VERSION_2_0) h_vertex_shader = glCreateShader(GL_VERTEX_SHADER); else h_vertex_shader = glCreateShaderObjectARB(GL_VERTEX_SHADER_ARB);
	if (GLEW_VERSION_2_0) h_frag_shader = glCreateShader(GL_FRAGMENT_SHADER); else h_frag_shader = glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);
	if (GLEW_VERSION_2_0) glShaderSource(h_vertex_shader, 1, (const GLchar **) &vertex_shader, NULL); else glShaderSourceARB(h_vertex_shader, 1, (const GLchar **) &vertex_shader, NULL);
	if (GLEW_VERSION_2_0) glShaderSource(h_frag_shader, 1, (const GLchar **) &frag_shader, NULL); else glShaderSourceARB(h_frag_shader, 1, (const GLchar **) &frag_shader, NULL);
	if (GLEW_VERSION_2_0) glCompileShader(h_vertex_shader); else glCompileShaderARB(h_vertex_shader);
	if (GLEW_VERSION_2_0) glGetShaderiv(h_vertex_shader, GL_COMPILE_STATUS, &vertex_compiled); 
	if (GLEW_VERSION_2_0) glCompileShader(h_frag_shader); else glCompileShaderARB(h_frag_shader);
	if (GLEW_VERSION_2_0) glGetShaderiv(h_frag_shader, GL_COMPILE_STATUS, &frag_compiled);
	printInfoLog(h_vertex_shader, vertexfile);
	printInfoLog(h_frag_shader, fragfile);
	if (!vertex_compiled)
	{
		printf("Error: could not compile vertex shader program %s\n", vertexfile);
		return false;
	}
	if (!frag_compiled)
	{
		printf("Error: could not compile fragment shader program %s\n", fragfile);
		return false;
	}
	if (GLEW_VERSION_2_0) *h_program = glCreateProgram(); else *h_program = glCreateProgramObjectARB();
	if (GLEW_VERSION_2_0) glAttachShader(*h_program, h_vertex_shader); else glAttachObjectARB(*h_program, h_vertex_shader);
	if (GLEW_VERSION_2_0) glAttachShader(*h_program, h_frag_shader); else glAttachObjectARB(*h_program, h_frag_shader);
	if (GLEW_VERSION_2_0) glLinkProgram(*h_program); else glLinkProgramARB(*h_program);
	if (GLEW_VERSION_2_0) glGetProgramiv(*h_program, GL_LINK_STATUS, &prog_linked);
	printInfoLog(*h_program, "");
	if (!prog_linked)
	{
		printf("Error: could not link shader programs\n");
		return false;
	}

	return true;
}


#endif

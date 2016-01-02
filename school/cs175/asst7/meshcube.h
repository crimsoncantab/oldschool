#pragma once
#include "shape.h"
#include "mesh.h"
#include "vec.h"
#include <cmath>

class MeshCube :
	public Shape
{
private:
	Mesh mesh_;
	Mesh refMesh_;
	int numSubdiv_;
	bool smooth_;
public:
	MeshCube(string name, Rbt frame, Vector4 color, Mesh& mesh) : Shape(name,frame,color), mesh_(mesh), refMesh_(mesh), smooth_(false), numSubdiv_(0){setNormals(mesh_);}
	MeshCube(Rbt frame, Vector4 color, Mesh& mesh) : Shape("a mesh cube", frame,color), mesh_(mesh), refMesh_(mesh), smooth_(false), numSubdiv_(0){setNormals(mesh_);}
	void increaseSubdiv() {
		if(numSubdiv_ < 5) numSubdiv_++;
	}
	void decreaseSubdiv() {
		if(numSubdiv_ > 0) numSubdiv_--;
	}
	void scalify(int t) {
		mesh_ = Mesh(refMesh_);
		int numVertices = mesh_.getNumVertices();
		for (int i = 0; i < numVertices; i++) {
			Mesh::Vertex & v = mesh_.getVertex(i);
			double scale = 1.;
			if (i % 2 == 0) {
				scale = std::sin((double)t * CS175_PI / 180. * (i % 5)) + 1.2;
			} else {
				scale = std::sin((double)t * CS175_PI / 180. * (i % 4)) + 1.2;
			}
			v.setPosition(v.getPosition() * scale);
		}
		subdivide();
	}
	void changeShading() {
		smooth_ = !smooth_;
		setNormals(mesh_);
	}
protected:
	void subdivide() {
		for (int i =0; i < numSubdiv_; i++) {
			subdivFaces(mesh_);
			subdivEdges(mesh_);
			subdivVertices(mesh_);
			mesh_.subdivide();
		}
		setNormals(mesh_);
	}
	void subdivEdges(Mesh& mesh){
		int numEdges = mesh.getNumEdges();
		for (int i = 0; i < numEdges; i++) {
			Mesh::Edge edge = mesh.getEdge(i);
			Vector3 sum = edge.getVertex(0).getPosition() + edge.getVertex(1).getPosition()
				+ mesh.getNewFaceVertex(edge.getFace(0)) + mesh.getNewFaceVertex(edge.getFace(1));
			Vector3 newVertex = sum/4;
			
			mesh.setNewEdgeVertex(edge,newVertex);
		}
	}
	void subdivVertices(Mesh& mesh){
		int numVertices = mesh.getNumVertices();
		for (int i = 0; i < numVertices; i++) {
			Mesh::Vertex vertex = mesh.getVertex(i);
			Mesh::VertexIterator it = vertex.getIterator(), it0(it);
			Vector3 vertexSum(0);
			Vector3 faceSum(0);
			int valence = 0;
			do {
				faceSum += mesh.getNewFaceVertex(it.getFace());
				vertexSum += it.getVertex().getPosition();
				valence++;
				++it;
			}while (it != it0);
			Vector3 newVertex = vertex.getPosition()*((valence - 2)/(double)valence);
			int valenceSq = valence * valence;
			newVertex += (vertexSum / valenceSq);
			newVertex += (faceSum / valenceSq);
			mesh.setNewVertexVertex(vertex, newVertex);
		}
	}
	void subdivFaces(Mesh& mesh) {
		int numFaces = mesh.getNumFaces();
		for (int i = 0; i < numFaces; i++) {
			Mesh::Face face = mesh.getFace(i);
			Vector3 sum(0);
			int valence = 0, numVertices = face.getNumVertices();
			for (int j = 0; j < numVertices; j++) {
				sum += face.getVertex(j).getPosition();
				valence++;
			}
			Vector3 newVertex = sum/valence;
			mesh.setNewFaceVertex(face,newVertex);
		}
	}
	static void setNormals(Mesh& mesh) {
		int numVertices = mesh.getNumVertices();
		for (int i = 0; i < numVertices; i++) {
			setNormal(mesh.getVertex(i));
		}

	}
	static void setNormal(Mesh::Vertex& vertex) {
		Mesh::VertexIterator it = vertex.getIterator(), it0(it);
		Vector3 sum(0);
		int valence = 0;
		do {
			sum += it.getFace().getNormal();
			valence++;
			++it;
		}while (it != it0);
		vertex.setNormal((sum/valence).normalize());

	}
	virtual void draw(ShaderState &glAccess) {
		safe_glVertexAttrib4f(glAccess.h_vColor_, color_[0], color_[1], color_[2], color_[3]);					// set color
		glBegin(GL_TRIANGLES);
		int numFaces = mesh_.getNumFaces();
		for (int i = 0; i < numFaces; i++) {
			drawFace(mesh_.getFace(i), smooth_);
		}
		glEnd();
	}
	static void drawFace(Mesh::Face face, bool smooth) {
		Mesh::Vertex v1 = face.getVertex(0);
		Mesh::Vertex v2 = face.getVertex(1);
		Mesh::Vertex v3 = face.getVertex(2);
		Mesh::Vertex v4 = face.getVertex(3);
		if (!smooth) {
			Vector3 newNormal = Vector3::cross(v1.getPosition() - v2.getPosition(), v1.getPosition()-v3.getPosition());
			sendNormal(newNormal.normalize());
		}
		if (smooth) sendNormal(v1.getNormal());
		sendVertex(v1.getPosition());
		if (smooth) sendNormal(v2.getNormal());
		sendVertex(v2.getPosition());
		if (smooth) sendNormal(v3.getNormal());
		sendVertex(v3.getPosition());
		
		if (smooth) sendNormal(v1.getNormal());
		sendVertex(v1.getPosition());
		if (smooth) sendNormal(v3.getNormal());
		sendVertex(v3.getPosition());
		if (smooth) sendNormal(v4.getNormal());
		sendVertex(v4.getPosition());
	}
	static void sendNormal(Vector3 normal) {
		glNormal3f(normal[0],normal[1],normal[2]);
	}
	static void sendVertex(Vector3 vertex) {
		glVertex3f(vertex[0],vertex[1],vertex[2]);
	}

}; 
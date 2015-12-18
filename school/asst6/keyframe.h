#pragma once

#include <cstdio>
#include <cstdlib>
#include <iostream>

class KeyFrame
{
protected:
	vector <Object3D*> objects_;													// all objects
public:

	KeyFrame(vector <Object3D*> objects) : objects_(objects){}
	KeyFrame(const KeyFrame & k) {
		for (size_t i = 0; i < k.objects_.size(); ++i)
		{
			objects_.push_back(k.objects_[i]->clone());
		}
	}
	vector <Object3D*> getObjects() {
		return objects_;
	}
	void addObject(Object3D* o) {objects_.push_back(o);}
	virtual void copyFrom(KeyFrame& f) {
		objects_.clear();
		for (size_t i = 0; i < f.objects_.size(); ++i)
		{
			objects_.push_back(f.objects_[i]->clone());
		}
	}
	virtual void lerpFrames(KeyFrame & f1, KeyFrame & f2, double t) {
		objects_.clear();
		for (size_t i = 0; i < f1.objects_.size(); ++i)
		{
			objects_.push_back(f1.objects_[i]->lerp(f2.objects_[i], t));
		}
	}
	virtual void bezerpFrames(KeyFrame & f1, KeyFrame & f2, KeyFrame & prev, KeyFrame & post, double t) {
		objects_.clear();
		for (size_t i = 0; i < f1.objects_.size(); ++i)
		{
			objects_.push_back(f1.objects_[i]->bezerp(f2.objects_[i], prev.objects_[i], post.objects_[i], t));
		}
	}
	void save(ostream * out) {
		for(vector<Object3D*>::iterator it = objects_.begin(); it != objects_.end(); it++) {
			(*it)->save(out);
		}
	}

	void load(istream * in) {
		for(vector<Object3D*>::iterator it = objects_.begin(); it != objects_.end(); it++) {
			(*it)->load(in);
		}
	}
	static void saveKeyFrames(list<KeyFrame*> frames) {
		ofstream * saveFile = new ofstream("out.sav");
		int numFrames = frames.size();
		*saveFile << numFrames << "\n";
		for(list<KeyFrame*>::iterator it = frames.begin(); it != frames.end(); it++) {
			(*it)->save(saveFile);
		}
		saveFile->close();
	}

	static list<KeyFrame*> loadKeyFrames(KeyFrame * templateFrame) {
		ifstream * read = new ifstream("out.sav");
		int numFrames = 0;
		*read>> numFrames;
		list<KeyFrame*> frames;
		for (int i = 0; i < numFrames; i++) {
			KeyFrame * temp = new KeyFrame(*templateFrame);
			temp->load(read);
			frames.push_back(temp);
		}
		return frames;
	}
};

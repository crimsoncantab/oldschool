# -*- coding: utf-8 -*-
### required - do no delete
def user(): return dict(form=auth())

def download(): return response.download(request,db)

@auth.requires(auth.user_id and auth.user.email=='mcginnis.loren@gmail.com')
def data(): return dict(form=crud())

def index():
    return dict()
    
def bib():
    return dict()
    
def intro():
    essays = db().select(db.intro.essay)
    return dict(essays = essays)

def error():
    return dict()

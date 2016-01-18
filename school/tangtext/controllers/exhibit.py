import logging
# coding: utf8
# try something like
def index():
    #for now just redirect
    redirect(URL(plain))
    #return dict(message=db.objects[1])

def plain():
    obj = None
    if len(request.args):
        obj = db.objects[int(request.args[0])]
    if not obj:
        if session.last_obj_id:
            redir_id = session.last_obj_id
            session.last_obj_id = None #to avoid redir loops
        else:
            #get default object
            try:
                redir_id = db(db.objects.isdefault == True).select(db.objects.id)[0].id
            except:
                logging.warn('No default object found')
                redirect(URL('default','index'))
        redirect(URL(args=redir_id))
    else:
        links = db(db.olinks.linker == obj.id).select(db.olinks.linkee)
        session.last_obj_id = obj.id
        return dict(obj = obj, links=links)
        
def catalog():
    objects = db().select(db.objects.id, db.objects.title)
    return dict(objects = objects)

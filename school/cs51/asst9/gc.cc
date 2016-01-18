#include "interp.h"
#include "value.h"
#include "eval.h"

lambda *lambda::freelist = NULL;
symbol *symbol::freelist = NULL;
number *number::freelist = NULL;
pair *pair::freelist = NULL;
env *env::freelist = NULL;
managed *managed::heap = NULL;
vector<managed*> managed::preserves;

const int GC_LIMIT = 20;
static int gc_count = 0;

void preserve(managed *v)
{
    preserves.push_back(v);
}

// Pop the top n elements from the preserves stack. 
void release_n(int n)
{
    preserves.resize(preserves.size() - n);
}

// Find unmarked objects and free them 
void managed::gc_sweep()
{
    managed **pv = &managed::heap;
    managed *next;
    int cnt = 0;

    while (*pv != NULL) {
        if (!(*pv)->marked) {
            next = (*pv)->next;
            (*pv)->freeValue();
            *pv = next;
            cnt++;
        }
        else {
            pv = &((*pv)->next);
        }
    }
}

// Clear all old marks 
void managed::gc_unmark_all()
{
    managed *v = managed::heap;

    while (v != NULL) {
        v->marked = false;
        v = v->next;
    }
}

void gc_collect()
{
    managed::gc_unmark_all();

    scheme.mark();

    managed::gc_sweep();
}

void gc_maybe()
{
    if (gc_count > GC_LIMIT) {
        gc_collect();
        gc_count = 0;
    }
    else {
        gc_count++;
    }
}

void interp::mark()
{
    // mark everything in the global environment 
    globals.mark(); 

    // mark all symbols 
    symtab_t::iterator i = symtab.begin();
    while (i != symtab.end()) {
        i->second->mark();
        i++;
    }

    // mark all special forms 
    map<symbol *, syntax_t>::iterator j = syntax.begin();
    while (j != syntax.end()) {
        j->first->mark();
        j++;
    }

    for (size_t k = 0; k < preserves.size(); k++)
        preserves[k]->mark();
}

void env::mark()
{
    if (marked)
        return;
    marked = true;

    // mark this envrionment and all its parents 
    for (env *cur = this; cur != NULL; cur = cur->_next) {
        envmap::iterator i = cur->_map.begin();

        /* mark all symbols in this envrionment and the values they
           point to */
        while (i != cur->_map.end()) {
            ((symbol*)i->first)->mark();
            i->second->mark();
            i++;
        }
    }
}

void lambda::mark()
{
    if (marked)
        return;
    marked = true;

    _locals->mark();
    ((value*)_formals)->mark();
    for(unsigned i=0; i < _body.size(); i++)
        _body[i]->mark();
}

void pair::mark()
{
    if (marked)
        return;
    marked = true;

    car->mark();
    cdr->mark();
}

/*
 * Overload the scheme types' operator new so that they call our alloc
 * function. Note that all these are subclasses of managed, so that is
 * why it knows what alloc function to call.
 */
void *pair::operator new(unsigned int size)
{
    return (void*)alloc<pair>(size); 
}

void *symbol::operator new(unsigned int size)
{
    return (void*)alloc<symbol>(size);
}

void *number::operator new(unsigned int size)
{
    return (void*)alloc<number>(size);
}

void *lambda::operator new(unsigned int size)
{
    return (void*)alloc<lambda>(size);
}

void *env::operator new(unsigned int size)
{
    return (void*)alloc<env>(size);
}

/* freeValue() does not destroy the object; it makes it so that it can
 * be reused. Therefore, we need to clear the environment map (you
 * have one of these too, right?) */
void env::freeValue()
{
    _map.clear();
    dealloc<env>(this);
}

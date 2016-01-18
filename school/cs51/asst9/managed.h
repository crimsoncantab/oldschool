#ifndef MANAGED_H
#define MANAGED_H

class managed {
 public:
    static void gc_unmark_all();
    static void gc_sweep();

    /* Prepare an instance to be placed on the free list so that it
     * can be reused. See and understand env::freeValue() for an
     * example. */
    virtual void freeValue() = 0;

    /* Mark a class and let it mark the data that it references. */
    virtual void mark() = 0;

protected:
    template<class C>
    static C *alloc(unsigned int size)
    {
        C *val;

        if (C::freelist != NULL) {
            val = C::freelist;
            C::freelist = (C*)C::freelist->next;
        }
        else {
            val = (C*)malloc(size);
        }
        val->next = heap;
        heap = (managed*)val;
        return val;
    }

    template<class C>
    static void dealloc(C *val)
    {
        val->next = C::freelist;
        C::freelist = val;
    }

    bool marked;
    managed *next;

    static managed *heap;
    /* a stack of objects that need to be preserved. */
    static vector<managed*> preserves;
};

/* GC interface */

/* gc_maybe():
   allow garbage collection to possibly happen
   it is up to the GC subsystem whether collection will actually occur */
void gc_maybe();

/* preserve():
   protect an object not reachable from the rootlist from collection */
void preserve(managed *v);

/* release():
   release n preserved objects so they can be collected */
void release_n(int n);

#endif

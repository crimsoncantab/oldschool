# $Id: Makefile.in,v 12.106 2007/07/05 18:35:02 bostic Exp $

.SUFFIXES: .lo

TARGET=asst4


##################################################
# Installation directories and permissions.
##################################################
prefix=/home/c/s/cs165/
exec_prefix=$(prefix)/bin

srcdir=	.
builddir=$(prefix)/db-4.8.26/build_unix
bindir=	$(exec_prefix)
libdir=	$(prefix)/lib
includedir=	$(prefix)/include

dmode=	755
emode=	555
fmode=	444

transform=s,x,x,

##################################################
# Paths for standard user-level commands.
##################################################
AR=	ar
CHMOD=	chmod
CP=	cp
LN=	ln
MKDIR=	mkdir
RANLIB=	ranlib
RM=	rm
SHELL=	/bin/sh
STRIP=	:

##################################################
# General library information.
##################################################
DEF_LIB=	$(libso_target)
DEF_LIB_CXX=	$(libxso_target)
INSTALLER=	$(LIBTOOL) --mode=install cp -p
LIBTOOL=	$(SHELL) $(prefix)/db-4.8.26/build_unix/libtool

POSTLINK=	$(LIBTOOL) --mode=execute true
SOLINK=		$(LIBTOOL) --mode=link cc -avoid-version
SOFLAGS=	-rpath $(libdir)
LIBMAJOR=	4
LIBVERSION=	4.8

CPPFLAGS=	-I$(builddir) -I$(srcdir) -I$(includedir) -I$(prefix)/db-4.8.26 -I$(prefix)/db-4.8.26/dbinc

##################################################
# C API.
##################################################
CFLAGS=		-c $(CPPFLAGS) -g 
CC=		$(LIBTOOL) --mode=compile cc
CCLINK=		$(LIBTOOL) --mode=link cc -g 

LDFLAGS=	
LIBS=		
TEST_LIBS=	 -lm -lpthread
LIBCSO_LIBS=	 

libdb_base=	libdb
libdb=		$(libdir)/$(libdb_base).a
libdb_version=	$(libdir)/$(libdb_base)-$(LIBVERSION).a
libso=		$(libdir)/$(libdb_base)-$(LIBVERSION).dylib
libso_target=	$(libdir)/$(libdb_base)-$(LIBVERSION).la
libso_default=	$(libdir)/$(libdb_base).dylib
libso_major=	$(libdir)/$(libdb_base)-$(LIBMAJOR).dylib

##################################################
# List of files installed into the library directory.
##################################################
LIB_INSTALL_FILE_LIST=\
	$(libdb) \
	$(libso) \
	$(libso_default) \
	$(libso_major) \
	$(libdb_version) \
	$(libso_target) \
	$(libcxx) \
	$(libxso) \
	$(libxso_default) \
	$(libxso_major) \
	$(libcxx_version) \
	$(libxso_target) \
	$(libtso) \
	$(libtso_default) \
	$(libtso_major) \
	$(libtso_static) \
	$(libtso_target) \
	$(libjso) \
	$(libjso_default) \
	$(libjso_g) \
	$(libjso_major) \
	$(libjso_static) \
	$(libjso_target) \
	$(libj_exjarfile) \
	$(libj_jarfile)

##################################################
# Example programs for C.
##################################################
asst4.lo: $(srcdir)/asst4.c
	$(CC) $(CFLAGS) $?

asst4_auto.lo: $(srcdir)/asst4_auto.c
	$(CC) $(CFLAGS) $?

asst4_autop.lo: $(srcdir)/asst4_autop.c
	$(CC) $(CFLAGS) $?

asst4_rec.lo: $(srcdir)/asst4_rec.c
	$(CC) $(CFLAGS) $?
 
ASST4_OBJS=\
	asst4.lo asst4_auto.lo asst4_autop.lo asst4_rec.lo

asst4: $(ASST4_OBJS) $(DEF_LIB)
	$(CCLINK) -o asst4 \
	    $(LDFLAGS) $(ASST4_OBJS) $(DEF_LIB) $(TEST_LIBS) $(LIBS)

clean:
	$(RM) -rf $(ASST4_OBJS) $(TARGET) *.o
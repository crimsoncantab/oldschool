# Makefile for CS 264 Homework 3

CCFILES = main.cpp bmploader.cpp
CUFILES = histeq_gpu.cu histogram256.cu

CC_DEPS  := $(wildcard *.hpp)
CU_DEPS  := $(wildcard *.hpp) Makefile


EXECUTABLE = mpi_histeq


SRCDIR = ./
BINDIR = ./bin/
OBJDIR = ./obj/

CUDA_INSTALL_PATH ?= /opt/cuda-3.2
CUDA_SDK_INSTALL_PATH ?= /opt/cudasdk-3.2/C

# Compilers
NVCC       := $(CUDA_INSTALL_PATH)/bin/nvcc 
CXX        := mpic++
LINK       := mpic++ -fPIC

# Includes
INCLUDES  += -I. -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SDK_INSTALL_PATH)/common/inc

# Warning flags
CXXWARN_FLAGS := \
        -W -Wall -Wextra -Weffc++ \
        -Wimplicit \
        -Wswitch \
        -Wformat \
        -Wchar-subscripts \
        -Wparentheses \
        -Wmultichar \
        -Wtrigraphs \
        -Wpointer-arith \
        -Wcast-align \
        -Wreturn-type \
        -Wno-unused-function \
        $(SPACE)



# Compiler-specific flags
NVCCFLAGS := --compiler-options -W,-Wall,-Wextra
CXXFLAGS  := $(CXXWARN_FLAGS)


# Common flags
COMMONFLAGS += $(INCLUDES) -DUNIX

ifeq ($(dbg),1)
   COMMONFLAGS += -g
   NVCCFLAGS   += -D_DEBUG
else
   COMMONFLAGS += -O2
endif

# Libs
LIB := -L$(CUDA_INSTALL_PATH)/lib64 -L$(CUDA_SDK_INSTALL_PATH)/lib -lcudart -lcutil_x86_64



TARGET    := $(BINDIR)/$(EXECUTABLE)


# Add common flags
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS  += $(COMMONFLAGS)


#####################
# Set up object files

OBJS += $(patsubst %.cpp,$(OBJDIR)/%.cpp.o,$(notdir $(CCFILES)))
OBJS += $(patsubst %.cu,$(OBJDIR)/%.cu.o,$(notdir $(CUFILES)))


#####################
# Rules

$(OBJDIR)/%.cpp.o : $(SRCDIR)%.cpp $(CC_DEPS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(OBJDIR)/%.cu.o : $(SRCDIR)%.cu $(CU_DEPS)
	$(NVCC) $(NVCCFLAGS) $(SMVERSIONFLAGS) -o $@ -c $<


$(TARGET) : $(OBJDIR)/.exist $(BINDIR)/.exist $(OBJS)
	$(LINK) -o $(TARGET) $(OBJS) $(LIB)


$(OBJDIR)/.exist:
	mkdir -p $(OBJDIR)
	touch $(OBJDIR)/.exist

$(BINDIR)/.exist:
	mkdir -p $(BINDIR)
	touch $(BINDIR)/.exist

tidy:
	find . | egrep "#" | xargs rm -f
	find . | egrep "\~" | xargs rm -f

clean: tidy
	-rm -f $(OBJS)
	-rm -f $(TARGET)

clobber: clean
	-rm -rf $(OBJDIR)
	-rm -rf $(BINDIR)

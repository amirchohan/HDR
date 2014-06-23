EXE=hdr
SRCDIR=../src
OBJDIR=obj

SUFFIXES += .d

JPGINC = -I/usr/include/
JPGDIR = /usr/include/x86_64-linux-gnu/
JPGLIBd = -L$(JPGDIR) -ljpeg

INCS = $(JPGINC)
LIBS = $(JPGLIBd)


CXX      = g++
CXXFLAGS = -I$(SRCDIR) -O2 -fopenmp -DCL_USE_DEPRECATED_OPENCL_1_1_APIS
LDFLAGS  = -lOpenCL -lSDL2_image -lGL
MODULES  = Filter HistEq ReinhardGlobal ReinhardLocal GradDom
OBJECTS  = $(MODULES:%=$(OBJDIR)/%.o)
SOURCES  = $(MODULES:%=$(SRCDIR)/%.cpp)
DEPFILES = $(MODULES:%=$(OBJDIR)/%.d)

all: prebuild $(OBJDIR) $(EXE)


$(EXE): $(OBJECTS) hdr.cpp
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) $(INCS) $(LIBS) -o $@

prebuild:
	$(MAKE) -C ../src/opencl -f $(shell pwd)/Makefile prebuild_opencl

prebuild_opencl:
	./stringify_kernels.sh

$(OBJDIR)/%.d: $(SRCDIR)/%.cpp $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCS) $(LIBS) -MM -MT $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$<) $< -MF $@ 2>/dev/null

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCS) $(LIBS) -o $@ -c $<

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) $(EXE) ../src/opencl/*.h

.PHONY: clean

ifeq (0, $(words $(findstring $(MAKECMDGOALS), clean opencl halide)))
-include $(DEPFILES)
endif

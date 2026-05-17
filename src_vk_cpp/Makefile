CXX   := g++
GLSLC := glslc

CXXFLAGS := -std=c++20 -O2 -Wall -Wextra
CXXFLAGS += $(shell pkg-config --cflags glfw3 vulkan)
LDFLAGS  := $(shell pkg-config --libs glfw3 vulkan) -lvk-bootstrap

SRCS   := main.cpp vk_context.cpp swapchain.cpp render_pass.cpp pipeline.cpp
OBJS   := $(SRCS:.cpp=.o)
TARGET := raytracer_vk

# Shaders: GLSL -> SPIR-V -> C header (embedded into the binary, no runtime file loading).
VERT_SPV := triangle_vert.spv
FRAG_SPV := triangle_frag.spv
VERT_HDR := triangle_vert.h
FRAG_HDR := triangle_frag.h

.PHONY: all shaders clean install

all: shaders $(TARGET)

shaders: $(VERT_HDR) $(FRAG_HDR)

$(VERT_SPV): shaders/triangle.vert
	$(GLSLC) --target-env=vulkan1.2 -o $@ $<

$(FRAG_SPV): shaders/triangle.frag
	$(GLSLC) --target-env=vulkan1.2 -o $@ $<

$(VERT_HDR): $(VERT_SPV)
	xxd -i $< > $@

$(FRAG_HDR): $(FRAG_SPV)
	xxd -i $< > $@

# pipeline.cpp includes the generated headers; rebuild it if shaders change.
pipeline.o: pipeline.cpp $(VERT_HDR) $(FRAG_HDR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

install:
	install -Dm755 $(TARGET) $(DESTDIR)/bin/$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET) $(VERT_SPV) $(FRAG_SPV) $(VERT_HDR) $(FRAG_HDR)

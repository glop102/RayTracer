CXX   := g++

CXXFLAGS := -std=c++20 -O2 -Wall -Wextra
CXXFLAGS += $(shell pkg-config --cflags glfw3 vulkan)
LDFLAGS  := $(shell pkg-config --libs glfw3 vulkan shaderc) -lvk-bootstrap

SRCS   := main.cpp vk_context.cpp swapchain.cpp render_pass.cpp pipeline.cpp frame_sync.cpp
OBJS   := $(SRCS:.cpp=.o)
TARGET := raytracer_vk

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

install:
	install -Dm755 $(TARGET) $(DESTDIR)/bin/$(TARGET)
	install -Dm644 shaders/triangle.vert $(DESTDIR)/share/raytracer_vk/shaders/triangle.vert
	install -Dm644 shaders/triangle.frag $(DESTDIR)/share/raytracer_vk/shaders/triangle.frag

clean:
	rm -f $(OBJS) $(TARGET)

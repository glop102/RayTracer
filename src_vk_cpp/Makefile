CXX   := g++

CXXFLAGS := -std=c++20 -O2 -Wall -Wextra
CXXFLAGS += $(shell pkg-config --cflags glfw3 vulkan glm)
LDFLAGS  := $(shell pkg-config --libs glfw3 vulkan shaderc) -lvk-bootstrap -lOpenImageDenoise

SRCS   := main.cpp vk_context.cpp swapchain.cpp frame_sync.cpp mesh.cpp camera.cpp \
          accel.cpp gpu_buffer.cpp gpu_texture.cpp scene_data.cpp gltf_scene.cpp \
          rt_output.cpp rt_pipeline.cpp pipeline.cpp render_pass.cpp denoiser.cpp denoiser_oidn.cpp \
          denoiser_svgf.cpp
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
	install -Dm644 shaders/triangle.vert      $(DESTDIR)/share/raytracer_vk/shaders/triangle.vert
	install -Dm644 shaders/triangle.frag      $(DESTDIR)/share/raytracer_vk/shaders/triangle.frag
	install -Dm644 shaders/raygen.rgen        $(DESTDIR)/share/raytracer_vk/shaders/raygen.rgen
	install -Dm644 shaders/miss.rmiss         $(DESTDIR)/share/raytracer_vk/shaders/miss.rmiss
	install -Dm644 shaders/shadow_miss.rmiss  $(DESTDIR)/share/raytracer_vk/shaders/shadow_miss.rmiss
	install -Dm644 shaders/closest_hit.rchit         $(DESTDIR)/share/raytracer_vk/shaders/closest_hit.rchit
	install -Dm644 shaders/glass_closest_hit.rchit   $(DESTDIR)/share/raytracer_vk/shaders/glass_closest_hit.rchit
	install -Dm644 shaders/glass_any_hit.rahit        $(DESTDIR)/share/raytracer_vk/shaders/glass_any_hit.rahit
	install -Dm644 shaders/hit_common.glsl            $(DESTDIR)/share/raytracer_vk/shaders/hit_common.glsl
	install -Dm644 shaders/svgf_atrous.comp          $(DESTDIR)/share/raytracer_vk/shaders/svgf_atrous.comp

clean:
	rm -f $(OBJS) $(TARGET)

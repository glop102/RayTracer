CXX      := g++
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra
LDFLAGS  :=

# glfw3 and vulkan provide pkg-config files; vk-bootstrap does not.
# VMA and GLM are header-only — no link step needed.
# In a nix build, NIX_CFLAGS_COMPILE and NIX_LDFLAGS already carry the paths
# from buildInputs, so the pkg-config calls are supplementary rather than load-bearing.
CXXFLAGS += $(shell pkg-config --cflags glfw3 vulkan)
LDFLAGS  += $(shell pkg-config --libs glfw3 vulkan) -lvk-bootstrap

SRCS   := main.cpp vk_context.cpp swapchain.cpp render_pass.cpp
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

clean:
	rm -f $(OBJS) $(TARGET)

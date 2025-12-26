OBJ_DIR = build
EXTRA_CXXOPTS = -std=c++20 -O3 -freciprocal-math -fno-rounding-math
# -msse -msse2 -msse3 -mavx -mavx2
LIBS = -lpng
CXXFLAGS := ${CXXFLAGS} ${EXTRA_CXXOPTS}

SRCS = $(shell find -name '*.cpp')
OBJS = $(patsubst %.cpp,${OBJ_DIR}/%.o,$(SRCS))

all: raytrace

raytrace: ${OBJS}
	$(CXX) $(OBJS) $(LIBS) -o $@ ${CXXFLAGS} -flto

${OBJ_DIR}/%.o: %.cpp | ${OBJ_DIR}
	$(CXX) -c $< -o $@ ${CXXFLAGS}

${OBJ_DIR}:
	mkdir ${OBJ_DIR}

.PHONY : clean
clean:
	rm -rf ${OBJ_DIR}

.PHONY: video videoslow
video:
	ffmpeg -y -r 30 -i video/%d.png camera_sweep_partial.webm
	ffmpeg -y -stream_loop 4 -i camera_sweep_partial.webm -c:v copy camera_sweep.webm
	rm camera_sweep_partial.webm
videoslow:
	ffmpeg -y -r 10 -i video/%d.png camera_sweep_partial.webm
	ffmpeg -y -stream_loop 4 -i camera_sweep_partial.webm -c:v copy camera_sweep.webm
	rm camera_sweep_partial.webm

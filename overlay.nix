final: prev: {
  raytracer = final.stdenv.mkDerivation {
    pname = "raytracer";
    version = "1.0";

    src = ./src_cpp;

    buildInputs = [final.libpng];
    makeFlags = [
      "DESTDIR=$(out)"
    ];

    meta = {
      description = "Does some raytracing with a custom concept implementation.";
    };
  };

  raytracer_vk = final.stdenv.mkDerivation {
    pname = "raytracer_vk";
    version = "0.1";

    src = ./src_vk_cpp;

    nativeBuildInputs = with final; [
      pkg-config
    ];

    buildInputs = with final; [
      vulkan-headers
      vulkan-loader
      vulkan-validation-layers
      vk-bootstrap
      vulkan-memory-allocator
      shaderc
      glfw
      glm
      tinygltf
    ];

    makeFlags = [
      "DESTDIR=$(out)"
    ];

    meta = {
      description = "Vulkan rework of the ray tracer — milestone 3: bunny mesh with depth buffer.";
    };
  };
}

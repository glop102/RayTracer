final: prev: {
  openimagedenoise = prev.openimagedenoise.overrideAttrs (old: {
    postPatch = (old.postPatch or "") + ''
      sed -i '/CMAKE_INSTALL_PREFIX.*hip.preinstall/d' devices/CMakeLists.txt
      sed -i '/^$/{N;/# Due to limitations/{N;N;N;/hip\/preinstall\//{N;N;N;d}}}' devices/CMakeLists.txt
    '';
    nativeBuildInputs = old.nativeBuildInputs ++ [
      final.rocmPackages.llvm.clang
    ];
    buildInputs = old.buildInputs ++ [
      final.rocmPackages.clr
    ];
    cmakeFlags = old.cmakeFlags ++ [
      (final.lib.cmakeBool "OIDN_DEVICE_HIP" true)
      (final.lib.cmakeFeature "ROCM_PATH" "${final.rocmPackages.clr}")
      (final.lib.cmakeFeature "OIDN_DEVICE_HIP_COMPILER" "${final.rocmPackages.llvm.clang}/bin/clang++")
    ];
  });

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
      makeWrapper
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
      openimagedenoise
    ];

    makeFlags = [
      "DESTDIR=$(out)"
    ];

    postInstall = ''
      wrapProgram $out/bin/raytracer_vk \
        --set ROCM_PATH ${final.rocmPackages.clr} \
        --prefix LD_LIBRARY_PATH : ${final.rocmPackages.clr}/lib
    '';

    meta = {
      description = "Vulkan rework of the ray tracer — milestone 3: bunny mesh with depth buffer.";
    };
  };
}

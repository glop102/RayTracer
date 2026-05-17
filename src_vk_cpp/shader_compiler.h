#pragma once
#include <shaderc/shaderc.hpp>
#include <vulkan/vulkan.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __linux__
#include <unistd.h>
#endif

// Locate the shaders/ directory at the Nix install path or ./shaders/ fallback.
inline std::filesystem::path find_shader_dir() {
#ifdef __linux__
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        auto candidate = std::filesystem::path(buf).parent_path()
                         / "../share/raytracer_vk/shaders";
        if (std::filesystem::is_directory(candidate))
            return candidate;
    }
#endif
    if (std::filesystem::is_directory("shaders"))
        return "shaders";
    throw std::runtime_error("Cannot locate shader directory");
}

inline std::string read_file(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open shader: " + path.string());
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

inline std::vector<uint32_t> compile_glsl(const std::string& source,
                                          const std::string& name,
                                          shaderc_shader_kind kind) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions opts;
    opts.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    opts.SetTargetSpirv(shaderc_spirv_version_1_5);
    opts.SetOptimizationLevel(shaderc_optimization_level_performance);

    auto result = compiler.CompileGlslToSpv(source, kind, name.c_str(), opts);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success)
        throw std::runtime_error("Shader compile error in " + name + ":\n" + result.GetErrorMessage());

    return {result.cbegin(), result.cend()};
}

inline VkShaderModule make_module(VkDevice dev, const std::vector<uint32_t>& spv) {
    VkShaderModuleCreateInfo info{};
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = spv.size() * sizeof(uint32_t);
    info.pCode    = spv.data();

    VkShaderModule mod;
    if (vkCreateShaderModule(dev, &info, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("Shader module creation failed");
    return mod;
}

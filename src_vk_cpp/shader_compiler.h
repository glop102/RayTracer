#pragma once
#include <shaderc/shaderc.hpp>
#include <vulkan/vulkan.h>

#include <filesystem>
#include <fstream>
#include <memory>
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

// Resolves #include "file" relative to a given shader directory.
class ShaderIncluder : public shaderc::CompileOptions::IncluderInterface {
    struct Data { std::string name; std::string content; };
    std::filesystem::path dir_;
public:
    explicit ShaderIncluder(std::filesystem::path dir) : dir_(std::move(dir)) {}

    shaderc_include_result* GetInclude(const char* requested, shaderc_include_type,
                                       const char*, size_t) override {
        auto* d = new Data;
        auto path = dir_ / requested;
        d->name = path.string();
        try { d->content = read_file(path); }
        catch (const std::exception& e) {
            d->name    = "";       // empty name signals error to shaderc
            d->content = e.what();
        }
        auto* r = new shaderc_include_result;
        r->source_name        = d->name.c_str();
        r->source_name_length = d->name.size();
        r->content            = d->content.c_str();
        r->content_length     = d->content.size();
        r->user_data          = d;
        return r;
    }

    void ReleaseInclude(shaderc_include_result* r) override {
        delete static_cast<Data*>(r->user_data);
        delete r;
    }
};

// Compile GLSL source to SPIR-V. Pass include_dir to support #include directives.
inline std::vector<uint32_t> compile_glsl(const std::string& source,
                                          const std::string& name,
                                          shaderc_shader_kind kind,
                                          const std::filesystem::path& include_dir = {}) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions opts;
    opts.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    opts.SetTargetSpirv(shaderc_spirv_version_1_5);
    opts.SetOptimizationLevel(shaderc_optimization_level_performance);

    if (!include_dir.empty())
        opts.SetIncluder(std::make_unique<ShaderIncluder>(include_dir));

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

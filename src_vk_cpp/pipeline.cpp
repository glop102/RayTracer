#include "pipeline.h"
#include "vk_context.h"
#include "render_pass.h"

#include <shaderc/shaderc.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef __linux__
#include <unistd.h>
#endif

// Locate the shaders/ directory. Checks the installed Nix layout first
// (<binary>/../share/raytracer_vk/shaders/), then falls back to ./shaders/
// for running directly from the build directory during development.
static std::filesystem::path find_shader_dir() {
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

static std::string read_file(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open shader: " + path.string());
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::vector<uint32_t> compile_glsl(const std::string& source,
                                           const std::string& name,
                                           shaderc_shader_kind kind) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions opts;
    opts.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    opts.SetOptimizationLevel(shaderc_optimization_level_performance);

    auto result = compiler.CompileGlslToSpv(source, kind, name.c_str(), opts);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success)
        throw std::runtime_error("Shader compile error in " + name + ":\n" + result.GetErrorMessage());

    return {result.cbegin(), result.cend()};
}

static VkShaderModule make_module(VkDevice dev, const std::vector<uint32_t>& spv) {
    VkShaderModuleCreateInfo info{};
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = spv.size() * sizeof(uint32_t);
    info.pCode    = spv.data();

    VkShaderModule mod;
    if (vkCreateShaderModule(dev, &info, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("Shader module creation failed");
    return mod;
}

Pipeline::Pipeline(VkContext& ctx, RenderPass& render_pass, VkExtent2D extent) {
    device = ctx.device.device;

    auto shader_dir = find_shader_dir();
    auto vert_spv = compile_glsl(read_file(shader_dir / "triangle.vert"), "triangle.vert", shaderc_glsl_vertex_shader);
    auto frag_spv = compile_glsl(read_file(shader_dir / "triangle.frag"), "triangle.frag", shaderc_glsl_fragment_shader);

    VkShaderModule vert = make_module(device, vert_spv);
    VkShaderModule frag = make_module(device, frag_spv);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName  = "main";

    VkPipelineVertexInputStateCreateInfo vert_input{};
    vert_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{};
    viewport.width    = static_cast<float>(extent.width);
    viewport.height   = static_cast<float>(extent.height);
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.extent = extent;

    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.pViewports    = &viewport;
    viewport_state.scissorCount  = 1;
    viewport_state.pScissors     = &scissor;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode    = VK_CULL_MODE_NONE;
    raster.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments    = &blend_attachment;

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    if (vkCreatePipelineLayout(device, &layout_info, nullptr, &layout) != VK_SUCCESS)
        throw std::runtime_error("Pipeline layout creation failed");

    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount          = 2;
    pipeline_info.pStages             = stages;
    pipeline_info.pVertexInputState   = &vert_input;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState      = &viewport_state;
    pipeline_info.pRasterizationState = &raster;
    pipeline_info.pMultisampleState   = &multisample;
    pipeline_info.pColorBlendState    = &blend;
    pipeline_info.layout              = layout;
    pipeline_info.renderPass          = render_pass.render_pass;
    pipeline_info.subpass             = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error("Graphics pipeline creation failed");

    vkDestroyShaderModule(device, vert, nullptr);
    vkDestroyShaderModule(device, frag, nullptr);
}

Pipeline::~Pipeline() {
    vkDestroyPipeline      (device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, layout,   nullptr);
}

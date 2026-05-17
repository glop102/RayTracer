#include "rt_pipeline.h"
#include "vk_context.h"
#include "camera.h"         // RtCameraPush
#include "shader_compiler.h"

#include <cstring>
#include <stdexcept>
#include <vector>

RtPipeline::RtPipeline(VkContext& ctx, VkDescriptorSetLayout rt_output_layout) {
    device    = ctx.device.device;
    allocator = ctx.allocator;

    // ------------------------------------------------------------------ Shaders
    auto shader_dir = find_shader_dir();

    auto rgen_spv = compile_glsl(read_file(shader_dir / "raygen.rgen"),       "raygen.rgen",    shaderc_raygen_shader);
    auto rmiss_spv= compile_glsl(read_file(shader_dir / "miss.rmiss"),        "miss.rmiss",     shaderc_miss_shader);
    auto rchit_spv= compile_glsl(read_file(shader_dir / "closest_hit.rchit"), "closest_hit.rchit", shaderc_closesthit_shader);

    VkShaderModule rgen_mod  = make_module(device, rgen_spv);
    VkShaderModule rmiss_mod = make_module(device, rmiss_spv);
    VkShaderModule rchit_mod = make_module(device, rchit_spv);

    VkPipelineShaderStageCreateInfo stages[3]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[0].module = rgen_mod;
    stages[0].pName  = "main";

    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[1].module = rmiss_mod;
    stages[1].pName  = "main";

    stages[2].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[2].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[2].module = rchit_mod;
    stages[2].pName  = "main";

    // ------------------------------------------------------------------ Shader groups
    // Group 0: raygen (general group, raygen stage)
    // Group 1: miss   (general group, miss stage)
    // Group 2: hit    (triangles group, closest-hit stage)
    VkRayTracingShaderGroupCreateInfoKHR groups[3]{};
    groups[0].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[0].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader      = 0;  // rgen stage index
    groups[0].closestHitShader   = VK_SHADER_UNUSED_KHR;
    groups[0].anyHitShader       = VK_SHADER_UNUSED_KHR;
    groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;

    groups[1].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[1].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader      = 1;  // miss stage index
    groups[1].closestHitShader   = VK_SHADER_UNUSED_KHR;
    groups[1].anyHitShader       = VK_SHADER_UNUSED_KHR;
    groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;

    groups[2].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[2].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[2].generalShader      = VK_SHADER_UNUSED_KHR;
    groups[2].closestHitShader   = 2;  // chit stage index
    groups[2].anyHitShader       = VK_SHADER_UNUSED_KHR;
    groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

    // ------------------------------------------------------------------ Pipeline layout
    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    push_range.offset     = 0;
    push_range.size       = sizeof(RtCameraPush);

    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount         = 1;
    layout_ci.pSetLayouts            = &rt_output_layout;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges    = &push_range;
    if (vkCreatePipelineLayout(device, &layout_ci, nullptr, &layout) != VK_SUCCESS)
        throw std::runtime_error("RT pipeline layout creation failed");

    // ------------------------------------------------------------------ RT pipeline
    VkRayTracingPipelineCreateInfoKHR pipe_ci{};
    pipe_ci.sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    pipe_ci.stageCount                   = 3;
    pipe_ci.pStages                      = stages;
    pipe_ci.groupCount                   = 3;
    pipe_ci.pGroups                      = groups;
    pipe_ci.maxPipelineRayRecursionDepth = 1;  // no recursive bounces for the milestone
    pipe_ci.layout                       = layout;

    if (ctx.pfn_vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                               1, &pipe_ci, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error("RT pipeline creation failed");

    vkDestroyShaderModule(device, rgen_mod,  nullptr);
    vkDestroyShaderModule(device, rmiss_mod, nullptr);
    vkDestroyShaderModule(device, rchit_mod, nullptr);

    // ------------------------------------------------------------------ Shader binding table
    uint32_t handle_size  = ctx.rt_pipeline_props.shaderGroupHandleSize;
    uint32_t handle_align = ctx.rt_pipeline_props.shaderGroupHandleAlignment;
    uint32_t base_align   = ctx.rt_pipeline_props.shaderGroupBaseAlignment;

    auto align_up = [](uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); };
    uint32_t handle_stride = align_up(handle_size, handle_align);

    // Three regions: raygen (group 0), miss (group 1), hit (group 2).
    uint32_t raygen_offset = 0;
    uint32_t miss_offset   = align_up(handle_stride, base_align);
    uint32_t hit_offset    = align_up(miss_offset + handle_stride, base_align);
    uint32_t sbt_size      = hit_offset + handle_stride;

    // Fetch all three handles from the driver.
    std::vector<uint8_t> handles(3 * handle_size);
    if (ctx.pfn_vkGetRayTracingShaderGroupHandlesKHR(device, pipeline,
                                                     0, 3, handles.size(), handles.data()) != VK_SUCCESS)
        throw std::runtime_error("vkGetRayTracingShaderGroupHandlesKHR failed");

    // Host-visible SBT buffer (handles are written by the CPU).
    VkBufferCreateInfo sbt_ci{};
    sbt_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    sbt_ci.size  = sbt_size;
    sbt_ci.usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VmaAllocationCreateInfo sbt_ai{};
    sbt_ai.usage = VMA_MEMORY_USAGE_AUTO;
    sbt_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                   VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo sbt_alloc_info{};
    if (vmaCreateBuffer(ctx.allocator, &sbt_ci, &sbt_ai,
                        &sbt_buf, &sbt_alloc, &sbt_alloc_info) != VK_SUCCESS)
        throw std::runtime_error("SBT buffer creation failed");

    auto* data = static_cast<uint8_t*>(sbt_alloc_info.pMappedData);
    std::memcpy(data + raygen_offset, handles.data() + 0 * handle_size, handle_size);
    std::memcpy(data + miss_offset,   handles.data() + 1 * handle_size, handle_size);
    std::memcpy(data + hit_offset,    handles.data() + 2 * handle_size, handle_size);
    vmaFlushAllocation(ctx.allocator, sbt_alloc, 0, VK_WHOLE_SIZE);

    VkBufferDeviceAddressInfo addr_info{};
    addr_info.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addr_info.buffer = sbt_buf;
    VkDeviceAddress sbt_addr = vkGetBufferDeviceAddress(device, &addr_info);

    raygen_region.deviceAddress = sbt_addr + raygen_offset;
    raygen_region.stride        = handle_stride;
    raygen_region.size          = handle_stride;

    miss_region.deviceAddress = sbt_addr + miss_offset;
    miss_region.stride        = handle_stride;
    miss_region.size          = handle_stride;

    hit_region.deviceAddress = sbt_addr + hit_offset;
    hit_region.stride        = handle_stride;
    hit_region.size          = handle_stride;
    // callable_region stays zero (no callable shaders)
}

RtPipeline::~RtPipeline() {
    vmaDestroyBuffer(allocator, sbt_buf, sbt_alloc);
    vkDestroyPipeline      (device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, layout,   nullptr);
}

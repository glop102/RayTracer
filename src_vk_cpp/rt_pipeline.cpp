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

    auto rgen_spv        = compile_glsl(read_file(shader_dir / "raygen.rgen"),            "raygen.rgen",            shaderc_raygen_shader);
    auto rmiss_spv       = compile_glsl(read_file(shader_dir / "miss.rmiss"),             "miss.rmiss",             shaderc_miss_shader);
    auto shadow_miss_spv = compile_glsl(read_file(shader_dir / "shadow_miss.rmiss"),      "shadow_miss.rmiss",      shaderc_miss_shader);
    auto rchit_spv       = compile_glsl(read_file(shader_dir / "closest_hit.rchit"),      "closest_hit.rchit",      shaderc_closesthit_shader);
    auto glass_chit_spv  = compile_glsl(read_file(shader_dir / "glass_closest_hit.rchit"),"glass_closest_hit.rchit",shaderc_closesthit_shader);
    auto glass_ahit_spv  = compile_glsl(read_file(shader_dir / "glass_any_hit.rahit"),    "glass_any_hit.rahit",    shaderc_anyhit_shader);

    VkShaderModule rgen_mod        = make_module(device, rgen_spv);
    VkShaderModule rmiss_mod       = make_module(device, rmiss_spv);
    VkShaderModule shadow_miss_mod = make_module(device, shadow_miss_spv);
    VkShaderModule rchit_mod       = make_module(device, rchit_spv);
    VkShaderModule glass_chit_mod  = make_module(device, glass_chit_spv);
    VkShaderModule glass_ahit_mod  = make_module(device, glass_ahit_spv);

    // Stage indices: 0=rgen, 1=miss, 2=shadow_miss, 3=opaque_chit, 4=glass_chit, 5=glass_ahit
    VkPipelineShaderStageCreateInfo stages[6]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[0].module = rgen_mod;
    stages[0].pName  = "main";

    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[1].module = rmiss_mod;
    stages[1].pName  = "main";

    stages[2].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[2].stage  = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[2].module = shadow_miss_mod;
    stages[2].pName  = "main";

    stages[3].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[3].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[3].module = rchit_mod;
    stages[3].pName  = "main";

    stages[4].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[4].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[4].module = glass_chit_mod;
    stages[4].pName  = "main";

    stages[5].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[5].stage  = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    stages[5].module = glass_ahit_mod;
    stages[5].pName  = "main";

    // ------------------------------------------------------------------ Shader groups
    // Group 0: raygen      (general, stage 0)
    // Group 1: miss        (general, stage 1)                    ← missIndex 0 (scene rays)
    // Group 2: shadow_miss (general, stage 2)                    ← missIndex 1 (shadow rays)
    // Group 3: opaque hit  (triangles, stage 3)                  ← sbt_record_offset 0
    // Group 4: glass hit   (triangles, stages 4+5)               ← sbt_record_offset 1
    VkRayTracingShaderGroupCreateInfoKHR groups[5]{};
    groups[0].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[0].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader      = 0;
    groups[0].closestHitShader   = VK_SHADER_UNUSED_KHR;
    groups[0].anyHitShader       = VK_SHADER_UNUSED_KHR;
    groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;

    groups[1].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[1].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader      = 1;
    groups[1].closestHitShader   = VK_SHADER_UNUSED_KHR;
    groups[1].anyHitShader       = VK_SHADER_UNUSED_KHR;
    groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;

    groups[2].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[2].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[2].generalShader      = 2;  // shadow_miss
    groups[2].closestHitShader   = VK_SHADER_UNUSED_KHR;
    groups[2].anyHitShader       = VK_SHADER_UNUSED_KHR;
    groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

    groups[3].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[3].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[3].generalShader      = VK_SHADER_UNUSED_KHR;
    groups[3].closestHitShader   = 3;  // opaque chit
    groups[3].anyHitShader       = VK_SHADER_UNUSED_KHR;
    groups[3].intersectionShader = VK_SHADER_UNUSED_KHR;

    groups[4].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[4].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[4].generalShader      = VK_SHADER_UNUSED_KHR;
    groups[4].closestHitShader   = 4;  // glass chit
    groups[4].anyHitShader       = 5;  // glass ahit
    groups[4].intersectionShader = VK_SHADER_UNUSED_KHR;

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
    pipe_ci.stageCount                   = 6;
    pipe_ci.pStages                      = stages;
    pipe_ci.groupCount                   = 5;
    pipe_ci.pGroups                      = groups;
    pipe_ci.maxPipelineRayRecursionDepth = 1;  // no recursive bounces for the milestone
    pipe_ci.layout                       = layout;

    if (ctx.pfn_vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                               1, &pipe_ci, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error("RT pipeline creation failed");

    vkDestroyShaderModule(device, rgen_mod,        nullptr);
    vkDestroyShaderModule(device, rmiss_mod,       nullptr);
    vkDestroyShaderModule(device, shadow_miss_mod, nullptr);
    vkDestroyShaderModule(device, rchit_mod,       nullptr);
    vkDestroyShaderModule(device, glass_chit_mod,  nullptr);
    vkDestroyShaderModule(device, glass_ahit_mod,  nullptr);

    // ------------------------------------------------------------------ Shader binding table
    uint32_t handle_size  = ctx.rt_pipeline_props.shaderGroupHandleSize;
    uint32_t handle_align = ctx.rt_pipeline_props.shaderGroupHandleAlignment;
    uint32_t base_align   = ctx.rt_pipeline_props.shaderGroupBaseAlignment;

    auto align_up = [](uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); };
    uint32_t handle_stride = align_up(handle_size, handle_align);

    // Regions: raygen (group 0), miss (groups 1+2 — scene then shadow), hit (groups 3+4 — opaque then glass).
    uint32_t raygen_offset = 0;
    uint32_t miss_offset   = align_up(handle_stride, base_align);
    uint32_t hit_offset    = align_up(miss_offset + 2 * handle_stride, base_align);
    uint32_t sbt_size      = hit_offset + 2 * handle_stride;  // two hit groups

    // Fetch all five handles (rgen, miss, shadow_miss, opaque_hit, glass_hit).
    std::vector<uint8_t> handles(5 * handle_size);
    if (ctx.pfn_vkGetRayTracingShaderGroupHandlesKHR(device, pipeline,
                                                     0, 5, handles.size(), handles.data()) != VK_SUCCESS)
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
    // SBT base address must be aligned to shaderGroupBaseAlignment — use the
    // explicit-alignment variant so VMA satisfies this regardless of suballoc placement.
    if (vmaCreateBufferWithAlignment(ctx.allocator, &sbt_ci, &sbt_ai,
                                     base_align,
                                     &sbt_buf, &sbt_alloc, &sbt_alloc_info) != VK_SUCCESS)
        throw std::runtime_error("SBT buffer creation failed");

    auto* data = static_cast<uint8_t*>(sbt_alloc_info.pMappedData);
    std::memcpy(data + raygen_offset,                  handles.data() + 0 * handle_size, handle_size);
    std::memcpy(data + miss_offset,                    handles.data() + 1 * handle_size, handle_size); // scene miss
    std::memcpy(data + miss_offset + handle_stride,    handles.data() + 2 * handle_size, handle_size); // shadow miss
    std::memcpy(data + hit_offset,                     handles.data() + 3 * handle_size, handle_size); // opaque
    std::memcpy(data + hit_offset + handle_stride,     handles.data() + 4 * handle_size, handle_size); // glass
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
    miss_region.size          = 2 * handle_stride;  // scene miss + shadow miss

    hit_region.deviceAddress = sbt_addr + hit_offset;
    hit_region.stride        = handle_stride;
    hit_region.size          = 2 * handle_stride;  // opaque + glass (hit groups 3,4)
    // callable_region stays zero (no callable shaders)
}

RtPipeline::~RtPipeline() {
    vmaDestroyBuffer(allocator, sbt_buf, sbt_alloc);
    vkDestroyPipeline      (device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, layout,   nullptr);
}

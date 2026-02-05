#include "../deps/wgpu-native/ffi/webgpu-headers/webgpu.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef uint8_t *moonbit_bytes_t;

moonbit_bytes_t moonbit_make_bytes(int32_t size, int init);

static WGPUInstance g_instance = NULL;

static WGPUStringView string_view_empty(void) {
  WGPUStringView view;
  view.data = NULL;
  view.length = 0;
  return view;
}

static WGPUStringView string_view_from_bytes(const uint8_t *data, int len) {
  WGPUStringView view;
  view.data = (const char *)data;
  view.length = (size_t)len;
  return view;
}

typedef struct {
  WGPURequestAdapterStatus status;
  WGPUAdapter adapter;
  bool done;
} request_adapter_state_t;

static void on_request_adapter(
  WGPURequestAdapterStatus status,
  WGPUAdapter adapter,
  WGPUStringView message,
  void *userdata1,
  void *userdata2
) {
  (void)message;
  (void)userdata2;
  request_adapter_state_t *state = (request_adapter_state_t *)userdata1;
  state->status = status;
  state->adapter = adapter;
  state->done = true;
}

typedef struct {
  WGPURequestDeviceStatus status;
  WGPUDevice device;
  bool done;
} request_device_state_t;

static void on_request_device(
  WGPURequestDeviceStatus status,
  WGPUDevice device,
  WGPUStringView message,
  void *userdata1,
  void *userdata2
) {
  (void)message;
  (void)userdata2;
  request_device_state_t *state = (request_device_state_t *)userdata1;
  state->status = status;
  state->device = device;
  state->done = true;
}

typedef struct {
  WGPUMapAsyncStatus status;
  bool done;
} map_state_t;

static void on_buffer_map(
  WGPUMapAsyncStatus status,
  WGPUStringView message,
  void *userdata1,
  void *userdata2
) {
  (void)message;
  (void)userdata2;
  map_state_t *state = (map_state_t *)userdata1;
  state->status = status;
  state->done = true;
}

static WGPUBufferBindingType buffer_binding_type_from_int(int ty) {
  switch (ty) {
    case 2:
      return WGPUBufferBindingType_Uniform;
    case 3:
      return WGPUBufferBindingType_Storage;
    case 4:
      return WGPUBufferBindingType_ReadOnlyStorage;
    default:
      return WGPUBufferBindingType_Undefined;
  }
}

static void fill_bind_group_layout_entry(
  WGPUBindGroupLayoutEntry *entry,
  int binding,
  int visibility,
  int binding_type
) {
  memset(entry, 0, sizeof(*entry));
  entry->binding = (uint32_t)binding;
  entry->visibility = (WGPUShaderStage)visibility;
  entry->buffer.type = buffer_binding_type_from_int(binding_type);
  entry->buffer.hasDynamicOffset = false;
  entry->buffer.minBindingSize = 0;
}

int wgpu_native_ptr_is_null(void *ptr) {
  return ptr == NULL ? 1 : 0;
}

WGPUInstance wgpu_native_instance_new(void) {
  WGPUInstanceDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.nextInChain = NULL;
  desc.features.nextInChain = NULL;
  desc.features.timedWaitAnyEnable = false;
  desc.features.timedWaitAnyMaxCount = 0;
  WGPUInstance instance = wgpuCreateInstance(&desc);
  g_instance = instance;
  return instance;
}

WGPUAdapter wgpu_native_request_adapter(
  WGPUInstance instance,
  int power_preference,
  int force_fallback
) {
  if (instance == NULL) {
    instance = wgpu_native_instance_new();
  }
  g_instance = instance;

  request_adapter_state_t state;
  memset(&state, 0, sizeof(state));

  WGPURequestAdapterOptions options;
  memset(&options, 0, sizeof(options));
  options.featureLevel = WGPUFeatureLevel_Core;
  options.powerPreference = (WGPUPowerPreference)power_preference;
  options.forceFallbackAdapter = force_fallback ? 1 : 0;
  options.backendType = WGPUBackendType_Undefined;
  options.compatibleSurface = NULL;

  WGPURequestAdapterCallbackInfo callback_info;
  memset(&callback_info, 0, sizeof(callback_info));
  callback_info.mode = WGPUCallbackMode_AllowProcessEvents;
  callback_info.callback = on_request_adapter;
  callback_info.userdata1 = &state;
  callback_info.userdata2 = NULL;

  wgpuInstanceRequestAdapter(instance, &options, callback_info);
  while (!state.done) {
    wgpuInstanceProcessEvents(instance);
  }

  if (state.status != WGPURequestAdapterStatus_Success) {
    return NULL;
  }
  return state.adapter;
}

WGPUDevice wgpu_native_request_device(WGPUAdapter adapter) {
  if (adapter == NULL) {
    return NULL;
  }

  request_device_state_t state;
  memset(&state, 0, sizeof(state));

  WGPUDeviceDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.nextInChain = NULL;
  desc.label = string_view_empty();
  desc.requiredFeatureCount = 0;
  desc.requiredFeatures = NULL;
  desc.requiredLimits = NULL;
  desc.defaultQueue.nextInChain = NULL;
  desc.defaultQueue.label = string_view_empty();
  desc.deviceLostCallbackInfo.nextInChain = NULL;
  desc.deviceLostCallbackInfo.callback = NULL;
  desc.deviceLostCallbackInfo.userdata1 = NULL;
  desc.deviceLostCallbackInfo.userdata2 = NULL;
  desc.uncapturedErrorCallbackInfo.nextInChain = NULL;
  desc.uncapturedErrorCallbackInfo.callback = NULL;
  desc.uncapturedErrorCallbackInfo.userdata1 = NULL;
  desc.uncapturedErrorCallbackInfo.userdata2 = NULL;

  WGPURequestDeviceCallbackInfo callback_info;
  memset(&callback_info, 0, sizeof(callback_info));
  callback_info.mode = WGPUCallbackMode_AllowProcessEvents;
  callback_info.callback = on_request_device;
  callback_info.userdata1 = &state;
  callback_info.userdata2 = NULL;

  wgpuAdapterRequestDevice(adapter, &desc, callback_info);
  while (!state.done) {
    if (g_instance != NULL) {
      wgpuInstanceProcessEvents(g_instance);
    }
  }

  if (state.status != WGPURequestDeviceStatus_Success) {
    return NULL;
  }
  return state.device;
}

WGPUQueue wgpu_native_device_get_queue(WGPUDevice device) {
  if (device == NULL) {
    return NULL;
  }
  return wgpuDeviceGetQueue(device);
}

WGPUBuffer wgpu_native_device_create_buffer(
  WGPUDevice device,
  int size,
  int usage,
  int mapped_at_creation
) {
  if (device == NULL) {
    return NULL;
  }
  WGPUBufferDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.nextInChain = NULL;
  desc.label = string_view_empty();
  desc.size = (uint64_t)size;
  desc.usage = (WGPUBufferUsage)usage;
  desc.mappedAtCreation = mapped_at_creation ? 1 : 0;
  return wgpuDeviceCreateBuffer(device, &desc);
}

WGPUShaderModule wgpu_native_device_create_shader_module(
  WGPUDevice device,
  const uint8_t *code,
  int code_len
) {
  if (device == NULL || code == NULL || code_len <= 0) {
    return NULL;
  }

  WGPUShaderSourceWGSL source;
  memset(&source, 0, sizeof(source));
  source.chain.sType = WGPUSType_ShaderSourceWGSL;
  source.chain.next = NULL;
  source.code = string_view_from_bytes(code, code_len);

  WGPUShaderModuleDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.nextInChain = (WGPUChainedStruct const *)&source;
  desc.label = string_view_empty();

  return wgpuDeviceCreateShaderModule(device, &desc);
}

WGPUBindGroupLayout wgpu_native_device_create_bind_group_layout_1(
  WGPUDevice device,
  int binding0,
  int visibility0,
  int type0
) {
  if (device == NULL) {
    return NULL;
  }
  WGPUBindGroupLayoutEntry entries[1];
  fill_bind_group_layout_entry(&entries[0], binding0, visibility0, type0);

  WGPUBindGroupLayoutDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.nextInChain = NULL;
  desc.label = string_view_empty();
  desc.entryCount = 1;
  desc.entries = entries;

  return wgpuDeviceCreateBindGroupLayout(device, &desc);
}

WGPUBindGroupLayout wgpu_native_device_create_bind_group_layout_2(
  WGPUDevice device,
  int binding0,
  int visibility0,
  int type0,
  int binding1,
  int visibility1,
  int type1
) {
  if (device == NULL) {
    return NULL;
  }
  WGPUBindGroupLayoutEntry entries[2];
  fill_bind_group_layout_entry(&entries[0], binding0, visibility0, type0);
  fill_bind_group_layout_entry(&entries[1], binding1, visibility1, type1);

  WGPUBindGroupLayoutDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.nextInChain = NULL;
  desc.label = string_view_empty();
  desc.entryCount = 2;
  desc.entries = entries;

  return wgpuDeviceCreateBindGroupLayout(device, &desc);
}

WGPUBindGroupLayout wgpu_native_device_create_bind_group_layout_3(
  WGPUDevice device,
  int binding0,
  int visibility0,
  int type0,
  int binding1,
  int visibility1,
  int type1,
  int binding2,
  int visibility2,
  int type2
) {
  if (device == NULL) {
    return NULL;
  }
  WGPUBindGroupLayoutEntry entries[3];
  fill_bind_group_layout_entry(&entries[0], binding0, visibility0, type0);
  fill_bind_group_layout_entry(&entries[1], binding1, visibility1, type1);
  fill_bind_group_layout_entry(&entries[2], binding2, visibility2, type2);

  WGPUBindGroupLayoutDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.nextInChain = NULL;
  desc.label = string_view_empty();
  desc.entryCount = 3;
  desc.entries = entries;

  return wgpuDeviceCreateBindGroupLayout(device, &desc);
}

WGPUBindGroupLayout wgpu_native_device_create_bind_group_layout_4(
  WGPUDevice device,
  int binding0,
  int visibility0,
  int type0,
  int binding1,
  int visibility1,
  int type1,
  int binding2,
  int visibility2,
  int type2,
  int binding3,
  int visibility3,
  int type3
) {
  if (device == NULL) {
    return NULL;
  }
  WGPUBindGroupLayoutEntry entries[4];
  fill_bind_group_layout_entry(&entries[0], binding0, visibility0, type0);
  fill_bind_group_layout_entry(&entries[1], binding1, visibility1, type1);
  fill_bind_group_layout_entry(&entries[2], binding2, visibility2, type2);
  fill_bind_group_layout_entry(&entries[3], binding3, visibility3, type3);

  WGPUBindGroupLayoutDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.nextInChain = NULL;
  desc.label = string_view_empty();
  desc.entryCount = 4;
  desc.entries = entries;

  return wgpuDeviceCreateBindGroupLayout(device, &desc);
}

WGPUBindGroupLayout wgpu_native_device_create_bind_group_layout_5(
  WGPUDevice device,
  int binding0,
  int visibility0,
  int type0,
  int binding1,
  int visibility1,
  int type1,
  int binding2,
  int visibility2,
  int type2,
  int binding3,
  int visibility3,
  int type3,
  int binding4,
  int visibility4,
  int type4
) {
  if (device == NULL) {
    return NULL;
  }
  WGPUBindGroupLayoutEntry entries[5];
  fill_bind_group_layout_entry(&entries[0], binding0, visibility0, type0);
  fill_bind_group_layout_entry(&entries[1], binding1, visibility1, type1);
  fill_bind_group_layout_entry(&entries[2], binding2, visibility2, type2);
  fill_bind_group_layout_entry(&entries[3], binding3, visibility3, type3);
  fill_bind_group_layout_entry(&entries[4], binding4, visibility4, type4);

  WGPUBindGroupLayoutDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.nextInChain = NULL;
  desc.label = string_view_empty();
  desc.entryCount = 5;
  desc.entries = entries;

  return wgpuDeviceCreateBindGroupLayout(device, &desc);
}

WGPUBindGroupLayout wgpu_native_device_create_bind_group_layout_6(
  WGPUDevice device,
  int binding0,
  int visibility0,
  int type0,
  int binding1,
  int visibility1,
  int type1,
  int binding2,
  int visibility2,
  int type2,
  int binding3,
  int visibility3,
  int type3,
  int binding4,
  int visibility4,
  int type4,
  int binding5,
  int visibility5,
  int type5
) {
  if (device == NULL) {
    return NULL;
  }
  WGPUBindGroupLayoutEntry entries[6];
  fill_bind_group_layout_entry(&entries[0], binding0, visibility0, type0);
  fill_bind_group_layout_entry(&entries[1], binding1, visibility1, type1);
  fill_bind_group_layout_entry(&entries[2], binding2, visibility2, type2);
  fill_bind_group_layout_entry(&entries[3], binding3, visibility3, type3);
  fill_bind_group_layout_entry(&entries[4], binding4, visibility4, type4);
  fill_bind_group_layout_entry(&entries[5], binding5, visibility5, type5);

  WGPUBindGroupLayoutDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.nextInChain = NULL;
  desc.label = string_view_empty();
  desc.entryCount = 6;
  desc.entries = entries;

  return wgpuDeviceCreateBindGroupLayout(device, &desc);
}

WGPUBindGroupLayout wgpu_native_device_create_bind_group_layout_7(
  WGPUDevice device,
  int binding0,
  int visibility0,
  int type0,
  int binding1,
  int visibility1,
  int type1,
  int binding2,
  int visibility2,
  int type2,
  int binding3,
  int visibility3,
  int type3,
  int binding4,
  int visibility4,
  int type4,
  int binding5,
  int visibility5,
  int type5,
  int binding6,
  int visibility6,
  int type6
) {
  if (device == NULL) {
    return NULL;
  }
  WGPUBindGroupLayoutEntry entries[7];
  fill_bind_group_layout_entry(&entries[0], binding0, visibility0, type0);
  fill_bind_group_layout_entry(&entries[1], binding1, visibility1, type1);
  fill_bind_group_layout_entry(&entries[2], binding2, visibility2, type2);
  fill_bind_group_layout_entry(&entries[3], binding3, visibility3, type3);
  fill_bind_group_layout_entry(&entries[4], binding4, visibility4, type4);
  fill_bind_group_layout_entry(&entries[5], binding5, visibility5, type5);
  fill_bind_group_layout_entry(&entries[6], binding6, visibility6, type6);

  WGPUBindGroupLayoutDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.nextInChain = NULL;
  desc.label = string_view_empty();
  desc.entryCount = 7;
  desc.entries = entries;

  return wgpuDeviceCreateBindGroupLayout(device, &desc);
}

WGPUBindGroupLayout wgpu_native_device_create_bind_group_layout_8(
  WGPUDevice device,
  int binding0,
  int visibility0,
  int type0,
  int binding1,
  int visibility1,
  int type1,
  int binding2,
  int visibility2,
  int type2,
  int binding3,
  int visibility3,
  int type3,
  int binding4,
  int visibility4,
  int type4,
  int binding5,
  int visibility5,
  int type5,
  int binding6,
  int visibility6,
  int type6,
  int binding7,
  int visibility7,
  int type7
) {
  if (device == NULL) {
    return NULL;
  }
  WGPUBindGroupLayoutEntry entries[8];
  fill_bind_group_layout_entry(&entries[0], binding0, visibility0, type0);
  fill_bind_group_layout_entry(&entries[1], binding1, visibility1, type1);
  fill_bind_group_layout_entry(&entries[2], binding2, visibility2, type2);
  fill_bind_group_layout_entry(&entries[3], binding3, visibility3, type3);
  fill_bind_group_layout_entry(&entries[4], binding4, visibility4, type4);
  fill_bind_group_layout_entry(&entries[5], binding5, visibility5, type5);
  fill_bind_group_layout_entry(&entries[6], binding6, visibility6, type6);
  fill_bind_group_layout_entry(&entries[7], binding7, visibility7, type7);

  WGPUBindGroupLayoutDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.nextInChain = NULL;
  desc.label = string_view_empty();
  desc.entryCount = 8;
  desc.entries = entries;

  return wgpuDeviceCreateBindGroupLayout(device, &desc);
}

WGPUBindGroup wgpu_native_device_create_bind_group_1(
  WGPUDevice device,
  WGPUBindGroupLayout layout,
  int binding0,
  WGPUBuffer buffer0,
  int offset0,
  int size0
) {
  if (device == NULL || layout == NULL) {
    return NULL;
  }
  WGPUBindGroupEntry entries[1];
  memset(entries, 0, sizeof(entries));
  entries[0].binding = (uint32_t)binding0;
  entries[0].buffer = buffer0;
  entries[0].offset = (uint64_t)offset0;
  entries[0].size = (uint64_t)size0;

  WGPUBindGroupDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.nextInChain = NULL;
  desc.label = string_view_empty();
  desc.layout = layout;
  desc.entryCount = 1;
  desc.entries = entries;

  return wgpuDeviceCreateBindGroup(device, &desc);
}

WGPUBindGroup wgpu_native_device_create_bind_group_2(
  WGPUDevice device,
  WGPUBindGroupLayout layout,
  int binding0,
  WGPUBuffer buffer0,
  int offset0,
  int size0,
  int binding1,
  WGPUBuffer buffer1,
  int offset1,
  int size1
) {
  if (device == NULL || layout == NULL) {
    return NULL;
  }
  WGPUBindGroupEntry entries[2];
  memset(entries, 0, sizeof(entries));
  entries[0].binding = (uint32_t)binding0;
  entries[0].buffer = buffer0;
  entries[0].offset = (uint64_t)offset0;
  entries[0].size = (uint64_t)size0;
  entries[1].binding = (uint32_t)binding1;
  entries[1].buffer = buffer1;
  entries[1].offset = (uint64_t)offset1;
  entries[1].size = (uint64_t)size1;

  WGPUBindGroupDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.nextInChain = NULL;
  desc.label = string_view_empty();
  desc.layout = layout;
  desc.entryCount = 2;
  desc.entries = entries;

  return wgpuDeviceCreateBindGroup(device, &desc);
}

WGPUBindGroup wgpu_native_device_create_bind_group_3(
  WGPUDevice device,
  WGPUBindGroupLayout layout,
  int binding0,
  WGPUBuffer buffer0,
  int offset0,
  int size0,
  int binding1,
  WGPUBuffer buffer1,
  int offset1,
  int size1,
  int binding2,
  WGPUBuffer buffer2,
  int offset2,
  int size2
) {
  if (device == NULL || layout == NULL) {
    return NULL;
  }
  WGPUBindGroupEntry entries[3];
  memset(entries, 0, sizeof(entries));
  entries[0].binding = (uint32_t)binding0;
  entries[0].buffer = buffer0;
  entries[0].offset = (uint64_t)offset0;
  entries[0].size = (uint64_t)size0;
  entries[1].binding = (uint32_t)binding1;
  entries[1].buffer = buffer1;
  entries[1].offset = (uint64_t)offset1;
  entries[1].size = (uint64_t)size1;
  entries[2].binding = (uint32_t)binding2;
  entries[2].buffer = buffer2;
  entries[2].offset = (uint64_t)offset2;
  entries[2].size = (uint64_t)size2;

  WGPUBindGroupDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.nextInChain = NULL;
  desc.label = string_view_empty();
  desc.layout = layout;
  desc.entryCount = 3;
  desc.entries = entries;

  return wgpuDeviceCreateBindGroup(device, &desc);
}

WGPUBindGroup wgpu_native_device_create_bind_group_4(
  WGPUDevice device,
  WGPUBindGroupLayout layout,
  int binding0,
  WGPUBuffer buffer0,
  int offset0,
  int size0,
  int binding1,
  WGPUBuffer buffer1,
  int offset1,
  int size1,
  int binding2,
  WGPUBuffer buffer2,
  int offset2,
  int size2,
  int binding3,
  WGPUBuffer buffer3,
  int offset3,
  int size3
) {
  if (device == NULL || layout == NULL) {
    return NULL;
  }
  WGPUBindGroupEntry entries[4];
  memset(entries, 0, sizeof(entries));
  entries[0].binding = (uint32_t)binding0;
  entries[0].buffer = buffer0;
  entries[0].offset = (uint64_t)offset0;
  entries[0].size = (uint64_t)size0;
  entries[1].binding = (uint32_t)binding1;
  entries[1].buffer = buffer1;
  entries[1].offset = (uint64_t)offset1;
  entries[1].size = (uint64_t)size1;
  entries[2].binding = (uint32_t)binding2;
  entries[2].buffer = buffer2;
  entries[2].offset = (uint64_t)offset2;
  entries[2].size = (uint64_t)size2;
  entries[3].binding = (uint32_t)binding3;
  entries[3].buffer = buffer3;
  entries[3].offset = (uint64_t)offset3;
  entries[3].size = (uint64_t)size3;

  WGPUBindGroupDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.nextInChain = NULL;
  desc.label = string_view_empty();
  desc.layout = layout;
  desc.entryCount = 4;
  desc.entries = entries;

  return wgpuDeviceCreateBindGroup(device, &desc);
}

WGPUBindGroup wgpu_native_device_create_bind_group_5(
  WGPUDevice device,
  WGPUBindGroupLayout layout,
  int binding0,
  WGPUBuffer buffer0,
  int offset0,
  int size0,
  int binding1,
  WGPUBuffer buffer1,
  int offset1,
  int size1,
  int binding2,
  WGPUBuffer buffer2,
  int offset2,
  int size2,
  int binding3,
  WGPUBuffer buffer3,
  int offset3,
  int size3,
  int binding4,
  WGPUBuffer buffer4,
  int offset4,
  int size4
) {
  if (device == NULL || layout == NULL) {
    return NULL;
  }
  WGPUBindGroupEntry entries[5];
  memset(entries, 0, sizeof(entries));
  entries[0].binding = (uint32_t)binding0;
  entries[0].buffer = buffer0;
  entries[0].offset = (uint64_t)offset0;
  entries[0].size = (uint64_t)size0;
  entries[1].binding = (uint32_t)binding1;
  entries[1].buffer = buffer1;
  entries[1].offset = (uint64_t)offset1;
  entries[1].size = (uint64_t)size1;
  entries[2].binding = (uint32_t)binding2;
  entries[2].buffer = buffer2;
  entries[2].offset = (uint64_t)offset2;
  entries[2].size = (uint64_t)size2;
  entries[3].binding = (uint32_t)binding3;
  entries[3].buffer = buffer3;
  entries[3].offset = (uint64_t)offset3;
  entries[3].size = (uint64_t)size3;
  entries[4].binding = (uint32_t)binding4;
  entries[4].buffer = buffer4;
  entries[4].offset = (uint64_t)offset4;
  entries[4].size = (uint64_t)size4;

  WGPUBindGroupDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.nextInChain = NULL;
  desc.label = string_view_empty();
  desc.layout = layout;
  desc.entryCount = 5;
  desc.entries = entries;

  return wgpuDeviceCreateBindGroup(device, &desc);
}

WGPUBindGroup wgpu_native_device_create_bind_group_6(
  WGPUDevice device,
  WGPUBindGroupLayout layout,
  int binding0,
  WGPUBuffer buffer0,
  int offset0,
  int size0,
  int binding1,
  WGPUBuffer buffer1,
  int offset1,
  int size1,
  int binding2,
  WGPUBuffer buffer2,
  int offset2,
  int size2,
  int binding3,
  WGPUBuffer buffer3,
  int offset3,
  int size3,
  int binding4,
  WGPUBuffer buffer4,
  int offset4,
  int size4,
  int binding5,
  WGPUBuffer buffer5,
  int offset5,
  int size5
) {
  if (device == NULL || layout == NULL) {
    return NULL;
  }
  WGPUBindGroupEntry entries[6];
  memset(entries, 0, sizeof(entries));
  entries[0].binding = (uint32_t)binding0;
  entries[0].buffer = buffer0;
  entries[0].offset = (uint64_t)offset0;
  entries[0].size = (uint64_t)size0;
  entries[1].binding = (uint32_t)binding1;
  entries[1].buffer = buffer1;
  entries[1].offset = (uint64_t)offset1;
  entries[1].size = (uint64_t)size1;
  entries[2].binding = (uint32_t)binding2;
  entries[2].buffer = buffer2;
  entries[2].offset = (uint64_t)offset2;
  entries[2].size = (uint64_t)size2;
  entries[3].binding = (uint32_t)binding3;
  entries[3].buffer = buffer3;
  entries[3].offset = (uint64_t)offset3;
  entries[3].size = (uint64_t)size3;
  entries[4].binding = (uint32_t)binding4;
  entries[4].buffer = buffer4;
  entries[4].offset = (uint64_t)offset4;
  entries[4].size = (uint64_t)size4;
  entries[5].binding = (uint32_t)binding5;
  entries[5].buffer = buffer5;
  entries[5].offset = (uint64_t)offset5;
  entries[5].size = (uint64_t)size5;

  WGPUBindGroupDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.nextInChain = NULL;
  desc.label = string_view_empty();
  desc.layout = layout;
  desc.entryCount = 6;
  desc.entries = entries;

  return wgpuDeviceCreateBindGroup(device, &desc);
}

WGPUBindGroup wgpu_native_device_create_bind_group_7(
  WGPUDevice device,
  WGPUBindGroupLayout layout,
  int binding0,
  WGPUBuffer buffer0,
  int offset0,
  int size0,
  int binding1,
  WGPUBuffer buffer1,
  int offset1,
  int size1,
  int binding2,
  WGPUBuffer buffer2,
  int offset2,
  int size2,
  int binding3,
  WGPUBuffer buffer3,
  int offset3,
  int size3,
  int binding4,
  WGPUBuffer buffer4,
  int offset4,
  int size4,
  int binding5,
  WGPUBuffer buffer5,
  int offset5,
  int size5,
  int binding6,
  WGPUBuffer buffer6,
  int offset6,
  int size6
) {
  if (device == NULL || layout == NULL) {
    return NULL;
  }
  WGPUBindGroupEntry entries[7];
  memset(entries, 0, sizeof(entries));
  entries[0].binding = (uint32_t)binding0;
  entries[0].buffer = buffer0;
  entries[0].offset = (uint64_t)offset0;
  entries[0].size = (uint64_t)size0;
  entries[1].binding = (uint32_t)binding1;
  entries[1].buffer = buffer1;
  entries[1].offset = (uint64_t)offset1;
  entries[1].size = (uint64_t)size1;
  entries[2].binding = (uint32_t)binding2;
  entries[2].buffer = buffer2;
  entries[2].offset = (uint64_t)offset2;
  entries[2].size = (uint64_t)size2;
  entries[3].binding = (uint32_t)binding3;
  entries[3].buffer = buffer3;
  entries[3].offset = (uint64_t)offset3;
  entries[3].size = (uint64_t)size3;
  entries[4].binding = (uint32_t)binding4;
  entries[4].buffer = buffer4;
  entries[4].offset = (uint64_t)offset4;
  entries[4].size = (uint64_t)size4;
  entries[5].binding = (uint32_t)binding5;
  entries[5].buffer = buffer5;
  entries[5].offset = (uint64_t)offset5;
  entries[5].size = (uint64_t)size5;
  entries[6].binding = (uint32_t)binding6;
  entries[6].buffer = buffer6;
  entries[6].offset = (uint64_t)offset6;
  entries[6].size = (uint64_t)size6;

  WGPUBindGroupDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.nextInChain = NULL;
  desc.label = string_view_empty();
  desc.layout = layout;
  desc.entryCount = 7;
  desc.entries = entries;

  return wgpuDeviceCreateBindGroup(device, &desc);
}

WGPUBindGroup wgpu_native_device_create_bind_group_8(
  WGPUDevice device,
  WGPUBindGroupLayout layout,
  int binding0,
  WGPUBuffer buffer0,
  int offset0,
  int size0,
  int binding1,
  WGPUBuffer buffer1,
  int offset1,
  int size1,
  int binding2,
  WGPUBuffer buffer2,
  int offset2,
  int size2,
  int binding3,
  WGPUBuffer buffer3,
  int offset3,
  int size3,
  int binding4,
  WGPUBuffer buffer4,
  int offset4,
  int size4,
  int binding5,
  WGPUBuffer buffer5,
  int offset5,
  int size5,
  int binding6,
  WGPUBuffer buffer6,
  int offset6,
  int size6,
  int binding7,
  WGPUBuffer buffer7,
  int offset7,
  int size7
) {
  if (device == NULL || layout == NULL) {
    return NULL;
  }
  WGPUBindGroupEntry entries[8];
  memset(entries, 0, sizeof(entries));
  entries[0].binding = (uint32_t)binding0;
  entries[0].buffer = buffer0;
  entries[0].offset = (uint64_t)offset0;
  entries[0].size = (uint64_t)size0;
  entries[1].binding = (uint32_t)binding1;
  entries[1].buffer = buffer1;
  entries[1].offset = (uint64_t)offset1;
  entries[1].size = (uint64_t)size1;
  entries[2].binding = (uint32_t)binding2;
  entries[2].buffer = buffer2;
  entries[2].offset = (uint64_t)offset2;
  entries[2].size = (uint64_t)size2;
  entries[3].binding = (uint32_t)binding3;
  entries[3].buffer = buffer3;
  entries[3].offset = (uint64_t)offset3;
  entries[3].size = (uint64_t)size3;
  entries[4].binding = (uint32_t)binding4;
  entries[4].buffer = buffer4;
  entries[4].offset = (uint64_t)offset4;
  entries[4].size = (uint64_t)size4;
  entries[5].binding = (uint32_t)binding5;
  entries[5].buffer = buffer5;
  entries[5].offset = (uint64_t)offset5;
  entries[5].size = (uint64_t)size5;
  entries[6].binding = (uint32_t)binding6;
  entries[6].buffer = buffer6;
  entries[6].offset = (uint64_t)offset6;
  entries[6].size = (uint64_t)size6;
  entries[7].binding = (uint32_t)binding7;
  entries[7].buffer = buffer7;
  entries[7].offset = (uint64_t)offset7;
  entries[7].size = (uint64_t)size7;

  WGPUBindGroupDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.nextInChain = NULL;
  desc.label = string_view_empty();
  desc.layout = layout;
  desc.entryCount = 8;
  desc.entries = entries;

  return wgpuDeviceCreateBindGroup(device, &desc);
}

WGPUPipelineLayout wgpu_native_device_create_pipeline_layout(
  WGPUDevice device,
  WGPUBindGroupLayout layout
) {
  if (device == NULL || layout == NULL) {
    return NULL;
  }
  WGPUBindGroupLayout layouts[1] = { layout };
  WGPUPipelineLayoutDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.nextInChain = NULL;
  desc.label = string_view_empty();
  desc.bindGroupLayoutCount = 1;
  desc.bindGroupLayouts = layouts;

  return wgpuDeviceCreatePipelineLayout(device, &desc);
}

WGPUComputePipeline wgpu_native_device_create_compute_pipeline(
  WGPUDevice device,
  WGPUPipelineLayout layout,
  WGPUShaderModule module,
  const uint8_t *entry,
  int entry_len
) {
  if (device == NULL || layout == NULL || module == NULL || entry == NULL || entry_len <= 0) {
    return NULL;
  }

  WGPUComputePipelineDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.nextInChain = NULL;
  desc.label = string_view_empty();
  desc.layout = layout;
  desc.compute.nextInChain = NULL;
  desc.compute.module = module;
  desc.compute.entryPoint = string_view_from_bytes(entry, entry_len);
  desc.compute.constantCount = 0;
  desc.compute.constants = NULL;

  return wgpuDeviceCreateComputePipeline(device, &desc);
}

void wgpu_native_device_dispatch_compute(
  WGPUDevice device,
  WGPUComputePipeline pipeline,
  WGPUBindGroup bind_group,
  int dispatch_x
) {
  if (device == NULL || pipeline == NULL || bind_group == NULL || dispatch_x <= 0) {
    return;
  }

  WGPUCommandEncoderDescriptor encoder_desc;
  memset(&encoder_desc, 0, sizeof(encoder_desc));
  encoder_desc.nextInChain = NULL;
  encoder_desc.label = string_view_empty();

  WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encoder_desc);
  if (encoder == NULL) {
    return;
  }

  WGPUComputePassDescriptor pass_desc;
  memset(&pass_desc, 0, sizeof(pass_desc));
  pass_desc.nextInChain = NULL;
  pass_desc.label = string_view_empty();
  pass_desc.timestampWrites = NULL;

  WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
  wgpuComputePassEncoderSetPipeline(pass, pipeline);
  wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
  wgpuComputePassEncoderDispatchWorkgroups(pass, (uint32_t)dispatch_x, 1, 1);
  wgpuComputePassEncoderEnd(pass);

  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, NULL);
  WGPUQueue queue = wgpuDeviceGetQueue(device);
  if (queue != NULL && cmd != NULL) {
    wgpuQueueSubmit(queue, 1, &cmd);
  }

  if (cmd != NULL) {
    wgpuCommandBufferRelease(cmd);
  }
  wgpuComputePassEncoderRelease(pass);
  wgpuCommandEncoderRelease(encoder);
}

void wgpu_native_queue_write_buffer(
  WGPUQueue queue,
  WGPUBuffer buffer,
  int offset,
  const uint8_t *data,
  int len
) {
  if (queue == NULL || buffer == NULL || data == NULL || len <= 0) {
    return;
  }
  wgpuQueueWriteBuffer(queue, buffer, (uint64_t)offset, data, (size_t)len);
}

moonbit_bytes_t wgpu_native_device_read_buffer_bytes(
  WGPUDevice device,
  WGPUBuffer buffer,
  int size
) {
  if (device == NULL || buffer == NULL || size <= 0) {
    return moonbit_make_bytes(0, 0);
  }

  WGPUBufferDescriptor read_desc;
  memset(&read_desc, 0, sizeof(read_desc));
  read_desc.nextInChain = NULL;
  read_desc.label = string_view_empty();
  read_desc.size = (uint64_t)size;
  read_desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
  read_desc.mappedAtCreation = false;

  WGPUBuffer read_buffer = wgpuDeviceCreateBuffer(device, &read_desc);
  if (read_buffer == NULL) {
    return moonbit_make_bytes(0, 0);
  }

  WGPUCommandEncoderDescriptor encoder_desc;
  memset(&encoder_desc, 0, sizeof(encoder_desc));
  encoder_desc.nextInChain = NULL;
  encoder_desc.label = string_view_empty();

  WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encoder_desc);
  if (encoder == NULL) {
    wgpuBufferRelease(read_buffer);
    return moonbit_make_bytes(0, 0);
  }

  wgpuCommandEncoderCopyBufferToBuffer(
    encoder,
    buffer,
    0,
    read_buffer,
    0,
    (uint64_t)size
  );

  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, NULL);
  WGPUQueue queue = wgpuDeviceGetQueue(device);
  if (queue != NULL && cmd != NULL) {
    wgpuQueueSubmit(queue, 1, &cmd);
  }

  map_state_t map_state;
  memset(&map_state, 0, sizeof(map_state));

  WGPUBufferMapCallbackInfo map_info;
  memset(&map_info, 0, sizeof(map_info));
  map_info.mode = WGPUCallbackMode_AllowProcessEvents;
  map_info.callback = on_buffer_map;
  map_info.userdata1 = &map_state;
  map_info.userdata2 = NULL;

  wgpuBufferMapAsync(read_buffer, WGPUMapMode_Read, 0, (size_t)size, map_info);

  while (!map_state.done) {
    if (g_instance != NULL) {
      wgpuInstanceProcessEvents(g_instance);
    }
  }

  moonbit_bytes_t out = moonbit_make_bytes(size, 0);
  if (map_state.status == WGPUMapAsyncStatus_Success) {
    void *mapped = wgpuBufferGetMappedRange(read_buffer, 0, (size_t)size);
    if (mapped != NULL) {
      memcpy(out, mapped, (size_t)size);
    }
  }

  wgpuBufferUnmap(read_buffer);

  if (cmd != NULL) {
    wgpuCommandBufferRelease(cmd);
  }
  wgpuCommandEncoderRelease(encoder);
  wgpuBufferRelease(read_buffer);

  return out;
}

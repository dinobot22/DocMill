<template>
  <div class="space-y-6">
    <!-- 页面标题 -->
    <div class="flex justify-between items-center">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">模型管理</h1>
        <p class="mt-1 text-sm text-gray-500">管理和监控 OCR 模型</p>
      </div>
      <button
        @click="showRegisterModal = true"
        class="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 transition-colors"
      >
        注册模型
      </button>
    </div>

    <!-- 模型列表 -->
    <div v-if="appStore.loading" class="text-center py-12">
      <svg class="animate-spin h-8 w-8 text-primary-600 mx-auto" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
      </svg>
      <p class="mt-2 text-gray-500">加载中...</p>
    </div>

    <div v-else-if="!appStore.models.length" class="text-center py-12 bg-white rounded-lg border border-gray-200">
      <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
      </svg>
      <p class="mt-2 text-gray-500">暂无注册模型</p>
      <button
        @click="showRegisterModal = true"
        class="mt-4 px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700"
      >
        注册第一个模型
      </button>
    </div>

    <div v-else class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      <div
        v-for="model in appStore.models"
        :key="model.name"
        class="bg-white rounded-lg border border-gray-200 p-4 hover:shadow-md transition-shadow"
      >
        <!-- 模型头部 -->
        <div class="flex items-start justify-between">
          <div>
            <h3 class="text-lg font-medium text-gray-900">{{ model.name }}</h3>
            <p class="text-sm text-gray-500">{{ model.engine }}</p>
          </div>
          <span
            class="px-2 py-1 text-xs font-medium rounded-full"
            :class="statusClass(model.status)"
          >
            {{ statusText(model.status) }}
          </span>
        </div>

        <!-- 模型信息 -->
        <div class="mt-4 space-y-2 text-sm">
          <div class="flex justify-between">
            <span class="text-gray-500">需要 vLLM</span>
            <span :class="model.requires_vllm ? 'text-green-600' : 'text-gray-400'">
              {{ model.requires_vllm ? '是' : '否' }}
            </span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-500">显存估算</span>
            <span class="text-gray-900">{{ model.vram_estimate_mb }} MB</span>
          </div>
          <div v-if="model.vllm_endpoint" class="flex justify-between">
            <span class="text-gray-500">vLLM 端点</span>
            <span class="text-gray-900 text-xs truncate max-w-[150px]">{{ model.vllm_endpoint }}</span>
          </div>
        </div>

        <!-- 操作按钮 -->
        <div class="mt-4 flex space-x-2">
          <button
            v-if="!model.is_loaded"
            @click="handleLoad(model.name)"
            :disabled="loadingModels.has(model.name)"
            class="flex-1 px-3 py-2 bg-primary-600 text-white text-sm rounded-md hover:bg-primary-700 disabled:bg-gray-300"
          >
            {{ loadingModels.has(model.name) ? '加载中...' : '加载' }}
          </button>
          <button
            v-else
            @click="handleUnload(model.name)"
            :disabled="loadingModels.has(model.name)"
            class="flex-1 px-3 py-2 bg-gray-100 text-gray-700 text-sm rounded-md hover:bg-gray-200 disabled:bg-gray-50"
          >
            {{ loadingModels.has(model.name) ? '卸载中...' : '卸载' }}
          </button>
          <button
            @click="handleUnregister(model.name)"
            class="px-3 py-2 border border-red-300 text-red-600 text-sm rounded-md hover:bg-red-50"
          >
            删除
          </button>
        </div>
      </div>
    </div>

    <!-- 注册模型弹窗 -->
    <div v-if="showRegisterModal" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div class="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
        <div class="px-6 py-4 border-b border-gray-200">
          <h3 class="text-lg font-medium text-gray-900">注册模型</h3>
        </div>
        <div class="px-6 py-4 space-y-4">
          <!-- 模型名称 -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">模型名称 *</label>
            <input
              v-model="registerForm.name"
              type="text"
              class="w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
              placeholder="my-ocr-model"
            />
          </div>

          <!-- Engine 选择 -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Engine 类型 *</label>
            <select
              v-model="registerForm.engine_name"
              class="w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
            >
              <option value="">-- 请选择 --</option>
              <option v-for="engine in appStore.engines" :key="engine.name" :value="engine.name">
                {{ engine.name }} {{ engine.requires_vllm ? '(需要 vLLM)' : '' }}
              </option>
            </select>
          </div>

          <!-- vLLM 模型路径 -->
          <div v-if="selectedEngine?.requires_vllm">
            <label class="block text-sm font-medium text-gray-700 mb-1">vLLM 模型路径 *</label>
            <input
              v-model="registerForm.vllm_model_path"
              type="text"
              class="w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
              placeholder="/models/my-llm 或 huggingface/model-name"
            />
          </div>

          <!-- 显存使用率 -->
          <div v-if="selectedEngine?.requires_vllm">
            <label class="block text-sm font-medium text-gray-700 mb-1">GPU 显存使用率</label>
            <input
              v-model.number="registerForm.gpu_memory_utilization"
              type="number"
              min="0.1"
              max="1.0"
              step="0.1"
              class="w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
            />
          </div>

          <!-- vLLM 服务模型名称 -->
          <div v-if="selectedEngine?.requires_vllm">
            <label class="block text-sm font-medium text-gray-700 mb-1">vLLM 服务模型名称</label>
            <input
              v-model="registerForm.served_model_name"
              type="text"
              class="w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
              placeholder="如: PaddleOCR-VL-0.9B"
            />
            <p class="mt-1 text-xs text-gray-500">vLLM 服务注册的模型名称，用于 API 调用</p>
          </div>
        </div>
        <div class="px-6 py-4 border-t border-gray-200 flex justify-end space-x-3">
          <button
            @click="showRegisterModal = false"
            class="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
          >
            取消
          </button>
          <button
            @click="handleRegister"
            :disabled="!canRegister || registering"
            class="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 disabled:bg-gray-300"
          >
            {{ registering ? '注册中...' : '注册' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useAppStore } from '@/stores/app'

const appStore = useAppStore()

// 状态
const showRegisterModal = ref(false)
const registering = ref(false)
const loadingModels = ref(new Set<string>())

const registerForm = ref({
  name: '',
  engine_name: '',
  vllm_model_path: '',
  gpu_memory_utilization: 0.8,
  served_model_name: '',
})

// 计算属性
const selectedEngine = computed(() => {
  return appStore.engines.find((e) => e.name === registerForm.value.engine_name)
})

const canRegister = computed(() => {
  if (!registerForm.value.name || !registerForm.value.engine_name) return false
  if (selectedEngine.value?.requires_vllm && !registerForm.value.vllm_model_path) return false
  return true
})

// 方法
function statusClass(status: string) {
  const classes: Record<string, string> = {
    cold: 'bg-gray-100 text-gray-600',
    loading: 'bg-yellow-100 text-yellow-600',
    ready: 'bg-green-100 text-green-600',
    idle: 'bg-blue-100 text-blue-600',
  }
  return classes[status] || 'bg-gray-100 text-gray-600'
}

function statusText(status: string) {
  const texts: Record<string, string> = {
    cold: '未加载',
    loading: '加载中',
    ready: '已就绪',
    idle: '空闲',
  }
  return texts[status] || status
}

async function handleLoad(name: string) {
  loadingModels.value.add(name)
  try {
    await appStore.loadModel(name)
  } catch (e: any) {
    alert(`加载失败: ${e.message}`)
  } finally {
    loadingModels.value.delete(name)
  }
}

async function handleUnload(name: string) {
  loadingModels.value.add(name)
  try {
    await appStore.unloadModel(name)
  } catch (e: any) {
    alert(`卸载失败: ${e.message}`)
  } finally {
    loadingModels.value.delete(name)
  }
}

async function handleUnregister(name: string) {
  if (!confirm(`确定要删除模型 "${name}" 吗？`)) return
  try {
    await appStore.unregisterModel(name)
  } catch (e: any) {
    alert(`删除失败: ${e.message}`)
  }
}

async function handleRegister() {
  registering.value = true
  try {
    const data: any = {
      name: registerForm.value.name,
      engine_name: registerForm.value.engine_name,
    }

    if (selectedEngine.value?.requires_vllm) {
      data.vllm_config = {
        model_path: registerForm.value.vllm_model_path,
        gpu_memory_utilization: registerForm.value.gpu_memory_utilization,
      }
      // 添加 served_model_name（如果填写）
      if (registerForm.value.served_model_name) {
        data.vllm_config.served_model_name = registerForm.value.served_model_name
      }
      data.engine_kwargs = {
        vllm_model_path: registerForm.value.vllm_model_path,
      }
      // 传递 vllm_model_name 给 engine
      if (registerForm.value.served_model_name) {
        data.engine_kwargs.vllm_model_name = registerForm.value.served_model_name
      }
    }

    await appStore.registerModel(data)
    showRegisterModal.value = false
    resetForm()
  } catch (e: any) {
    alert(`注册失败: ${e.message}`)
  } finally {
    registering.value = false
  }
}

function resetForm() {
  registerForm.value = {
    name: '',
    engine_name: '',
    vllm_model_path: '',
    gpu_memory_utilization: 0.8,
    served_model_name: '',
  }
}

// 生命周期
onMounted(() => {
  appStore.fetchModels()
  appStore.fetchEngines()
})
</script>
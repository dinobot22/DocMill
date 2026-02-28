import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { modelsApi, type ModelInfo, type EngineInfo } from '@/api'

export const useAppStore = defineStore('app', () => {
  // 状态
  const models = ref<ModelInfo[]>([])
  const engines = ref<EngineInfo[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  // 计算属性
  const readyModels = computed(() => models.value.filter((m) => m.status === 'ready' || m.status === 'idle'))
  const loadedModels = computed(() => models.value.filter((m) => m.is_loaded))

  // 操作
  async function fetchModels() {
    loading.value = true
    error.value = null
    try {
      models.value = await modelsApi.list()
    } catch (e: any) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  async function fetchEngines() {
    try {
      engines.value = await modelsApi.listEngines()
    } catch (e: any) {
      console.error('获取 Engine 列表失败:', e)
    }
  }

  async function loadModel(name: string) {
    try {
      await modelsApi.load(name)
      await fetchModels()
    } catch (e: any) {
      throw e
    }
  }

  async function unloadModel(name: string) {
    try {
      await modelsApi.unload(name)
      await fetchModels()
    } catch (e: any) {
      throw e
    }
  }

  async function registerModel(data: Parameters<typeof modelsApi.register>[0]) {
    try {
      await modelsApi.register(data)
      await fetchModels()
    } catch (e: any) {
      throw e
    }
  }

  async function unregisterModel(name: string) {
    try {
      await modelsApi.unregister(name)
      await fetchModels()
    } catch (e: any) {
      throw e
    }
  }

  return {
    // 状态
    models,
    engines,
    loading,
    error,
    // 计算属性
    readyModels,
    loadedModels,
    // 操作
    fetchModels,
    fetchEngines,
    loadModel,
    unloadModel,
    registerModel,
    unregisterModel,
  }
})
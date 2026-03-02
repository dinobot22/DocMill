<template>
  <div class="space-y-6">
    <!-- 页面标题 -->
    <div class="flex justify-between items-center">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">GPU 监控</h1>
        <p class="mt-1 text-sm text-gray-500">实时查看 NVIDIA GPU 资源使用情况</p>
      </div>
      <div class="flex items-center space-x-4">
        <span v-if="lastRefresh" class="text-sm text-gray-500">
          最后更新: {{ lastRefresh.toLocaleTimeString() }}
        </span>
        <button
          @click="refresh"
          :disabled="loading"
          class="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
        >
          <svg
            class="w-4 h-4"
            :class="{ 'animate-spin': loading }"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
            />
          </svg>
          <span>刷新</span>
        </button>
      </div>
    </div>

    <!-- 错误提示 -->
    <div v-if="error" class="bg-red-50 border border-red-200 rounded-lg p-4">
      <div class="flex items-center">
        <svg class="w-5 h-5 text-red-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
        <span class="text-red-700">{{ error }}</span>
      </div>
    </div>

    <!-- GPU 不可用 -->
    <div v-else-if="!gpuStatus?.available" class="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
      <div class="flex items-center">
        <svg class="w-5 h-5 text-yellow-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>
        <span class="text-yellow-700">{{ gpuStatus?.error || 'GPU 不可用' }}</span>
      </div>
    </div>

    <!-- GPU 列表 -->
    <div v-else class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
      <div
        v-for="gpu in gpuStatus?.gpus"
        :key="gpu.index"
        class="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden"
      >
        <!-- GPU 头部 -->
        <div class="bg-gray-800 text-white px-4 py-3">
          <div class="flex items-center justify-between">
            <div class="flex items-center space-x-2">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
                />
              </svg>
              <span class="font-medium">GPU {{ gpu.index }}</span>
            </div>
            <span class="text-xs text-gray-300 truncate max-w-[200px]">{{ gpu.name }}</span>
          </div>
        </div>

        <!-- GPU 内容 -->
        <div class="p-4 space-y-4">
          <!-- 显存使用 -->
          <div>
            <div class="flex justify-between text-sm mb-1">
              <span class="text-gray-600">显存</span>
              <span class="text-gray-900 font-medium">
                {{ formatMemory(gpu.memory_used_mb) }} / {{ formatMemory(gpu.memory_total_mb) }}
              </span>
            </div>
            <div class="w-full bg-gray-200 rounded-full h-2.5">
              <div
                class="h-2.5 rounded-full transition-all duration-300"
                :class="getUtilizationColor(gpu.memory_utilization)"
                :style="{ width: `${gpu.memory_utilization}%` }"
              ></div>
            </div>
            <div class="text-right text-xs text-gray-500 mt-1">{{ gpu.memory_utilization.toFixed(1) }}%</div>
          </div>

          <!-- GPU 利用率 -->
          <div>
            <div class="flex justify-between text-sm mb-1">
              <span class="text-gray-600">GPU 利用率</span>
              <span class="text-gray-900 font-medium">{{ gpu.gpu_utilization.toFixed(1) }}%</span>
            </div>
            <div class="w-full bg-gray-200 rounded-full h-2.5">
              <div
                class="h-2.5 rounded-full transition-all duration-300"
                :class="getUtilizationColor(gpu.gpu_utilization)"
                :style="{ width: `${gpu.gpu_utilization}%` }"
              ></div>
            </div>
          </div>

          <!-- 温度和功耗 -->
          <div class="grid grid-cols-2 gap-4 pt-2 border-t border-gray-100">
            <div>
              <div class="text-xs text-gray-500 mb-1">温度</div>
              <div class="flex items-center space-x-1">
                <svg class="w-4 h-4 text-orange-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                  />
                </svg>
                <span class="text-lg font-semibold" :class="getTempColor(gpu.temperature)">
                  {{ gpu.temperature }}°C
                </span>
              </div>
            </div>
            <div>
              <div class="text-xs text-gray-500 mb-1">功耗</div>
              <div class="flex items-center space-x-1">
                <svg class="w-4 h-4 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M13 10V3L4 14h7v7l9-11h-7z"
                  />
                </svg>
                <span class="text-lg font-semibold text-gray-900">
                  {{ gpu.power_draw.toFixed(0) }}W
                </span>
                <span class="text-xs text-gray-400">/ {{ gpu.power_limit.toFixed(0) }}W</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 空状态 -->
    <div
      v-if="gpuStatus?.available && gpuStatus?.gpus.length === 0"
      class="text-center py-12 bg-white rounded-lg border border-gray-200"
    >
      <svg class="mx-auto h-12 w-12 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="2"
          d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
        />
      </svg>
      <p class="mt-2 text-gray-500">未检测到 GPU 设备</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { gpuApi, type GpuStatus } from '@/api'

// 状态
const gpuStatus = ref<GpuStatus | null>(null)
const loading = ref(false)
const error = ref('')
const lastRefresh = ref<Date | null>(null)

let refreshTimer: ReturnType<typeof setInterval> | null = null

// 方法
async function refresh() {
  loading.value = true
  error.value = ''
  try {
    gpuStatus.value = await gpuApi.getStatus()
    lastRefresh.value = new Date()
  } catch (e: any) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

function formatMemory(mb: number): string {
  if (mb >= 1024) {
    return `${(mb / 1024).toFixed(1)} GB`
  }
  return `${mb} MB`
}

function getUtilizationColor(util: number): string {
  if (util >= 80) return 'bg-red-500'
  if (util >= 60) return 'bg-yellow-500'
  return 'bg-green-500'
}

function getTempColor(temp: number): string {
  if (temp >= 80) return 'text-red-600'
  if (temp >= 60) return 'text-yellow-600'
  return 'text-gray-900'
}

// 生命周期
onMounted(() => {
  refresh()
  // 每 10 秒自动刷新
  refreshTimer = setInterval(refresh, 10000)
})

onUnmounted(() => {
  if (refreshTimer) {
    clearInterval(refreshTimer)
  }
})
</script>
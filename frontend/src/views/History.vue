<template>
  <div class="space-y-6">
    <!-- 页面标题 -->
    <div>
      <h1 class="text-2xl font-bold text-gray-900">历史记录</h1>
      <p class="mt-1 text-sm text-gray-500">查看 OCR 识别历史</p>
    </div>

    <!-- 筛选栏 -->
    <div class="bg-white rounded-lg border border-gray-200 p-4">
      <div class="flex items-center space-x-4">
        <div>
          <label class="block text-xs text-gray-500 mb-1">模型</label>
          <select
            v-model="filterModel"
            class="rounded-md border-gray-300 text-sm focus:border-primary-500 focus:ring-primary-500"
            @change="fetchHistory"
          >
            <option value="">全部</option>
            <option v-for="model in appStore.models" :key="model.name" :value="model.name">
              {{ model.name }}
            </option>
          </select>
        </div>
        <div>
          <label class="block text-xs text-gray-500 mb-1">状态</label>
          <select
            v-model="filterStatus"
            class="rounded-md border-gray-300 text-sm focus:border-primary-500 focus:ring-primary-500"
            @change="fetchHistory"
          >
            <option value="">全部</option>
            <option value="completed">已完成</option>
            <option value="failed">失败</option>
            <option value="processing">处理中</option>
          </select>
        </div>
        <div class="flex-1"></div>
        <button
          @click="fetchHistory"
          class="px-4 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200"
        >
          刷新
        </button>
      </div>
    </div>

    <!-- 历史列表 -->
    <div v-if="loading" class="text-center py-12">
      <svg class="animate-spin h-8 w-8 text-primary-600 mx-auto" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
      </svg>
      <p class="mt-2 text-gray-500">加载中...</p>
    </div>

    <div v-else-if="!historyItems.length" class="text-center py-12 bg-white rounded-lg border border-gray-200">
      <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <p class="mt-2 text-gray-500">暂无历史记录</p>
    </div>

    <div v-else class="space-y-4">
      <!-- 记录卡片 -->
      <div
        v-for="item in historyItems"
        :key="item.id"
        class="bg-white rounded-lg border border-gray-200 p-4 hover:shadow-md transition-shadow cursor-pointer"
        @click="showDetail(item)"
      >
        <div class="flex items-start justify-between">
          <div class="flex items-center space-x-4">
            <!-- 状态图标 -->
            <div
              class="w-10 h-10 rounded-full flex items-center justify-center"
              :class="statusBgClass(item.status)"
            >
              <svg v-if="item.status === 'completed'" class="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
              </svg>
              <svg v-else-if="item.status === 'failed'" class="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
              <svg v-else class="w-5 h-5 text-yellow-600 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
              </svg>
            </div>
            <!-- 信息 -->
            <div>
              <p class="text-sm font-medium text-gray-900">{{ item.filename }}</p>
              <p class="text-xs text-gray-500">
                {{ item.model }} · {{ formatTime(item.created_at) }}
              </p>
            </div>
          </div>
          <!-- 状态标签 -->
          <span
            class="px-2 py-1 text-xs font-medium rounded-full"
            :class="statusClass(item.status)"
          >
            {{ statusText(item.status) }}
          </span>
        </div>
        <!-- 结果预览 -->
        <div v-if="item.status === 'completed' && item.result_text" class="mt-3 p-3 bg-gray-50 rounded text-sm text-gray-600 line-clamp-2">
          {{ item.result_text.slice(0, 200) }}{{ item.result_text.length > 200 ? '...' : '' }}
        </div>
        <!-- 错误信息 -->
        <div v-else-if="item.status === 'failed' && item.error" class="mt-3 p-3 bg-red-50 rounded text-sm text-red-600">
          {{ item.error }}
        </div>
      </div>

      <!-- 分页 -->
      <div v-if="total > pageSize" class="flex justify-center items-center space-x-4">
        <button
          @click="prevPage"
          :disabled="offset === 0"
          class="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 disabled:opacity-50"
        >
          上一页
        </button>
        <span class="text-sm text-gray-500">
          共 {{ total }} 条记录
        </span>
        <button
          @click="nextPage"
          :disabled="offset + pageSize >= total"
          class="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 disabled:opacity-50"
        >
          下一页
        </button>
      </div>
    </div>

    <!-- 详情弹窗 -->
    <div v-if="selectedItem" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div class="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[80vh] flex flex-col">
        <div class="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
          <h3 class="text-lg font-medium text-gray-900">{{ selectedItem.filename }}</h3>
          <button @click="selectedItem = null" class="text-gray-400 hover:text-gray-600">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div class="px-6 py-4 flex-1 overflow-auto">
          <!-- 信息 -->
          <div class="flex items-center space-x-4 text-sm text-gray-500 mb-4">
            <span>模型: {{ selectedItem.model }}</span>
            <span>状态: {{ statusText(selectedItem.status) }}</span>
            <span>时间: {{ formatTime(selectedItem.created_at) }}</span>
          </div>
          <!-- 结果 -->
          <div v-if="selectedItem.status === 'completed'" class="space-y-4">
            <div>
              <h4 class="text-sm font-medium text-gray-700 mb-2">识别结果</h4>
              <pre class="bg-gray-50 p-4 rounded-md text-sm overflow-auto max-h-[300px] whitespace-pre-wrap">{{ selectedItem.result_text }}</pre>
            </div>
          </div>
          <!-- 错误 -->
          <div v-else-if="selectedItem.status === 'failed'" class="p-4 bg-red-50 rounded-md text-red-600">
            {{ selectedItem.error }}
          </div>
        </div>
        <div class="px-6 py-4 border-t border-gray-200 flex justify-between">
          <div class="flex space-x-2">
            <button
              v-if="selectedItem.status === 'completed'"
              @click="downloadResult(selectedItem.id, 'txt')"
              class="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
            >
              下载 TXT
            </button>
            <button
              v-if="selectedItem.status === 'completed'"
              @click="downloadResult(selectedItem.id, 'md')"
              class="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
            >
              下载 MD
            </button>
          </div>
          <button
            @click="deleteRecord(selectedItem.id)"
            class="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
          >
            删除记录
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useAppStore } from '@/stores/app'
import { historyApi, type HistoryRecord } from '@/api'

const appStore = useAppStore()

// 状态
const loading = ref(false)
const historyItems = ref<HistoryRecord[]>([])
const total = ref(0)
const pageSize = 20
const offset = ref(0)
const filterModel = ref('')
const filterStatus = ref('')
const selectedItem = ref<HistoryRecord | null>(null)

// 方法
function statusClass(status: string) {
  const classes: Record<string, string> = {
    pending: 'bg-gray-100 text-gray-600',
    processing: 'bg-yellow-100 text-yellow-600',
    completed: 'bg-green-100 text-green-600',
    failed: 'bg-red-100 text-red-600',
  }
  return classes[status] || 'bg-gray-100 text-gray-600'
}

function statusBgClass(status: string) {
  const classes: Record<string, string> = {
    pending: 'bg-gray-100',
    processing: 'bg-yellow-100',
    completed: 'bg-green-100',
    failed: 'bg-red-100',
  }
  return classes[status] || 'bg-gray-100'
}

function statusText(status: string) {
  const texts: Record<string, string> = {
    pending: '等待中',
    processing: '处理中',
    completed: '已完成',
    failed: '失败',
  }
  return texts[status] || status
}

function formatTime(isoString: string) {
  const date = new Date(isoString)
  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

async function fetchHistory() {
  loading.value = true
  try {
    const res = await historyApi.list({
      limit: pageSize,
      offset: offset.value,
      model: filterModel.value || undefined,
      status: filterStatus.value || undefined,
    })
    historyItems.value = res.items
    total.value = res.total
  } catch (e) {
    console.error('获取历史记录失败:', e)
  } finally {
    loading.value = false
  }
}

function prevPage() {
  if (offset.value >= pageSize) {
    offset.value -= pageSize
    fetchHistory()
  }
}

function nextPage() {
  if (offset.value + pageSize < total.value) {
    offset.value += pageSize
    fetchHistory()
  }
}

function showDetail(item: HistoryRecord) {
  selectedItem.value = item
}

async function downloadResult(id: string, format: 'txt' | 'md' | 'json') {
  try {
    const blob = await historyApi.download(id, format)
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `result.${format}`
    a.click()
    URL.revokeObjectURL(url)
  } catch (e) {
    console.error('下载失败:', e)
  }
}

async function deleteRecord(id: string) {
  if (!confirm('确定要删除这条记录吗？')) return
  try {
    await historyApi.delete(id)
    selectedItem.value = null
    await fetchHistory()
  } catch (e) {
    console.error('删除失败:', e)
  }
}

// 生命周期
onMounted(() => {
  appStore.fetchModels()
  fetchHistory()
})
</script>
<template>
  <div class="space-y-6">
    <!-- 页面标题 -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">任务队列</h1>
        <p class="mt-1 text-sm text-gray-500">管理异步 OCR 识别任务</p>
      </div>
      <button
        @click="refresh"
        class="px-4 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 flex items-center space-x-2"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
        <span>刷新</span>
      </button>
    </div>

    <!-- 统计卡片 -->
    <div class="grid grid-cols-5 gap-4">
      <div class="bg-white rounded-lg border border-gray-200 p-4">
        <div class="flex items-center">
          <div class="w-10 h-10 rounded-full bg-gray-100 flex items-center justify-center">
            <svg class="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div class="ml-3">
            <p class="text-2xl font-bold text-gray-900">{{ stats.pending }}</p>
            <p class="text-sm text-gray-500">等待中</p>
          </div>
        </div>
      </div>
      <div class="bg-white rounded-lg border border-gray-200 p-4">
        <div class="flex items-center">
          <div class="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center">
            <svg class="w-5 h-5 text-blue-600 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
            </svg>
          </div>
          <div class="ml-3">
            <p class="text-2xl font-bold text-blue-600">{{ stats.processing }}</p>
            <p class="text-sm text-gray-500">处理中</p>
          </div>
        </div>
      </div>
      <div class="bg-white rounded-lg border border-gray-200 p-4">
        <div class="flex items-center">
          <div class="w-10 h-10 rounded-full bg-green-100 flex items-center justify-center">
            <svg class="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <div class="ml-3">
            <p class="text-2xl font-bold text-green-600">{{ stats.completed }}</p>
            <p class="text-sm text-gray-500">已完成</p>
          </div>
        </div>
      </div>
      <div class="bg-white rounded-lg border border-gray-200 p-4">
        <div class="flex items-center">
          <div class="w-10 h-10 rounded-full bg-red-100 flex items-center justify-center">
            <svg class="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </div>
          <div class="ml-3">
            <p class="text-2xl font-bold text-red-600">{{ stats.failed }}</p>
            <p class="text-sm text-gray-500">失败</p>
          </div>
        </div>
      </div>
      <div class="bg-white rounded-lg border border-gray-200 p-4">
        <div class="flex items-center">
          <div class="w-10 h-10 rounded-full bg-yellow-100 flex items-center justify-center">
            <svg class="w-5 h-5 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
            </svg>
          </div>
          <div class="ml-3">
            <p class="text-2xl font-bold text-yellow-600">{{ stats.cancelled }}</p>
            <p class="text-sm text-gray-500">已取消</p>
          </div>
        </div>
      </div>
    </div>

    <!-- 筛选栏 -->
    <div class="bg-white rounded-lg border border-gray-200 p-4">
      <div class="flex items-center space-x-4">
        <div>
          <label class="block text-xs text-gray-500 mb-1">状态筛选</label>
          <select
            v-model="filterStatus"
            class="rounded-md border-gray-300 text-sm focus:border-primary-500 focus:ring-primary-500"
            @change="fetchTasks"
          >
            <option value="">全部</option>
            <option value="pending">等待中</option>
            <option value="processing">处理中</option>
            <option value="completed">已完成</option>
            <option value="failed">失败</option>
            <option value="cancelled">已取消</option>
          </select>
        </div>
        <div class="flex-1"></div>
        <span class="text-sm text-gray-500">共 {{ total }} 个任务</span>
      </div>
    </div>

    <!-- 任务列表 -->
    <div v-if="loading" class="text-center py-12">
      <svg class="animate-spin h-8 w-8 text-primary-600 mx-auto" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
      </svg>
      <p class="mt-2 text-gray-500">加载中...</p>
    </div>

    <div v-else-if="!tasks.length" class="text-center py-12 bg-white rounded-lg border border-gray-200">
      <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
      </svg>
      <p class="mt-2 text-gray-500">暂无任务</p>
    </div>

    <div v-else class="space-y-3">
      <!-- 任务卡片 -->
      <div
        v-for="task in tasks"
        :key="task.task_id"
        class="bg-white rounded-lg border border-gray-200 p-4 hover:shadow-md transition-shadow"
      >
        <div class="flex items-start justify-between">
          <div class="flex items-start space-x-4 flex-1">
            <!-- 状态图标 -->
            <div
              class="w-12 h-12 rounded-full flex items-center justify-center flex-shrink-0"
              :class="statusBgClass(task.status)"
            >
              <svg v-if="task.status === 'completed'" class="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
              </svg>
              <svg v-else-if="task.status === 'failed'" class="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
              <svg v-else-if="task.status === 'cancelled'" class="w-6 h-6 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
              </svg>
              <svg v-else-if="task.status === 'processing'" class="w-6 h-6 text-blue-600 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
              </svg>
              <svg v-else class="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            
            <!-- 任务信息 -->
            <div class="flex-1 min-w-0">
              <div class="flex items-center space-x-2">
                <span
                  class="px-2 py-1 text-xs font-medium rounded-full"
                  :class="statusClass(task.status)"
                >
                  {{ statusText(task.status) }}
                </span>
                <span v-if="task.priority > 0" class="px-2 py-1 text-xs font-medium rounded-full bg-purple-100 text-purple-600">
                  优先级 {{ task.priority }}
                </span>
                <span v-if="task.is_parent" class="px-2 py-1 text-xs font-medium rounded-full bg-indigo-100 text-indigo-600">
                  父任务
                </span>
              </div>
              
              <div class="mt-2 space-y-1">
                <p class="text-sm font-medium text-gray-900 truncate">
                  {{ task.file_path.split('/').pop() || task.file_path }}
                </p>
                <p class="text-xs text-gray-500">
                  Engine: {{ task.engine }} · Task ID: {{ task.task_id.slice(0, 8) }}...
                </p>
                <p class="text-xs text-gray-500">
                  创建时间: {{ formatTime(task.created_at) }}
                  <span v-if="task.started_at"> · 开始时间: {{ formatTime(task.started_at) }}</span>
                  <span v-if="task.completed_at"> · 完成时间: {{ formatTime(task.completed_at) }}</span>
                </p>
                <p v-if="task.worker_id" class="text-xs text-gray-500">
                  Worker: {{ task.worker_id }}
                </p>
                <!-- 子任务进度 -->
                <p v-if="task.is_parent && task.child_count > 0" class="text-xs text-gray-500">
                  子任务: {{ task.child_completed }}/{{ task.child_count }}
                </p>
              </div>
              
              <!-- 错误信息 -->
              <div v-if="task.status === 'failed' && task.error_message" class="mt-2 p-2 bg-red-50 rounded text-xs text-red-600">
                {{ task.error_message }}
              </div>
            </div>
          </div>
          
          <!-- 操作按钮 -->
          <div class="flex items-center space-x-2 ml-4">
            <button
              v-if="task.status === 'pending' || task.status === 'processing'"
              @click="cancelTask(task.task_id)"
              class="px-3 py-1 text-sm border border-gray-300 rounded text-gray-700 hover:bg-gray-50"
            >
              取消
            </button>
            <button
              v-if="task.status === 'completed' && task.result_path"
              @click="viewResult(task)"
              class="px-3 py-1 text-sm bg-primary-600 text-white rounded hover:bg-primary-700"
            >
              查看结果
            </button>
            <button
              v-if="task.status === 'processing'"
              @click="refreshTask(task.task_id)"
              class="px-3 py-1 text-sm border border-gray-300 rounded text-gray-700 hover:bg-gray-50"
            >
              刷新
            </button>
          </div>
        </div>
        
        <!-- 进度条 -->
        <div v-if="task.status === 'processing'" class="mt-3">
          <div class="w-full bg-gray-200 rounded-full h-2">
            <div
              class="bg-blue-600 h-2 rounded-full transition-all duration-300"
              :style="{ width: '50%' }"
            ></div>
          </div>
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
        <button
          @click="nextPage"
          :disabled="offset + pageSize >= total"
          class="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 disabled:opacity-50"
        >
          下一页
        </button>
      </div>
    </div>

    <!-- 结果弹窗 -->
    <div v-if="resultTask" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div class="bg-white rounded-lg shadow-xl max-w-3xl w-full mx-4 max-h-[80vh] flex flex-col">
        <div class="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
          <h3 class="text-lg font-medium text-gray-900">识别结果</h3>
          <button @click="resultTask = null" class="text-gray-400 hover:text-gray-600">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div class="px-6 py-4 flex-1 overflow-auto">
          <pre class="whitespace-pre-wrap text-sm text-gray-800 bg-gray-50 p-4 rounded-md">{{ resultContent }}</pre>
        </div>
        <div class="px-6 py-4 border-t border-gray-200 flex justify-end space-x-3">
          <button
            @click="copyResult"
            class="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
          >
            复制
          </button>
          <button
            @click="resultTask = null"
            class="px-4 py-2 bg-gray-800 text-white rounded-md hover:bg-gray-900"
          >
            关闭
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { tasksApi, type TaskInfo, type TaskStats } from '@/api'

// 状态
const loading = ref(false)
const tasks = ref<TaskInfo[]>([])
const stats = ref<TaskStats>({ pending: 0, processing: 0, completed: 0, failed: 0, cancelled: 0 })
const total = ref(0)
const pageSize = 20
const offset = ref(0)
const filterStatus = ref('')
const resultTask = ref<TaskInfo | null>(null)
const resultContent = ref('')
let refreshInterval: number | null = null

// 状态样式
function statusClass(status: string) {
  const classes: Record<string, string> = {
    pending: 'bg-gray-100 text-gray-600',
    processing: 'bg-blue-100 text-blue-600',
    completed: 'bg-green-100 text-green-600',
    failed: 'bg-red-100 text-red-600',
    cancelled: 'bg-yellow-100 text-yellow-600',
  }
  return classes[status] || 'bg-gray-100 text-gray-600'
}

function statusBgClass(status: string) {
  const classes: Record<string, string> = {
    pending: 'bg-gray-100',
    processing: 'bg-blue-100',
    completed: 'bg-green-100',
    failed: 'bg-red-100',
    cancelled: 'bg-yellow-100',
  }
  return classes[status] || 'bg-gray-100'
}

function statusText(status: string) {
  const texts: Record<string, string> = {
    pending: '等待中',
    processing: '处理中',
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消',
  }
  return texts[status] || status
}

function formatTime(isoString: string | null) {
  if (!isoString) return '-'
  const date = new Date(isoString)
  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

// 获取任务列表
async function fetchTasks() {
  loading.value = true
  try {
    tasks.value = await tasksApi.list({
      status: filterStatus.value || undefined,
      limit: pageSize,
      offset: offset.value,
    })
    total.value = tasks.value.length
  } catch (e) {
    console.error('获取任务列表失败:', e)
  } finally {
    loading.value = false
  }
}

// 获取统计
async function fetchStats() {
  try {
    stats.value = await tasksApi.getStats()
  } catch (e) {
    console.error('获取统计失败:', e)
  }
}

// 刷新
async function refresh() {
  await Promise.all([fetchTasks(), fetchStats()])
}

// 分页
function prevPage() {
  if (offset.value >= pageSize) {
    offset.value -= pageSize
    fetchTasks()
  }
}

function nextPage() {
  offset.value += pageSize
  fetchTasks()
}

// 取消任务
async function cancelTask(taskId: string) {
  if (!confirm('确定要取消这个任务吗？')) return
  try {
    await tasksApi.cancel(taskId)
    await refresh()
  } catch (e) {
    console.error('取消任务失败:', e)
    alert('取消任务失败')
  }
}

// 刷新单个任务
async function refreshTask(taskId: string) {
  try {
    const task = await tasksApi.getStatus(taskId)
    const index = tasks.value.findIndex(t => t.task_id === taskId)
    if (index !== -1) {
      tasks.value[index] = task
    }
  } catch (e) {
    console.error('刷新任务失败:', e)
  }
}

// 查看结果
async function viewResult(task: TaskInfo) {
  resultTask.value = task
  // 简单显示任务信息，实际应从 result_path 读取
  resultContent.value = `任务 ID: ${task.task_id}\n文件: ${task.file_path}\n状态: ${statusText(task.status)}\n结果路径: ${task.result_path || '-'}`
}

// 复制结果
async function copyResult() {
  try {
    await navigator.clipboard.writeText(resultContent.value)
    alert('已复制到剪贴板')
  } catch {
    alert('复制失败')
  }
}

// 自动刷新
function startAutoRefresh() {
  refreshInterval = window.setInterval(() => {
    fetchTasks()
    fetchStats()
  }, 5000)
}

function stopAutoRefresh() {
  if (refreshInterval) {
    clearInterval(refreshInterval)
    refreshInterval = null
  }
}

// 生命周期
onMounted(() => {
  refresh()
  startAutoRefresh()
})

onUnmounted(() => {
  stopAutoRefresh()
})
</script>

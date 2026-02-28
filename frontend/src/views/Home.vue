<template>
  <div class="space-y-6">
    <!-- 页面标题 -->
    <div>
      <h1 class="text-2xl font-bold text-gray-900">OCR 识别</h1>
      <p class="mt-1 text-sm text-gray-500">上传图片或 PDF 文件进行 OCR 识别</p>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- 左侧：上传区域 -->
      <div class="space-y-4">
        <!-- 文件上传 -->
        <div
          class="border-2 border-dashed rounded-lg p-8 text-center transition-colors"
          :class="dragOver ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-primary-400'"
          @dragover.prevent="dragOver = true"
          @dragleave="dragOver = false"
          @drop.prevent="handleDrop"
        >
          <input
            ref="fileInput"
            type="file"
            class="hidden"
            accept="image/*,.pdf"
            @change="handleFileSelect"
          />
          <svg class="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
            <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
          <p class="mt-2 text-sm text-gray-600">
            拖拽文件到此处，或
            <button @click="() => $refs.fileInput?.click()" class="text-primary-600 hover:text-primary-700 font-medium">
              点击上传
            </button>
          </p>
          <p class="mt-1 text-xs text-gray-500">支持 PNG, JPG, GIF, BMP, WebP, PDF（最大 50MB）</p>
        </div>

        <!-- 已选文件 -->
        <div v-if="selectedFile" class="bg-white rounded-lg border border-gray-200 p-4">
          <div class="flex items-center justify-between">
            <div class="flex items-center space-x-3">
              <svg class="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <div>
                <p class="text-sm font-medium text-gray-900">{{ selectedFile.name }}</p>
                <p class="text-xs text-gray-500">{{ formatSize(selectedFile.size) }}</p>
              </div>
            </div>
            <button @click="clearFile" class="text-gray-400 hover:text-gray-600">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        <!-- 模型选择 -->
        <div class="bg-white rounded-lg border border-gray-200 p-4">
          <label class="block text-sm font-medium text-gray-700 mb-2">选择模型</label>
          <select
            v-model="selectedModel"
            class="w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
            :disabled="appStore.loading"
          >
            <option value="">-- 请选择模型 --</option>
            <option v-for="model in appStore.models" :key="model.name" :value="model.name">
              {{ model.name }} ({{ model.status }})
            </option>
          </select>
          <p v-if="!appStore.models.length" class="mt-2 text-sm text-yellow-600">
            暂无可用模型，请先在「模型管理」中注册模型
          </p>
        </div>

        <!-- 执行按钮 -->
        <button
          @click="executeOcr"
          :disabled="!canExecute || executing"
          class="w-full py-3 px-4 rounded-md text-white font-medium transition-colors"
          :class="canExecute && !executing ? 'bg-primary-600 hover:bg-primary-700' : 'bg-gray-300 cursor-not-allowed'"
        >
          <span v-if="executing" class="flex items-center justify-center">
            <svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            识别中...
          </span>
          <span v-else>开始识别</span>
        </button>
      </div>

      <!-- 右侧：结果展示 -->
      <div class="bg-white rounded-lg border border-gray-200">
        <!-- Tab 切换 -->
        <div class="border-b border-gray-200">
          <nav class="flex -mb-px">
            <button
              v-for="tab in ['text', 'markdown']"
              :key="tab"
              @click="activeTab = tab"
              class="px-4 py-3 text-sm font-medium border-b-2 transition-colors"
              :class="activeTab === tab ? 'border-primary-500 text-primary-600' : 'border-transparent text-gray-500 hover:text-gray-700'"
            >
              {{ tab === 'text' ? '文本' : 'Markdown' }}
            </button>
          </nav>
        </div>

        <!-- 结果内容 -->
        <div class="p-4 min-h-[300px]">
          <div v-if="result" class="space-y-4">
            <pre v-if="activeTab === 'text'" class="whitespace-pre-wrap text-sm text-gray-800 bg-gray-50 p-4 rounded-md overflow-auto max-h-[400px]">{{ result.text }}</pre>
            <div v-else class="prose prose-sm max-w-none" v-html="renderedMarkdown"></div>
          </div>
          <div v-else-if="error" class="text-red-600 text-sm">
            {{ error }}
          </div>
          <div v-else class="text-gray-400 text-center py-12">
            <svg class="mx-auto h-12 w-12 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <p class="mt-2">识别结果将显示在这里</p>
          </div>
        </div>

        <!-- 操作按钮 -->
        <div v-if="result" class="border-t border-gray-200 p-4 flex space-x-3">
          <button
            @click="copyResult"
            class="flex-1 py-2 px-4 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
          >
            复制
          </button>
          <button
            @click="downloadResult('txt')"
            class="flex-1 py-2 px-4 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
          >
            下载 TXT
          </button>
          <button
            @click="downloadResult('md')"
            class="flex-1 py-2 px-4 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
          >
            下载 MD
          </button>
          <button
            @click="downloadResult('json')"
            class="flex-1 py-2 px-4 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
          >
            下载 JSON
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useAppStore } from '@/stores/app'
import { filesApi, inferApi, historyApi, type InferResponse } from '@/api'

const appStore = useAppStore()

// 状态
const selectedFile = ref<File | null>(null)
const selectedModel = ref('')
const executing = ref(false)
const result = ref<InferResponse | null>(null)
const error = ref('')
const activeTab = ref('text')
const dragOver = ref(false)

// 计算属性
const canExecute = computed(() => selectedFile.value && selectedModel.value)

const renderedMarkdown = computed(() => {
  if (!result.value?.markdown) return ''
  return result.value.markdown
    .replace(/\n/g, '<br>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
})

// 方法
function handleDrop(e: DragEvent) {
  dragOver.value = false
  const files = e.dataTransfer?.files
  if (files?.length) {
    selectedFile.value = files[0]
  }
}

function handleFileSelect(e: Event) {
  const target = e.target as HTMLInputElement
  if (target.files?.length) {
    selectedFile.value = target.files[0]
  }
}

function clearFile() {
  selectedFile.value = null
  result.value = null
  error.value = ''
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

async function executeOcr() {
  if (!selectedFile.value || !selectedModel.value) return

  executing.value = true
  error.value = ''
  result.value = null

  try {
    // 1. 上传文件
    const fileInfo = await filesApi.upload(selectedFile.value)

    // 2. 执行推理
    result.value = await inferApi.infer({
      model: selectedModel.value,
      file_id: fileInfo.file_id,
      save_history: true,
    })
  } catch (e: any) {
    error.value = e.message
  } finally {
    executing.value = false
  }
}

async function copyResult() {
  if (!result.value) return
  try {
    await navigator.clipboard.writeText(result.value.text)
    alert('已复制到剪贴板')
  } catch {
    alert('复制失败')
  }
}

async function downloadResult(format: 'txt' | 'md' | 'json') {
  if (!result.value?.id) return
  try {
    const blob = await historyApi.download(result.value.id, format)
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

// 生命周期
onMounted(() => {
  appStore.fetchModels()
})
</script>
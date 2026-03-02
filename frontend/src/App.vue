<template>
  <div class="min-h-screen bg-gray-50">
    <!-- 导航栏 -->
    <nav class="bg-white shadow-sm border-b border-gray-200">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16">
          <div class="flex items-center">
            <router-link to="/" class="flex items-center space-x-2">
              <svg class="w-8 h-8 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span class="text-xl font-bold text-gray-900">DocMill</span>
            </router-link>
          </div>
          <div class="flex items-center space-x-4">
            <router-link
              to="/"
              class="px-3 py-2 rounded-md text-sm font-medium text-gray-700 hover:text-primary-600 hover:bg-gray-100 transition-colors"
              :class="{ 'text-primary-600 bg-gray-100': $route.path === '/' }"
            >
              OCR 识别
            </router-link>
            <router-link
              to="/models"
              class="px-3 py-2 rounded-md text-sm font-medium text-gray-700 hover:text-primary-600 hover:bg-gray-100 transition-colors"
              :class="{ 'text-primary-600 bg-gray-100': $route.path === '/models' }"
            >
              模型管理
            </router-link>
            <router-link
              to="/history"
              class="px-3 py-2 rounded-md text-sm font-medium text-gray-700 hover:text-primary-600 hover:bg-gray-100 transition-colors"
              :class="{ 'text-primary-600 bg-gray-100': $route.path === '/history' }"
            >
              历史记录
            </router-link>
            <router-link
              to="/gpu"
              class="px-3 py-2 rounded-md text-sm font-medium text-gray-700 hover:text-primary-600 hover:bg-gray-100 transition-colors"
              :class="{ 'text-primary-600 bg-gray-100': $route.path === '/gpu' }"
            >
              GPU监控
            </router-link>
          </div>
        </div>
      </div>
    </nav>

    <!-- 主内容区 -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <router-view />
    </main>

    <!-- 页脚 -->
    <footer class="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 py-4">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-sm text-gray-500">
        DocMill v0.2.0 - OCR 推理平台
      </div>
    </footer>
  </div>
</template>

<script setup lang="ts">
</script>
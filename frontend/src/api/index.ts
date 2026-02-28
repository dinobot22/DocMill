import axios from 'axios'

const api = axios.create({
  baseURL: '',
  timeout: 60000,
})

// 响应拦截器
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const message = error.response?.data?.detail || error.message || '请求失败'
    return Promise.reject(new Error(message))
  }
)

// ==================== 类型定义 ====================

export interface ModelInfo {
  name: string
  engine: string
  requires_vllm: boolean
  vllm_endpoint: string
  status: 'cold' | 'loading' | 'ready' | 'idle'
  is_loaded: boolean
  vram_estimate_mb: number
}

export interface EngineInfo {
  name: string
  requires_vllm: boolean
}

export interface RegisterModelRequest {
  name: string
  engine_name?: string
  vllm_config?: {
    model_path: string
    gpu_id?: number
    gpu_memory_utilization?: number
    max_model_len?: number
    tensor_parallel_size?: number
    trust_remote_code?: boolean
    extra_args?: string[]
  }
  engine_kwargs?: Record<string, any>
}

export interface InferResponse {
  id: string | null
  model: string
  text: string
  markdown: string
  structured: Record<string, any>
  metadata: Record<string, any>
}

export interface HistoryRecord {
  id: string
  model: string
  file_id: string
  filename: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  created_at: string
  completed_at: string | null
  result_text: string
  result_markdown: string
  result_structured: Record<string, any>
  error: string
}

export interface HistoryListResponse {
  total: number
  items: HistoryRecord[]
}

export interface FileInfo {
  file_id: string
  filename: string
  content_type: string
  size: number
  hash: string
  created_at: string
}

// ==================== 模型 API ====================

export const modelsApi = {
  list: () => api.get<ModelInfo[]>('/models').then((r) => r.data),

  get: (name: string) => api.get<ModelInfo>(`/models/${name}`).then((r) => r.data),

  register: (data: RegisterModelRequest) =>
    api.post<{ status: string; model: string }>('/models/register', data).then((r) => r.data),

  unregister: (name: string) =>
    api.post<{ status: string; model: string }>('/models/unregister', null, { params: { model: name } }).then((r) => r.data),

  load: (name: string) =>
    api.post<{ status: string; model: string }>(`/models/${name}/load`).then((r) => r.data),

  unload: (name: string) =>
    api.post<{ status: string; model: string }>(`/models/${name}/unload`).then((r) => r.data),

  listEngines: () => api.get<EngineInfo[]>('/models/engines/list').then((r) => r.data),
}

// ==================== 推理 API ====================

export const inferApi = {
  infer: (data: {
    model: string
    file_id?: string
    file_path?: string
    image_bytes?: string
    url?: string
    options?: Record<string, any>
    save_history?: boolean
  }) => api.post<InferResponse>('/infer', data).then((r) => r.data),

  inferWithModel: (modelName: string, data: Omit<Parameters<typeof inferApi.infer>[0], 'model'>) =>
    api.post<InferResponse>(`/infer/${modelName}`, data).then((r) => r.data),
}

// ==================== 历史 API ====================

export const historyApi = {
  list: (params?: { limit?: number; offset?: number; model?: string; status?: string }) =>
    api.get<HistoryListResponse>('/history', { params }).then((r) => r.data),

  get: (id: string) => api.get<HistoryRecord>(`/history/${id}`).then((r) => r.data),

  delete: (id: string) => api.delete<{ status: string; id: string }>(`/history/${id}`).then((r) => r.data),

  download: (id: string, format: 'txt' | 'md' | 'json' = 'txt') =>
    api.get(`/history/${id}/download`, { params: { format }, responseType: 'blob' }).then((r) => r.data),
}

// ==================== 文件 API ====================

export const filesApi = {
  upload: (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post<FileInfo>('/files/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then((r) => r.data)
  },

  list: (params?: { limit?: number; offset?: number }) =>
    api.get<{ items: FileInfo[] }>('/files', { params }).then((r) => r.data),

  get: (id: string) => api.get<FileInfo>(`/files/${id}`).then((r) => r.data),

  delete: (id: string) => api.delete<{ status: string; file_id: string }>(`/files/${id}`).then((r) => r.data),
}

// ==================== 健康检查 ====================

export const healthApi = {
  check: () => api.get<{ status: string; version: string }>('/health').then((r) => r.data),
}

export default api
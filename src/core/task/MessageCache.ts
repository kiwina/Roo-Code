import * as vscode from "vscode"
import * as path from "path"
import * as fs from "fs/promises"
import { ClineMessage } from "../../shared/types"

/**
 * LRU Cache for lazy loading messages to reduce memory retention
 * Implements the fix from ui-message-trace-2/retained-global-message/Task_ts_clineMessages_Backend.md
 */
export class MessageCache {
	private cache = new Map<number, ClineMessage>()
	private messageIds: number[] = []
	private maxCacheSize: number = 100
	private taskId: string
	private globalStoragePath: string

	constructor(taskId: string, globalStoragePath: string, maxCacheSize: number = 100) {
		this.taskId = taskId
		this.globalStoragePath = globalStoragePath
		this.maxCacheSize = maxCacheSize
	}

	/**
	 * Initialize the cache by loading message metadata (IDs/timestamps only)
	 */
	async initialize(): Promise<void> {
		try {
			const messagesPath = path.join(this.globalStoragePath, `task_${this.taskId}`, "messages.json")
			const messagesData = await fs.readFile(messagesPath, "utf8")
			const messages: ClineMessage[] = JSON.parse(messagesData)

			// Store only message IDs/indices, not full message content
			this.messageIds = messages.map((_, index) => index)
		} catch (error) {
			// If no messages file exists yet, start with empty array
			this.messageIds = []
		}
	}

	/**
	 * Get a message by index with lazy loading
	 */
	async getClineMessage(index: number): Promise<ClineMessage | undefined> {
		if (index < 0 || index >= this.messageIds.length) {
			return undefined
		}

		// Check cache first
		if (this.cache.has(index)) {
			// Move to end (most recently used)
			const message = this.cache.get(index)!
			this.cache.delete(index)
			this.cache.set(index, message)
			return message
		}

		// Load message on-demand from storage
		const message = await this.loadMessageByIndex(index)
		if (message) {
			// Implement LRU eviction if cache grows too large
			if (this.cache.size >= this.maxCacheSize) {
				const firstKey = this.cache.keys().next().value
				this.cache.delete(firstKey)
			}

			this.cache.set(index, message)
		}

		return message
	}

	/**
	 * Get messages in a range with pagination
	 */
	async getClineMessages(startIndex = 0, limit = 50): Promise<ClineMessage[]> {
		const endIndex = Math.min(startIndex + limit, this.messageIds.length)
		const messages: ClineMessage[] = []

		for (let i = startIndex; i < endIndex; i++) {
			const message = await this.getClineMessage(i)
			if (message) {
				messages.push(message)
			}
		}

		return messages
	}

	/**
	 * Get total message count without loading messages
	 */
	getMessageCount(): number {
		return this.messageIds.length
	}

	/**
	 * Add a new message to the cache and storage
	 */
	async addMessage(message: ClineMessage): Promise<void> {
		const index = this.messageIds.length
		this.messageIds.push(index)

		// Add to cache if there's room, or evict LRU
		if (this.cache.size >= this.maxCacheSize) {
			const firstKey = this.cache.keys().next().value
			this.cache.delete(firstKey)
		}

		this.cache.set(index, message)

		// Persist to storage
		await this.saveMessageToStorage(message, index)
	}

	/**
	 * Clear cache to free memory
	 */
	clearCache(): void {
		this.cache.clear()
	}

	/**
	 * Get cache stats for debugging
	 */
	getCacheStats(): { size: number; maxSize: number; totalMessages: number } {
		return {
			size: this.cache.size,
			maxSize: this.maxCacheSize,
			totalMessages: this.messageIds.length,
		}
	}

	/**
	 * Load a specific message by index from storage
	 */
	private async loadMessageByIndex(index: number): Promise<ClineMessage | undefined> {
		try {
			const messagesPath = path.join(this.globalStoragePath, `task_${this.taskId}`, "messages.json")
			const messagesData = await fs.readFile(messagesPath, "utf8")
			const messages: ClineMessage[] = JSON.parse(messagesData)

			return messages[index]
		} catch (error) {
			console.error(`[MessageCache] Failed to load message at index ${index}:`, error)
			return undefined
		}
	}

	/**
	 * Save a message to storage (for new messages)
	 */
	private async saveMessageToStorage(message: ClineMessage, index: number): Promise<void> {
		try {
			const taskDir = path.join(this.globalStoragePath, `task_${this.taskId}`)
			const messagesPath = path.join(taskDir, "messages.json")

			// Ensure directory exists
			await fs.mkdir(taskDir, { recursive: true })

			// Load existing messages
			let messages: ClineMessage[] = []
			try {
				const messagesData = await fs.readFile(messagesPath, "utf8")
				messages = JSON.parse(messagesData)
			} catch {
				// File doesn't exist yet, start with empty array
			}

			// Add/update message at index
			messages[index] = message

			// Save back to storage
			await fs.writeFile(messagesPath, JSON.stringify(messages, null, 2))
		} catch (error) {
			console.error(`[MessageCache] Failed to save message at index ${index}:`, error)
		}
	}
}

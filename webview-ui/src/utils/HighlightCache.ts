/**
 * Shared highlighting cache to prevent duplicate Shiki HTML generation
 * Implements the fix from ui-message-trace/duplicated-highlighted-code/CodeBlock_tsx_Shiki_State.md
 */
export class HighlightCache {
	private static cache = new Map<string, string>()
	private static accessOrder = new Map<string, number>() // Track access order for LRU
	private static accessCounter = 0
	private static maxSize = 500 // Limit cache growth

	/**
	 * Get highlighted HTML from cache if available
	 */
	static getHighlighted(source: string, language: string, theme: string): string | null {
		const key = `${language}:${theme}:${this.hashCode(source)}`
		const cached = this.cache.get(key)

		if (cached) {
			// Update access order for LRU
			this.accessOrder.set(key, ++this.accessCounter)
			return cached
		}

		return null
	}

	/**
	 * Store highlighted HTML in cache
	 */
	static setHighlighted(source: string, language: string, theme: string, html: string): void {
		const key = `${language}:${theme}:${this.hashCode(source)}`

		// Implement LRU eviction - remove oldest entries
		if (this.cache.size >= this.maxSize) {
			const entriesToRemove = Math.max(1, Math.floor(this.maxSize * 0.1)) // Remove 10% when full

			// Sort by access order and remove least recently used
			const sortedByAccess = Array.from(this.accessOrder.entries())
				.sort(([, a], [, b]) => a - b)
				.slice(0, entriesToRemove)

			for (const [keyToRemove] of sortedByAccess) {
				this.cache.delete(keyToRemove)
				this.accessOrder.delete(keyToRemove)
			}
		}

		this.cache.set(key, html)
		this.accessOrder.set(key, ++this.accessCounter)
	}

	/**
	 * Clear the entire cache (useful for theme changes)
	 */
	static clear(): void {
		this.cache.clear()
		this.accessOrder.clear()
		this.accessCounter = 0
	}

	/**
	 * Get cache statistics for debugging
	 */
	static getStats(): { size: number; maxSize: number } {
		return {
			size: this.cache.size,
			maxSize: this.maxSize,
		}
	}

	/**
	 * Simple hash function for cache keys
	 */
	private static hashCode(str: string): string {
		let hash = 0
		for (let i = 0; i < str.length; i++) {
			const char = str.charCodeAt(i)
			hash = (hash << 5) - hash + char
			hash = hash & hash // Convert to 32-bit integer
		}
		return hash.toString(36)
	}
}

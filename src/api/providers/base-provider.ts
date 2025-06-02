import { Anthropic } from "@anthropic-ai/sdk"

import type { ModelInfo } from "@roo-code/types"

import type { ApiHandler, ApiHandlerCreateMessageMetadata } from "../index"
import { ApiStream } from "../transform/stream"
import { countTokens } from "../../utils/countTokens"

/**
 * Base class for API providers that implements common functionality.
 */
export abstract class BaseProvider implements ApiHandler {
	protected client: any // Added protected client field

	abstract createMessage(
		systemPrompt: string,
		messages: Anthropic.Messages.MessageParam[],
		metadata?: ApiHandlerCreateMessageMetadata,
	): ApiStream

	abstract getModel(): { id: string; info: ModelInfo }

	/**
	 * Default token counting implementation using tiktoken.
	 * Providers can override this to use their native token counting endpoints.
	 *
	 * @param content The content to count tokens for
	 * @returns A promise resolving to the token count
	 */
	async countTokens(content: Anthropic.Messages.ContentBlockParam[]): Promise<number> {
		if (content.length === 0) {
			return 0
		}

		return countTokens(content, { useWorker: true })
	}

	/**
	 * Disposes of any resources held by the provider.
	 * Attempts common disposal methods on the client if it exists.
	 */
	public dispose(): void {
		if (this.client) {
			// Try common disposal methods that SDKs might have
			if (typeof (this.client as any).close === "function") {
				;(this.client as any).close()
			} else if (typeof (this.client as any).destroy === "function") {
				;(this.client as any).destroy()
			} else if (typeof (this.client as any).dispose === "function") {
				;(this.client as any).dispose()
			}
			// Clear the reference
			this.client = undefined
		}
	}
}

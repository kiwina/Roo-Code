import type { Anthropic } from "@anthropic-ai/sdk"
import {
	GoogleGenAI,
	type GenerateContentResponseUsageMetadata,
	type GenerateContentParameters,
	type GenerateContentConfig,
	CountTokensParameters,
} from "@google/genai"
import type { JWTInput } from "google-auth-library"

import { type ModelInfo, type VertexModelId, vertexDefaultModelId, vertexModels } from "@roo-code/types"

import type { ApiHandlerOptions } from "../../shared/api"
import { safeJsonParse } from "../../shared/safeJsonParse"

import { convertAnthropicContentToGemini, convertAnthropicMessageToGemini } from "../transform/gemini-format"
import type { ApiStream } from "../transform/stream"

import type { SingleCompletionHandler, ApiHandlerCreateMessageMetadata } from "../index"
import { BaseProvider } from "./base-provider"

export class VertexHandler extends BaseProvider implements SingleCompletionHandler {
	protected options: ApiHandlerOptions
	private client: GoogleGenAI

	constructor(options: ApiHandlerOptions) {
		super()
		this.options = options

		const project = this.options.vertexProjectId ?? "not-provided"
		const location = this.options.vertexRegion ?? "not-provided"

		if (this.options.vertexJsonCredentials) {
			this.client = new GoogleGenAI({
				vertexai: true,
				project,
				location,
				googleAuthOptions: {
					credentials: safeJsonParse<JWTInput>(this.options.vertexJsonCredentials, undefined),
				},
			})
		} else if (this.options.vertexKeyFile) {
			this.client = new GoogleGenAI({
				vertexai: true,
				project,
				location,
				googleAuthOptions: { keyFile: this.options.vertexKeyFile },
			})
		} else if (this.options.vertexApiKey) {
			this.client = new GoogleGenAI({
				vertexai: true,
				apiKey: this.options.vertexApiKey,
				apiVersion: "v1",
			})
		} else {
			this.client = new GoogleGenAI({ vertexai: true, project, location })
		}
	}

	async *createMessage(
		systemInstruction: string,
		messages: Anthropic.Messages.MessageParam[],
		metadata?: ApiHandlerCreateMessageMetadata,
	): ApiStream {
		const { id: model, thinkingConfig, maxOutputTokens, info } = this.getModel()

		const contents = messages.map(convertAnthropicMessageToGemini)

		const config: GenerateContentConfig = {
			systemInstruction,
			thinkingConfig,
			maxOutputTokens,
			temperature: this.options.modelTemperature ?? 0,
		}

		const params: GenerateContentParameters = { model, contents, config }

		if (this.options.vertexApiKey && this.options.vertexProjectId) {
			params.model = `projects/${this.options.vertexProjectId}/locations/${this.options.vertexRegion}/publishers/google/models/${model}`
		}

		const result = await this.client.models.generateContentStream(params)

		let lastUsageMetadata: GenerateContentResponseUsageMetadata | undefined

		for await (const chunk of result) {
			if (chunk.text) {
				yield { type: "text", text: chunk.text }
			}

			if (chunk.usageMetadata) {
				lastUsageMetadata = chunk.usageMetadata
			}
		}

		if (lastUsageMetadata) {
			const inputTokens = lastUsageMetadata.promptTokenCount ?? 0
			const outputTokens = lastUsageMetadata.candidatesTokenCount ?? 0
			const cacheReadTokens = lastUsageMetadata.cachedContentTokenCount
			const reasoningTokens = lastUsageMetadata.thoughtsTokenCount

			yield {
				type: "usage",
				inputTokens,
				outputTokens,
				cacheReadTokens,
				reasoningTokens,
				totalCost: this.calculateCost({ info, inputTokens, outputTokens, cacheReadTokens }),
			}
		}
	}

	getModel() {
		let id = this.options.apiModelId ?? vertexDefaultModelId
		let info: ModelInfo = vertexModels[id as VertexModelId]

		if (id?.endsWith(":thinking")) {
			id = id.slice(0, -":thinking".length) as VertexModelId

			if (vertexModels[id as VertexModelId]) {
				info = vertexModels[id as VertexModelId]

				return {
					id,
					info,
					thinkingConfig: this.options.modelMaxThinkingTokens
						? { thinkingBudget: this.options.modelMaxThinkingTokens }
						: undefined,
					maxOutputTokens: this.options.modelMaxTokens ?? info.maxTokens ?? undefined,
				}
			}
		}

		if (!info) {
			id = vertexDefaultModelId
			info = vertexModels[vertexDefaultModelId]
		}

		return { id, info }
	}

	async completePrompt(prompt: string): Promise<string> {
		try {
			const { id: model } = this.getModel()

			const result = await this.client.models.generateContent({
				model,
				contents: [{ role: "user", parts: [{ text: prompt }] }],
				config: {
					temperature: this.options.modelTemperature ?? 0,
				},
			})

			return result.text ?? ""
		} catch (error) {
			if (error instanceof Error) {
				throw new Error(`Vertex completion error: ${error.message}`)
			}

			throw error
		}
	}

	override async countTokens(content: Array<Anthropic.Messages.ContentBlockParam>): Promise<number> {
		try {
			const { id: model } = this.getModel()

			const params: CountTokensParameters = {
				model,
				contents: convertAnthropicContentToGemini(content),
			}

			if (this.options.vertexApiKey && this.options.vertexProjectId) {
				params.model = `projects/${this.options.vertexProjectId}/locations/${this.options.vertexRegion}/publishers/google/models/${model}`
			}

			const response = await this.client.models.countTokens(params)

			if (response.totalTokens === undefined) {
				console.warn("Vertex token counting returned undefined, using fallback")
				return super.countTokens(content)
			}

			return response.totalTokens
		} catch (error) {
			console.warn("Vertex token counting failed, using fallback", error)
			return super.countTokens(content)
		}
	}

	public calculateCost({
		info,
		inputTokens,
		outputTokens,
		cacheReadTokens = 0,
	}: {
		info: ModelInfo
		inputTokens: number
		outputTokens: number
		cacheReadTokens?: number
	}) {
		if (!info.inputPrice || !info.outputPrice || !info.cacheReadsPrice) {
			return undefined
		}

		let inputPrice = info.inputPrice
		let outputPrice = info.outputPrice
		let cacheReadsPrice = info.cacheReadsPrice

		// If there's tiered pricing then adjust the input and output token prices
		// based on the input tokens used.
		if (info.tiers) {
			const tier = info.tiers.find((tier) => inputTokens <= tier.contextWindow)

			if (tier) {
				inputPrice = tier.inputPrice ?? inputPrice
				outputPrice = tier.outputPrice ?? outputPrice
				cacheReadsPrice = tier.cacheReadsPrice ?? cacheReadsPrice
			}
		}

		// Subtract the cached input tokens from the total input tokens.
		const uncachedInputTokens = inputTokens - cacheReadTokens

		let cacheReadCost = cacheReadTokens > 0 ? cacheReadsPrice * (cacheReadTokens / 1_000_000) : 0

		const inputTokensCost = inputPrice * (uncachedInputTokens / 1_000_000)
		const outputTokensCost = outputPrice * (outputTokens / 1_000_000)
		const totalCost = inputTokensCost + outputTokensCost + cacheReadCost

		const trace: Record<string, { price: number; tokens: number; cost: number }> = {
			input: { price: inputPrice, tokens: uncachedInputTokens, cost: inputTokensCost },
			output: { price: outputPrice, tokens: outputTokens, cost: outputTokensCost },
		}

		if (cacheReadTokens > 0) {
			trace.cacheRead = { price: cacheReadsPrice, tokens: cacheReadTokens, cost: cacheReadCost }
		}

		// console.log(`[VertexHandler] calculateCost -> ${totalCost}`, trace)

		return totalCost
	}

	public destruct() {}
}

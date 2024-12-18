const chromadb = require("chromadb");
const { toChunks, getEmbeddingEngineSelection } = require("../../helpers");
const { TextSplitter } = require("../../TextSplitter");
const { SystemSettings } = require("../../../models/systemSettings");
const { storeVectorResult, cachedVectorInformation } = require("../../files");
const { v4: uuidv4 } = require("uuid");
const { sourceIdentifier } = require("../../chats");

/**
 * ChromaDB Client connection object
 * @typedef {import('chromadb').Client} ChromaClient
 */

const ChromaDb = {
    uri: process.env.STORAGE_DIR || "./storage/",
    name: "ChromaDb",

    /** @returns {Promise<{client: ChromaClient}>} */
    connect: async function () {
        if (process.env.VECTOR_DB !== "chromadb")
            throw new Error("ChromaDb::Invalid ENV settings");

        const client = await chromadb.connect({ path: this.uri });
        return { client };
    },
    distanceToSimilarity: function (distance = null) {
        if (distance === null || typeof distance !== "number") return 0.0;
        if (distance >= 1.0) return 1;
        if (distance <= 0) return 0;
        return 1 - distance;
    },
    heartbeat: async function () {
        await this.connect();
        return { heartbeat: Number(new Date()) };
    },
    collections: async function () {
        const { client } = await this.connect();
        return await client.listCollections();
    },
    totalVectors: async function () {
        const { client } = await this.connect();
        const collections = await client.listCollections();
        let count = 0;
        for (const collectionName of collections) {
            const collection = await client.getCollection(collectionName);
            count += await collection.count();
        }
        return count;
    },
    namespaceCount: async function (_namespace = null) {
        const { client } = await this.connect();
        const exists = await this.namespaceExists(client, _namespace);
        if (!exists) return 0;

        const collection = await client.getCollection(_namespace);
        return (await collection.count()) || 0;
    },
    /**
     * Performs a SimilaritySearch on a given ChromaDB namespace.
     * @param {ChromaClient} client
     * @param {string} namespace
     * @param {number[]} queryVector
     * @param {number} similarityThreshold
     * @param {number} topN
     * @param {string[]} filterIdentifiers
     * @returns
     */
    similarityResponse: async function (
        client,
        namespace,
        queryVector,
        similarityThreshold = 0.25,
        topN = 4,
        filterIdentifiers = []
    ) {
        const collection = await client.getCollection(namespace);
        const results = await collection.query({
            queryVector,
            nResults: topN,
        });

        const result = {
            contextTexts: [],
            sourceDocuments: [],
            scores: [],
        };

        results.forEach((item) => {
            const { vector: _, distance, ...rest } = item;
            const similarity = this.distanceToSimilarity(distance);

            if (similarity < similarityThreshold) return;
            if (filterIdentifiers.includes(sourceIdentifier(rest))) {
                console.log(
                    "ChromaDB: A source was filtered from context as its parent document is pinned."
                );
                return;
            }

            result.contextTexts.push(rest.text);
            result.sourceDocuments.push({ ...rest, score: similarity });
            result.scores.push(similarity);
        });

        return result;
    },
    namespace: async function (client, namespace = null) {
        if (!namespace) throw new Error("No namespace value provided.");
        const collection = await client.getCollection(namespace).catch(() => false);
        if (!collection) return null;

        return collection;
    },
    updateOrCreateCollection: async function (client, data = [], namespace) {
        const hasNamespace = await this.hasNamespace(namespace);
        if (hasNamespace) {
            const collection = await client.getCollection(namespace);
            await collection.upsert(data);
            return true;
        }

        await client.createCollection(namespace);
        const collection = await client.getCollection(namespace);
        await collection.add(data);
        return true;
    },
    hasNamespace: async function (namespace = null) {
        if (!namespace) return false;
        const { client } = await this.connect();
        const exists = await this.namespaceExists(client, namespace);
        return exists;
    },
    namespaceExists: async function (client, namespace = null) {
        if (!namespace) throw new Error("No namespace value provided.");
        const collections = await client.listCollections();
        return collections.includes(namespace);
    },
    deleteVectorsInNamespace: async function (client, namespace = null) {
        const collection = await client.getCollection(namespace);
        await collection.clear();
        return true;
    },
    deleteDocumentFromNamespace: async function (namespace, docId) {
        const { client } = await this.connect();
        const exists = await this.namespaceExists(client, namespace);
        if (!exists) {
            console.error(
                `ChromaDB:deleteDocumentFromNamespace - namespace ${namespace} does not exist.`
            );
            return;
        }

        const { DocumentVectors } = require("../../../models/vectors");
        const collection = await client.getCollection(namespace);
        const vectorIds = (await DocumentVectors.where({ docId })).map(
            (record) => record.vectorId
        );

        if (vectorIds.length === 0) return;
        await collection.delete({ ids: vectorIds });
        return true;
    },
    addDocumentToNamespace: async function (
        namespace,
        documentData = {},
        fullFilePath = null,
        skipCache = false
    ) {
        const { DocumentVectors } = require("../../../models/vectors");
        try {
            const { pageContent, docId, ...metadata } = documentData;
            if (!pageContent || pageContent.length == 0) return false;

            console.log("Adding new vectorized document into namespace", namespace);
            if (!skipCache) {
                const cacheResult = await cachedVectorInformation(fullFilePath);
                if (cacheResult.exists) {
                    const { client } = await this.connect();
                    const { chunks } = cacheResult;
                    const documentVectors = [];
                    const submissions = [];

                    for (const chunk of chunks) {
                        chunk.forEach((chunk) => {
                            const id = uuidv4();
                            const { id: _id, ...metadata } = chunk.metadata;
                            documentVectors.push({ docId, vectorId: id });
                            submissions.push({ id: id, vector: chunk.values, ...metadata });
                        });
                    }

                    await this.updateOrCreateCollection(client, submissions, namespace);
                    await DocumentVectors.bulkInsert(documentVectors);
                    return { vectorized: true, error: null };
                }
            }

            const EmbedderEngine = getEmbeddingEngineSelection();
            const textSplitter = new TextSplitter({
                chunkSize: TextSplitter.determineMaxChunkSize(
                    await SystemSettings.getValueOrFallback({
                        label: "text_splitter_chunk_size",
                    }),
                    EmbedderEngine?.embeddingMaxChunkLength
                ),
                chunkOverlap: await SystemSettings.getValueOrFallback(
                    { label: "text_splitter_chunk_overlap" },
                    20
                ),
                chunkHeaderMeta: TextSplitter.buildHeaderMeta(metadata),
            });
            const textChunks = await textSplitter.splitText(pageContent);

            console.log("Chunks created from document:", textChunks.length);
            const documentVectors = [];
            const vectors = [];
            const submissions = [];
            const vectorValues = await EmbedderEngine.embedChunks(textChunks);

            if (!!vectorValues && vectorValues.length > 0) {
                for (const [i, vector] of vectorValues.entries()) {
                    const vectorRecord = {
                        id: uuidv4(),
                        values: vector,
                        metadata: { ...metadata, text: textChunks[i] },
                    };

                    vectors.push(vectorRecord);
                    submissions.push({
                        ...vectorRecord.metadata,
                        id: vectorRecord.id,
                        vector: vectorRecord.values,
                    });
                    documentVectors.push({ docId, vectorId: vectorRecord.id });
                }
            } else {
                throw new Error(
                    "Could not embed document chunks! This document will not be recorded."
                );
            }

            if (vectors.length > 0) {
                const chunks = [];
                for (const chunk of toChunks(vectors, 500)) chunks.push(chunk);

                console.log("Inserting vectorized chunks into ChromaDB collection.");
                const { client } = await this.connect();
                await this.updateOrCreateCollection(client, submissions, namespace);
                await storeVectorResult(chunks, fullFilePath);
            }

            await DocumentVectors.bulkInsert(documentVectors);
            return { vectorized: true, error: null };
        } catch (e) {
            console.error("addDocumentToNamespace", e.message);
            return { vectorized: false, error: e.message };
        }
    },
    performSimilaritySearch: async function (params) {
        const {
            namespace = null,
            input = "",
            LLMConnector = null,
            similarityThreshold = 0.25,
            topN = 4,
            filterIdentifiers = [],
        } = params;

        if (!namespace || !input || !LLMConnector)
            throw new Error("Invalid request to performSimilaritySearch.");

        const { client } = await this.connect();
        if (!(await this.namespaceExists(client, namespace))) {
            return {
                contextTexts: [],
                sources: [],
                message: "Invalid query - no documents found for workspace!",
            };
        }

        const queryVector = await LLMConnector.embedTextInput(input);
        const { contextTexts, sourceDocuments } = await this.similarityResponse(
            client,
            namespace,
            queryVector,
            similarityThreshold,
            topN,
            filterIdentifiers
        );

        const sources = sourceDocuments.map((metadata, i) => {
            return { metadata: { ...metadata, text: contextTexts[i] } };
        });
        return {
            contextTexts,
            sources: this.curateSources(sources),
            message: false,
        };
    },
    "namespace-stats": async function (reqBody = {}) {
        const { namespace = null } = reqBody;
        if (!namespace) throw new Error("namespace required");
        const { client } = await this.connect();
        if (!(await this.namespaceExists(client, namespace)))
            throw new Error("Namespace by that name does not exist.");
        const stats = await this.namespace(client, namespace);
        return stats
            ? stats
            : { message: "No stats were able to be fetched from DB for namespace" };
    },
    "delete-namespace": async function (reqBody = {}) {
        const { namespace = null } = reqBody;
        const { client } = await this.connect();
        if (!(await this.namespaceExists(client, namespace)))
            throw new Error("Namespace by that name does not exist.");

        await this.deleteVectorsInNamespace(client, namespace);
        return {
            message: `Namespace ${namespace} was deleted.`,
        };
    },
    reset: async function () {
        const { client } = await this.connect();
        const fs = require("fs");
        fs.rm(`${this.uri}`, { recursive: true }, () => null);
        return { reset: true };
    },
    curateSources: function (sources = []) {
        const documents = [];
        for (const source of sources) {
            const { text, vector: _v, _distance: _d, ...rest } = source;
            const metadata = rest.hasOwnProperty("metadata") ? rest.metadata : rest;
            if (Object.keys(metadata).length > 0) {
                documents.push({
                    ...metadata,
                    ...(text ? { text } : {}),
                });
            }
        }

        return documents;
    },
};

module.exports.ChromaDb = ChromaDb;
